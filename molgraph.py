import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'
import rdkit.Chem
import dgl
import numpy as np
from tensorflow import Variable,cast

def create_segment(Max_atoms, total_num_atoms):
    seg = []
    for i, num_atoms_each_mol in enumerate(total_num_atoms):
        if num_atoms_each_mol > Max_atoms:
            print("Error: # of atoms in a molecule > Max_atoms ")
            sys.exit()
        segment_one_mol = [i] * num_atoms_each_mol
        segment_one_mol += [-1] * (Max_atoms - num_atoms_each_mol)
        seg += segment_one_mol
    return tf.constant(seg)

def atom_features(mol, Max_atoms, explicit_Hs = False):
    features = []

    if explicit_Hs:
        feature_pos = ['C','O','H', 0, 1, 2, 3, 4, 'True','False', 'Aro_True', 'Aro_False']
    else:
        feature_pos = ['C','O', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'True','False', 'Aro_True', 'Aro_False']
    for atom in mol.GetAtoms():
        # Symbols - Degrees - 5 + Total_Hs - IsInring - IsAromatic
        # (2 + 5 + 5 + 2 + 2)
        feature = np.zeros((len(feature_pos)))
        feature[ feature_pos.index(atom.GetSymbol()) ] = 1
        feature[ feature_pos.index(atom.GetDegree()) ] = 1

        if not explicit_Hs:
            feature[ feature_pos.index(atom.GetTotalNumHs() + 5) ] = 1

        feature[ feature_pos.index(str(atom.IsInRing())) ] = 1
        feature[ feature_pos.index( 'Aro_' + str(atom.GetIsAromatic())) ] = 1
        features.append(np.asarray(feature))   
    #zero-padding
    for i in range(Max_atoms - len(features)):
        feature = np.zeros((len(feature_pos)))
        features.append(feature)
    return features

def dgl_molgraph(data, Max_atoms, device, explicit_Hs = False):
    src_ids = []
    dst_ids = []
    all_atom_features = []

    curr_index = 0
    for _, row in data.iterrows():
        mol = rdkit.Chem.MolFromSmiles(row['smiles'])

        if explicit_Hs:
            mol = rdkit.Chem.AddHs(mol)

        for atom_ind in range(len(mol.GetAtoms())):
            src_ids.append(curr_index + atom_ind)
            dst_ids.append(curr_index + atom_ind)

        for bond in mol.GetBonds():
            src = bond.GetBeginAtom().GetIdx()
            dst = bond.GetEndAtom().GetIdx()
            src_ids.append(curr_index + src)
            dst_ids.append(curr_index + dst)
            src_ids.append(curr_index + dst)
            dst_ids.append(curr_index + src)
        curr_index += Max_atoms
        all_atom_features += atom_features(mol, Max_atoms, explicit_Hs) 

    all_atom_features = cast( Variable(all_atom_features), dtype=tf.float32)

    g = dgl.graph((src_ids, dst_ids), num_nodes = len(data) * Max_atoms).to(device)
    g.ndata['feat'] = all_atom_features
    return g

def dgl_molgraph_one_molecule(smiles, Max_atoms, device, explicit_Hs = False):
    src_ids = []
    dst_ids = []
    all_atom_features = []

    curr_index = 0

    mol = rdkit.Chem.MolFromSmiles(smiles)

    if explicit_Hs:
        mol = rdkit.Chem.AddHs(mol)

    for atom_ind in range(len(mol.GetAtoms())):
        src_ids.append(atom_ind)
        dst_ids.append(atom_ind)

    for bond in mol.GetBonds():
        src = bond.GetBeginAtom().GetIdx()
        dst = bond.GetEndAtom().GetIdx()
        src_ids.append(src)
        dst_ids.append(dst)
        src_ids.append(dst)
        dst_ids.append(src)

    Atom_features = cast( Variable(atom_features(mol, Max_atoms, explicit_Hs)), dtype=tf.float32 )

    g = dgl.graph((src_ids, dst_ids), num_nodes = Max_atoms).to(device)
    g.ndata['feat'] = Atom_features

    return g

def single_mol_from_supergraph(Graphs, H, Attention, Max_atoms):
    Num_molecules = int(H.shape[0] / Max_atoms)
    src_indices, dst_indices, edge_ids = Graphs.all_edges('all')

    mol_indices_src_nodes = [ int(x / Max_atoms) for x in src_indices.numpy() ]
    mol_indices_dst_nodes = [ int(x / Max_atoms) for x in dst_indices.numpy() ]


    if mol_indices_src_nodes != mol_indices_dst_nodes:
        print("!!!!!!!!")
        sys.exit()

    atom_indices_src_nodes = [ x % Max_atoms for x in src_indices.numpy() ]
    atom_indices_dst_nodes = [ x % Max_atoms for x in dst_indices.numpy() ]

    mol_indices_edge_ids = {}
    for i, edge_id in enumerate(edge_ids.numpy()):
        mol_indices_edge_ids[ edge_id ] = mol_indices_src_nodes[i]

    H_per_each_mol = np.reshape(H.numpy(), (Num_molecules, -1, H.shape[-1]) )

    A = Attention.numpy().reshape(-1)

    A_per_each_mol = []
    src_per_each_mol = []
    dst_per_each_mol = []
    for i in range(Num_molecules):
        A_per_each_mol.append([])
        src_per_each_mol.append([])
        dst_per_each_mol.append([])

    for i, mol_ind in enumerate(mol_indices_src_nodes):
        src_per_each_mol[mol_ind].append( atom_indices_src_nodes[i] )
        dst_per_each_mol[mol_ind].append( atom_indices_dst_nodes[i] )

    for i, edge_id in enumerate(edge_ids.numpy()):
        mol_index_this_edge = mol_indices_edge_ids[ edge_id ]
        A_per_each_mol[mol_index_this_edge].append(A[i])

    return A_per_each_mol, src_per_each_mol, dst_per_each_mol

