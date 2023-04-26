import pandas as pd
import tensorflow as tf
import rdkit
from argparse import ArgumentParser
from collections import Counter
import numpy as np
from molgraph import * 
from dgl import batch


def read_hov_model():
    atom_feat_dim = 16
    num_heads = 5 
    num_layers = 5 
    num_out_heads = 1

    heads = ([num_heads] * num_layers) + [num_out_heads]

    from gnn import GAT_unc
    model = GAT_unc(num_layers=5,
                in_dim=atom_feat_dim,
                num_hidden=32,
                num_classes=32,
                heads=heads,
                activation=tf.nn.relu,
                feat_drop=0.0,
                attn_drop=0.0,
                negative_slope=0.2,
                residual=True,
                equation='')
    model.load_model('best_211007/my_model')
    print('HoV model loaded')
    print('Test whether the model was loaded properly - HoV prediction')
    ####################################
    df = pd.read_csv('data/Data_211005.csv')
    pd.DataFrame(df[['smiles','Train/Valid/Test','temperature','HoV (kJ/mol)']]).to_csv('molecules_to_predict.csv', index=False)

    device ='/gpu:0'
    data = pd.read_csv('molecules_to_predict.csv')
    data['total_atoms'] = [ rdkit.Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in data.smiles]

    parser = ArgumentParser()
    parser.add_argument('-predict', action="store_true", default=True)
    parser.add_argument('-K_fold', action='store_true', default=False)
    parser.add_argument('-maxatoms', type=int, default=64)
    parser.add_argument('-lr', type=float, default=5.0e-4)
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batchsize', type=int, default=256)
    parser.add_argument('-layers', type=int, default=5)
    parser.add_argument('-heads', type=int, default=5)
    parser.add_argument('-residcon', action="store_true", default=True)
    parser.add_argument('-explicitH', action="store_true", default=False)
    parser.add_argument('-dropout', type=float, default=0.0)
    parser.add_argument('-modelname', type=str, default='best_211007')
    parser.add_argument('-num_hidden', type=int, default=32)
    parser.add_argument('-train_only', action="store_true", default=False)
    parser.add_argument('-sw_thr', type=float, default=0.0)
    parser.add_argument('-sw_decay', type=int, default=1)
    parser.add_argument('-loss', type=str, default='kl_div_normal')
    args = parser.parse_args('')

    INPUT = create_input(data, device, args)
    data, num_mols, T, Graphs, seg = INPUT
    atom_feat_dim = Graphs.ndata['feat'].shape[-1]
    ####################################
     
    Predicted_HoV, atom_features_each_layer, Attention_each_layer, T_part, updated_T = predict(model, Graphs.ndata['feat'], Graphs, \
                                                        seg, args.maxatoms, T, '', num_mols, mu_s_NLR=[])
    pred_mean, pred_stddev = Predicted_HoV[:, 0], Predicted_HoV[:, 1]
    data['Predicted'] = pred_mean.numpy()
    data['ML_unc'] = pred_stddev.numpy()

    train = data[data['Train/Valid/Test'] == 'Train']
    valid = data[data['Train/Valid/Test'] == 'Valid']
    test = data[data['Train/Valid/Test'] == 'Test']
    MAE_train, MAE_valid, MAE_test = np.abs(train.Predicted - train['HoV (kJ/mol)']).mean(), \
                                     np.abs(valid.Predicted - valid['HoV (kJ/mol)']).mean(), \
                                     np.abs(test.Predicted - test['HoV (kJ/mol)']).mean()
    print(MAE_train, MAE_valid, MAE_test)
    return model

def create_input(DATA, device, args, train_molecule_molgraphs_dict=''):
    data = DATA.sort_values(by=['smiles'])
    smiles_counter = Counter(np.array(data.smiles))

    num_mols = tf.constant([ smiles_counter[smi] for smi in sorted(smiles_counter.keys())], dtype=tf.int32)
    T = tf.constant(data.temperature, dtype=tf.float32)

    if train_molecule_molgraphs_dict == '':
        Graphs = batch( [ dgl_molgraph_one_molecule(smi, args.maxatoms, device, args.explicitH) \
                                                                     for smi in sorted(list(data.smiles.unique())) ] ).to(device)
    else:
        Graphs = batch( [ train_molecule_molgraphs_dict[smi] \
                                                        for smi in sorted(list(data.smiles.unique())) ] ).to(device)

    seg = create_segment(args.maxatoms, list( data.total_atoms ))
    try:
        Y = tf.constant(data['HoV (kJ/mol)'], dtype=tf.float32)
        Err = tf.constant(data.Error, dtype=tf.float32)
        sample_weight = tf.constant(data.sample_weight, dtype=tf.float32)
        INPUT = data, num_mols, Y, T, Graphs, seg, Err, sample_weight
    except:
        INPUT = data, num_mols, T, Graphs, seg

    return INPUT

def predict(model, features, g, segment, Max_atoms, T, equation, num_mols, mu_s_NLR):
    pred, atom_features_each_layer, Attention_each_layer, T_part, updated_T = model(features, g, segment, Max_atoms, T, equation, num_mols, training=False, verbose=True, mu_s_NLR=mu_s_NLR)
    return pred,  atom_features_each_layer, Attention_each_layer, T_part, updated_T 

