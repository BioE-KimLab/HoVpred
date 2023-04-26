import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'
import numpy as np
import pandas as pd
import rdkit.Chem
from molgraph import * 
from dgl import batch
import sys
from argparse import ArgumentParser
import datetime
from collections import Counter

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from read_hov_model import read_hov_model

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
        if args.prop == 'FP':
            Y = tf.constant(data['flash point'], dtype=tf.float32)
        elif args.prop == 'Tc':
            Y = tf.constant(data['Tc'], dtype=tf.float32)
        elif args.prop == 'Tm':
            Y = tf.constant(data['Melting Point'], dtype=tf.float32)
        elif args.prop == 'Tb':
            Y = tf.constant(data['Normal Boiling Point'], dtype=tf.float32)
        elif args.prop == 'Cp_liq':
            Y = tf.constant(data['Cp_liq_298K'], dtype=tf.float32)

        INPUT = data, num_mols, Y, Graphs, seg
    except:
        INPUT = data, num_mols, Graphs, seg

    return INPUT

def create_model(args, atom_feat_dim, equation):
    num_heads = args.heads
    num_layers = args.layers
    num_out_heads = 1

    heads = ([num_heads] * num_layers) + [num_out_heads]

    from gnn_other_prop import GAT
    model = GAT(num_layers=args.layers,
                in_dim=atom_feat_dim,
                num_hidden=args.num_hidden,
                num_classes=args.num_hidden,
                heads=heads,
                activation=tf.nn.relu,
                feat_drop=args.dropout,
                attn_drop=args.dropout,
                negative_slope=0.2,
                residual=args.residcon,
                equation=equation)
    
    if args.modelname == '':
        model_name = "_".join([ str(x) for x in [args.lr,args.batchsize,args.layers,\
                                                 args.heads,args.residcon,args.explicitH,\
                                                 args.dropout,args.num_hidden]])
    else:
        model_name = args.modelname

    if not os.path.exists('results_'+equation):
        os.mkdir('results_'+equation)

    if not os.path.exists('results_'+equation+'/'+model_name):
        os.mkdir('results_'+equation+'/'+model_name)

    return model, model_name



def initialize_model(model, args, device, equation, TRAIN, train_molecule_molgraphs_dict):
    # just one epoch training for a small batch for initialzing weights and biases
    # This enables matching of HoV model weights and FP model weights
    with tf.device(device): 
        weight_decay = 5e-4
        train_data, num_mols_train, Y_train, Graphs_train, seg_train = TRAIN
        data_batch = train_data.iloc[0:32]
    
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-8)
        INPUT_batch = create_input(data_batch, device, args, train_molecule_molgraphs_dict)
        data_batch, num_mols_batch, Y_batch, \
                train_graphs_batch, seg_train_batch = INPUT_batch
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logits = model(train_graphs_batch.ndata['feat'], train_graphs_batch, seg_train_batch, args.maxatoms, equation, num_mols_batch, training=True)

            loss_value = tf.reduce_mean(tf.pow((Y_batch-logits),2))
            for weight in model.trainable_weights:
                loss_value = loss_value + weight_decay * tf.nn.l2_loss(weight)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))


def train_model(model, model_name, args, device, equation,\
                        TRAIN, VALID, TEST, train_molecule_molgraphs_dict):
    with tf.device(device): 
        weight_decay = 5e-4
        train_data, num_mols_train, Y_train, Graphs_train, seg_train = TRAIN

        if VALID != '':
            valid_data, num_mols_valid, Y_valid, Graphs_valid, seg_valid = VALID
        if TEST != '':
            test_data, num_mols_test, Y_test, Graphs_test, seg_test = TEST

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-8)
        try:
            train_valid_costs = [[row['train_cost'], row['valid_cost']] \
                                                for _, row in pd.read_csv('results_'+equation+'/'+ model_name +'/costs.csv').iterrows()]
            batch_costs = [row['batch_cost'] \
                                            for _, row in pd.read_csv('results_'+equation+'/'+ model_name +'/batch_costs.csv').iterrows()]
        except:
            train_valid_costs = []
            batch_costs = []

        for epoch in range(args.epoch):
            train_data_shuffled = train_data.sample(frac = 1.0, random_state = epoch)
            num_batches = int(np.ceil(len(Y_train) / args.batchsize))

            for _iter in range(num_batches):
                data_batch = train_data_shuffled.iloc[_iter*args.batchsize:(_iter+1)*args.batchsize]

                INPUT_batch = create_input(data_batch, device, args, train_molecule_molgraphs_dict)
                data_batch, num_mols_batch, Y_batch, \
                        train_graphs_batch, seg_train_batch = INPUT_batch

                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_weights)
                    logits = model(train_graphs_batch.ndata['feat'], train_graphs_batch, seg_train_batch, args.maxatoms, equation, num_mols_batch, training=True)

                    loss_value = tf.reduce_mean(tf.pow((Y_batch-logits),2))
                    for weight in model.trainable_weights:
                        loss_value = loss_value + weight_decay * tf.nn.l2_loss(weight)

                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                batch_costs.append(loss_value.numpy())

            train_loss, _Y_train = evaluate(model, Graphs_train.ndata['feat'], Graphs_train, \
                                                                seg_train, args.maxatoms,  equation, num_mols_train, Y_train)
            print('epoch '+str(epoch)+': train loss- '+str(train_loss.numpy()))

            valid_loss, _Y_valid = evaluate(model, Graphs_valid.ndata['feat'], Graphs_valid, \
                                                                seg_valid, args.maxatoms, equation, num_mols_valid, Y_valid)
            print('epoch '+str(epoch)+': valid loss- '+str(valid_loss.numpy()))


            if not args.K_fold:
                train_valid_costs.append([train_loss.numpy(), valid_loss.numpy()])
                pd.DataFrame(train_valid_costs).to_csv('results_'+equation+'/'+ model_name +'/costs.csv', header = ['train_cost','valid_cost'])

                if np.abs(min(np.array(train_valid_costs)[:, 1]) - valid_loss.numpy()) < 1.0e-4:
                    model.save_model('results_'+equation+'/'+ model_name +'/my_model')
                pd.DataFrame(batch_costs).to_csv('results_'+equation+'/'+model_name +'/batch_costs.csv', header = ['batch_cost'])


        print('load model with the lowest valid MAE, evaluate train,valid,test')
        model.load_model('results_'+equation+'/'+ model_name +'/my_model')

        train_loss, _Y_train = evaluate(model, Graphs_train.ndata['feat'], Graphs_train, \
                                                            seg_train, args.maxatoms, equation, num_mols_train, Y_train)
        valid_loss, _Y_valid = evaluate(model, Graphs_valid.ndata['feat'], Graphs_valid, \
                                                                seg_valid, args.maxatoms, equation, num_mols_valid, Y_valid)
        test_loss, _Y_test = evaluate(model, Graphs_test.ndata['feat'], Graphs_test, \
                                                     seg_test, args.maxatoms, equation, num_mols_test, Y_test)
        print(len(Y_train), len(Y_valid), len(Y_test))
        print ("MAE_train : ", train_loss.numpy(), ", MAE_valid : ", valid_loss.numpy(), ", MAE_test :  ", test_loss.numpy())

        result_train = np.transpose([_Y_train,np.array(Y_train)])
        result_valid = np.transpose([_Y_valid,np.array(Y_valid)])
        result_test = np.transpose([_Y_test,np.array(Y_test)])


        SMILES_train, SMILES_valid, SMILES_test = train_data.smiles, valid_data.smiles, test_data.smiles
        results = pd.DataFrame(np.concatenate((result_train, result_valid, result_test), axis=0), columns = ['Predicted', 'DB_value'])

        results['Train/Valid/Test'] = pd.DataFrame(['Train']*len(Y_train) + ['Valid']*len(Y_valid) + ['Test']*len(Y_test))
        results['smiles'] = pd.DataFrame(list(SMILES_train) + list(SMILES_valid) + list(SMILES_test))

        pd.DataFrame(results).to_csv('results_'+equation+'/' +model_name+ '/results.csv')

def evaluate(model, features, g, segment, Max_atoms, equation, num_mols, Y):
    logits = model(features, g, segment, Max_atoms, equation, num_mols, training=False)
    loss_value = tf.reduce_mean(tf.math.abs(Y-logits))

    return loss_value, logits

def predict(model, features, g, segment, Max_atoms, equation, num_mols):
    return model(features, g, segment, Max_atoms, equation, num_mols, training=False, verbose=False)

def main():
    parser = ArgumentParser()
    parser.add_argument('-predict', action="store_true", default=False, help='If specified, prediction is carried out (default=False)')
    parser.add_argument('-K_fold', action='store_true', default=False, help='whether to run KFoldCV (default=False)')
    parser.add_argument('-prop', type=str, default='', help='target property name. Currently available: FP, Tc,')

    parser.add_argument('-maxatoms', type=int, default=64, help='Maximum number of atoms in a molecule (default=64)')
    parser.add_argument('-lr', type=float, default=1.0e-2, help='Learning rate (default=1.0e-2)')
    parser.add_argument('-epoch', type=int, default=500, help='epoch (default=500)')
    parser.add_argument('-batchsize', type=int, default=32, help='batch_size (default=32)')
    parser.add_argument('-layers', type=int, default=5, help='number of gnn layers (default=5)')
    parser.add_argument('-heads', type=int, default=5, help='number of gat heads (default=5)')
    parser.add_argument('-residcon', action="store_true", default=True, help='whether to use residual connection (default=True)')
    parser.add_argument('-modelname', type=str, default='', help='model name (default=an array of hyperparameter values)')

    parser.add_argument('-explicitH', action="store_true", default=False, help='whether to use explicit hydrogens (default=False)')
    parser.add_argument('-dropout', type=float, default=0.0, help='dropout rate (default=0.0)')
    parser.add_argument('-num_hidden', type=int, default=32, help='number of nodes in hidden layers (default=32)')
    
    parser.add_argument('-num_layers_transfer', type=int, default=0, help='number of GAT layers to transfer from the HoV prediction model to predict FP (default=0)')
    parser.add_argument('-fix_transferred_weights', action="store_true", default=False, help='whether to fix transferred weights (default=False)')
    parser.add_argument('-num_layers_to_fix', type=int, default=0, help='number of layers to fix (default=0)')
    parser.add_argument('-random_state', type=int, default=1, help='random seed number (default=1)')
    args = parser.parse_args()

    equation = ''
    if len(gpus) > 0:
        device = "/gpu:0"
    else:
        device = "/cpu:0"

    with tf.device(device):
        ############## predict (221222)
        if args.predict:
            data = pd.read_csv('molecules_to_predict.csv')
            data['temperature'] = [298.15] * len(data) # just dummy. not gonna use it
            data['total_atoms'] = [ rdkit.Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in data.smiles]
            INPUT = create_input(data, device, args)
            data, num_mols, Graphs, seg = INPUT
            atom_feat_dim = Graphs.ndata['feat'].shape[-1]
            model, model_name = create_model(args, atom_feat_dim, equation)

            model.load_model('results_/'+model_name+'/my_model')

            predicted = predict(model, Graphs.ndata['feat'], Graphs, seg, args.maxatoms, equation, num_mols)

            data['predicted'] = predicted

            data.to_csv('molecules_to_predict_results.csv', index=False)

            sys.exit()
        ############## predict (221222)


        #### train (221222)
        if args.prop == 'FP':
            data = pd.read_csv('data/FP_211116.csv') 
            #DIPPR only or the whole DB?
            data = data[data.source == 'DIPPR']
        elif args.prop == 'Tc':
            data = pd.read_csv('data/Tc_211222.csv')
        elif args.prop == 'Tm':
            data = pd.read_csv('data/Tm.csv')
        elif args.prop == 'Tb':
            data = pd.read_csv('data/Tb.csv')
        elif args.prop == 'Cp_liq':
            data = pd.read_csv('data/Cp_liq_298K.csv')

        if args.explicitH:
            data['total_atoms'] = [ len(rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smi)).GetAtoms()) for smi in data.smiles]
        else:
            data['total_atoms'] = [ rdkit.Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in data.smiles]
        data = data[data.total_atoms <= 64]

        data['temperature'] = [298.15] * len(data) # just dummy. not gonna use it
        hov_model = read_hov_model() 
        INPUT = create_input(data, device, args)

        data, num_mols, T, Graphs, seg = INPUT
        atom_feat_dim = Graphs.ndata['feat'].shape[-1]
        model, model_name = create_model(args, atom_feat_dim, equation)
        print('model created')

        hov_layer_dict = dict( [(layer.name, layer) for layer in hov_model.layers]   )
        layer_dict = dict( [(layer.name, layer) for layer in model.layers]   )

        train_data = data.sample(frac=.8, random_state=args.random_state)
        valid_data = data[~data.index.isin(train_data.index)].sample(frac=.5, random_state=args.random_state)
        test_data = data[~data.index.isin(train_data.index) & ~data.index.isin(valid_data.index)]

        TRAIN, VALID, TEST = create_input(train_data, device, args), \
                             create_input(valid_data, device, args), \
                             create_input(test_data, device, args)

        train_molecule_molgraphs_dict = {} # for finding molgraphs after the batch being shuffled
        for smi in train_data.smiles:
            one_mol_graph = dgl_molgraph_one_molecule(smi, args.maxatoms, device, args.explicitH) 
            train_molecule_molgraphs_dict[smi] = one_mol_graph

        atom_feat_dim = TRAIN[-2].ndata['feat'].shape[-1] # -2: molgraphs


        print('initialize model (weights, biases)')
        initialize_model( model, args, device, equation, TRAIN, train_molecule_molgraphs_dict)
        print('***************************')
        for layer in model.layers:
            print(layer.name)
            print(len(layer.get_weights()))
            print([w.shape for w in layer.get_weights()])

        hov_gat_first_layer_weights = hov_layer_dict['gat_conv'].get_weights() 
        gat_first_layer_weights_before_transfer = layer_dict['gat_conv_6'].get_weights()

        #########
        if args.num_layers_transfer >= 1:
            layer_dict['gat_conv_6'].set_weights( hov_layer_dict['gat_conv'].get_weights()   )

            for i in range(1,args.num_layers_transfer):
                layer_dict['gat_conv_'+str(i+6)].set_weights( hov_layer_dict['gat_conv_'+str(i)].get_weights()   )

            if args.fix_transferred_weights:
                #for i in range(args.num_layers_transfer):
                for i in range(args.num_layers_to_fix):
                    layer_dict['gat_conv_'+str(i+6)].trainable = False
        else:
            print('no layers were transferred from the HoV model')
        ##########

        gat_first_layer_weights_after_transfer = layer_dict['gat_conv_6'].get_weights()

        for i in range(len(gat_first_layer_weights_after_transfer)):
            isclose_1 = np.allclose(       gat_first_layer_weights_before_transfer[i], hov_gat_first_layer_weights[i]     ) 
            isclose_2 = np.allclose(       gat_first_layer_weights_after_transfer[i], hov_gat_first_layer_weights[i]     )
            print(i, isclose_1, isclose_2)

        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #_Y_train, _Y_valid, _Y_test = train_model( model, model_name, args, device, equation, TRAIN, VALID, TEST, train_molecule_molgraphs_dict)
        train_model( model, model_name, args, device, equation, TRAIN, VALID, TEST, train_molecule_molgraphs_dict)
        gat_first_layer_weights_after_training = layer_dict['gat_conv_6'].get_weights()
        for i in range(len(gat_first_layer_weights_after_training)):
            isclose_3 = np.allclose(       gat_first_layer_weights_after_transfer[i], gat_first_layer_weights_after_training[i]     ) 
            print(i, isclose_3)

if __name__ == '__main__':
    main()
