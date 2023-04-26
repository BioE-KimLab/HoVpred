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

def kl_div_normal(mu1, mu2, sigma1, sigma2):
    return tf.math.log(sigma2/sigma1) + ( ( sigma1 ** 2.0 + (mu1-mu2) ** 2.0 )  /   (2 * (sigma2**2.0))  ) - 0.5

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

def create_model(args, atom_feat_dim, equation):
    num_heads = args.heads
    num_layers = args.layers
    num_out_heads = 1

    heads = ([num_heads] * num_layers) + [num_out_heads]

    if args.loss[0:2] == 'kl':
        from gnn import GAT_unc
        model = GAT_unc(num_layers=args.layers,
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

        #graph convolutional network (no attention)
        '''
        from gcn import GCN_unc
        model = GCN_unc(num_layers=args.layers,
                    in_dim=atom_feat_dim,
                    num_hidden=args.num_hidden,
                    activation=tf.nn.relu)
        '''
    else:
        from gnn import GAT
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
    print(model)

    if args.modelname == '':
        model_name = "_".join([ str(x) for x in [args.lr,args.batchsize,args.layers,\
                                                 args.heads,args.residcon,args.loss,args.sw_thr,args.sw_decay,args.num_hidden]])
        if args.train_only:
            model_name += '_trainonly'
    else:
        model_name = args.modelname

    if not os.path.exists('results_'+equation):
        os.mkdir('results_'+equation)

    if not os.path.exists('results_'+equation+'/'+model_name):
        os.mkdir('results_'+equation+'/'+model_name)

    if not args.K_fold:
        try:
            model.load_model('results_'+equation+'/'+ model_name +'/my_model')
        except:
            pass

    return model, model_name

def train_model(model, model_name, args, device, equation,\
                        TRAIN, VALID, TEST, train_molecule_molgraphs_dict, mu_s_NLR = []):
    with tf.device(device): 
        weight_decay = 5e-4
        train_data, num_mols_train, Y_train, T_train, Graphs_train, seg_train, Err_train, sw_train = TRAIN

        if VALID != '':
            valid_data, num_mols_valid, Y_valid, T_valid, Graphs_valid, seg_valid, Err_valid, sw_valid = VALID
        if TEST != '':
            test_data, num_mols_test, Y_test, T_test, Graphs_test, seg_test, Err_test, sw_test = TEST

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-8)
        try:
            if args.train_only:
                train_valid_costs = [[row['train_cost']] \
                                                for _, row in pd.read_csv('results_'+equation+'/'+ model_name +'/costs.csv').iterrows()]
            else:
                train_valid_costs = [  [ row['train_cost'], row['valid_cost'] ] \
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
                        T_batch, train_graphs_batch, seg_train_batch, Err_batch, sw_batch = INPUT_batch

                #print(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]))
                with tf.GradientTape() as tape:
                    tape.watch(model.trainable_weights)
                    pred = model(train_graphs_batch.ndata['feat'], train_graphs_batch, seg_train_batch, args.maxatoms, T_batch, equation, num_mols_batch, training=True, mu_s_NLR=mu_s_NLR)

                    if args.loss == 'mae':
                        loss_value = tf.reduce_mean(   tf.math.multiply(  sw_batch, tf.math.abs(Y_batch-pred)  ) )
                    elif args.loss == 'mse':
                        loss_value = tf.reduce_mean(  tf.math.multiply(  sw_batch, tf.pow((Y_batch-pred),2)  ) )
                    elif args.loss == 'kl_div_normal':
                        pred_mean, pred_stddev = pred[:, 0] + 0.01, pred[:, 1] + 0.01
                        loss_value = tf.reduce_mean(  tf.math.multiply( sw_batch, kl_div_normal( Y_batch, pred_mean, Err_batch, pred_stddev ) )    )

                    for weight in model.trainable_weights:
                        loss_value = loss_value + weight_decay * tf.nn.l2_loss(weight)

                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                batch_costs.append(loss_value.numpy())

            ##### Evaluate train/validation set
            train_loss, _Y_train = evaluate(model, Graphs_train.ndata['feat'], Graphs_train, \
                                                                seg_train, args.maxatoms, T_train, equation, num_mols_train, Y_train, mu_s_NLR)
            if len(_Y_train) == 2:
                _Y_train, _s_train = _Y_train

            print('epoch '+str(epoch)+': train MAE- '+str(train_loss.numpy()))

            if not args.train_only:
                valid_loss, _Y_valid = evaluate(model, Graphs_valid.ndata['feat'], Graphs_valid, \
                                                                    seg_valid, args.maxatoms, T_valid, equation, num_mols_valid, Y_valid, mu_s_NLR)
                if len(_Y_valid) == 2:
                    _Y_valid, _s_valid = _Y_valid

                print('epoch '+str(epoch)+': valid MAE- '+str(valid_loss.numpy()))

            if args.train_only:
                train_valid_costs.append([train_loss.numpy()])
            else:
                train_valid_costs.append([train_loss.numpy(), valid_loss.numpy()])

            if args.train_only:
                pd.DataFrame(train_valid_costs).to_csv('results_'+equation+'/'+ model_name +'/costs.csv', header = ['train_cost'])
            elif not args.K_fold:
                pd.DataFrame(train_valid_costs).to_csv('results_'+equation+'/'+ model_name +'/costs.csv', header = ['train_cost', 'valid_cost'])
            
            if not args.K_fold:
                pd.DataFrame(batch_costs).to_csv('results_'+equation+'/'+model_name +'/batch_costs.csv', header = ['batch_cost'])

            if not args.train_only:
                if np.abs(min(np.array(train_valid_costs)[:, 1]) - valid_loss.numpy()) < 1.0e-4:
                    model.save_model('results_'+equation+'/'+ model_name +'/my_model')

        result_train = np.transpose([_Y_train,np.array(Y_train),np.array(T_train)])
        if args.train_only:
            hov_results = pd.DataFrame(result_train, columns = ['Predicted', 'NIST', 'Temperature'])
            hov_results['smiles'] = pd.DataFrame(list(train_data.smiles))
        else:
            print('load model with the lowest valid MAE, evaluate train,valid,test')
            model.load_model('results_'+equation+'/'+ model_name +'/my_model')

            train_loss, _Y_train = evaluate(model, Graphs_train.ndata['feat'], Graphs_train, \
                                                                seg_train, args.maxatoms, T_train, equation, num_mols_train, Y_train, mu_s_NLR)
            valid_loss, _Y_valid = evaluate(model, Graphs_valid.ndata['feat'], Graphs_valid, \
                                                                    seg_valid, args.maxatoms, T_valid, equation, num_mols_valid, Y_valid, mu_s_NLR)
            test_loss, _Y_test = evaluate(model, Graphs_test.ndata['feat'], Graphs_test, \
                                                         seg_test, args.maxatoms, T_test, equation, num_mols_test, Y_test, mu_s_NLR)
            if len(_Y_test) == 2:
                _Y_train, _s_train = _Y_train
                _Y_valid, _s_valid = _Y_valid
                _Y_test, _s_test = _Y_test
            
            print('test MAE- '+str(test_loss.numpy()) )

            result_valid = np.transpose([_Y_valid,np.array(Y_valid),np.array(T_valid)])
            result_test = np.transpose([_Y_test,np.array(Y_test),np.array(T_test)])

            SMILES_train, SMILES_valid, SMILES_test = train_data.smiles, valid_data.smiles, test_data.smiles
            hov_results = pd.DataFrame(np.concatenate((result_train, result_valid, result_test), axis=0), columns = ['Predicted', 'NIST', 'Temperature'])
            hov_results['Train/Valid/Test'] = pd.DataFrame(['Train']*len(Y_train) + ['Valid']*len(Y_valid) + ['Test']*len(Y_test))
            hov_results['smiles'] = pd.DataFrame(list(SMILES_train) + list(SMILES_valid) + list(SMILES_test))
            if args.loss[0:2] == 'kl':
                hov_results['ML_unc'] = pd.DataFrame( np.concatenate(  (np.array(_s_train), np.array(_s_valid), np.array(_s_test)), axis = 0  ))
            hov_results['DB_unc'] = pd.DataFrame(list(train_data.Error) + list(valid_data.Error) + list(test_data.Error) ) 
            hov_results['sample_weight'] = pd.DataFrame(list(train_data.sample_weight) + list(valid_data.sample_weight) + list(test_data.sample_weight) ) 

    print('------------------------------------------')
    if args.train_only:
        print(len(Y_train))
        print ("MAE_train : ", train_loss.numpy())
        return _Y_train, hov_results
    else:
        print(len(Y_train), len(Y_valid), len(Y_test))
        print ("MAE_train : ", train_loss.numpy(), ", MAE_valid : ", valid_loss.numpy(), ", MAE_test :  ", test_loss.numpy())
        return _Y_train, _Y_valid, _Y_test, hov_results

def evaluate(model, features, g, segment, Max_atoms, T, equation, num_mols, Y, mu_s_NLR):
    pred = model(features, g, segment, Max_atoms, T, equation, num_mols, training=False, mu_s_NLR=mu_s_NLR)

    if len(pred.shape) == 2:
        pred_mean, pred_stddev = pred[:, 0] + 0.01, pred[:, 1] + 0.01
        loss_value = tf.reduce_mean(tf.math.abs(Y-pred_mean)) 
        return loss_value, [pred_mean, pred_stddev]
    else:
        loss_value = tf.reduce_mean(tf.math.abs(Y-pred))
        return loss_value, pred

def predict(model, features, g, segment, Max_atoms, T, equation, num_mols, mu_s_NLR):
    if equation == 'Watson':
        pred, A,B,C = model(features, g, segment, Max_atoms, T, equation, num_mols, training=False, verbose=True, mu_s_NLR=mu_s_NLR)
        return pred, A,B,C
    else:
        pred, atom_features_each_layer, Attention_each_layer, T_part, updated_T = model(features, g, segment, Max_atoms, T, equation, num_mols, training=False, verbose=True, mu_s_NLR=mu_s_NLR)
        return pred,  atom_features_each_layer, Attention_each_layer, T_part, updated_T 

def main():
    parser = ArgumentParser()
    parser.add_argument('-predict', action="store_true", default=False, help='If specified, prediction is carried out (default=False)')
    parser.add_argument('-watsoneq', action='store_true', default=False, help='whether to use watson equation (default=False)')
    parser.add_argument('-K_fold', action='store_true', default=False, help='whether to run KFoldCV (default=False)')
    parser.add_argument('-maxatoms', type=int, default=64, help='Maximum number of atoms in a molecule (default=64)')
    parser.add_argument('-lr', type=float, default=5.0e-4, help='Learning rate (default=5.0e-4)')
    parser.add_argument('-epoch', type=int, default=200, help='epoch (default=200)')
    parser.add_argument('-batchsize', type=int, default=256, help='batch_size (default=256)')
    parser.add_argument('-layers', type=int, default=5, help='number of gnn layers (default=5)')
    parser.add_argument('-heads', type=int, default=5, help='number of gat heads (default=5)')
    parser.add_argument('-residcon', action="store_true", default=True, help='whether to use residual connection (default=True)')
    parser.add_argument('-explicitH', action="store_true", default=False, help='whether to use explicit hydrogens (default=False)')
    parser.add_argument('-dropout', type=float, default=0.0, help='dropout rate (default=0.0)')
    parser.add_argument('-modelname', type=str, default='', help='model name (default=an array of hyperparameter values)')
    parser.add_argument('-num_hidden', type=int, default=32, help='number of nodes in hidden layers (default=32)')
    parser.add_argument('-train_only', action="store_true", default=False, help='If specified, no 8:1:1 split is carried out, the whole database is used for training (default=False)')
    parser.add_argument('-loss', type=str, default='mse', help='loss function (default=mse). Options - mae, mse, kl_div_normal')
    #parser.add_argument('-fraction_for_learning_curve', type=int, default=1, help='training set fraction to obtain learning curve')
    # Eventually, we did not use sample weights. Using the KL divergence is better. Now deprecated
    parser.add_argument('-sw_thr', type=float, default=0.0, help='Minimum uncertainty to which 1/Err sample weight is applied (default=0.0 - no sample weights)')
    parser.add_argument('-sw_decay', type=int, default=1, help='Sample weight decay function (default=1 - 1/x^-1)')
    args = parser.parse_args()

    if args.watsoneq:
        equation = 'Watson'
    else:
        equation = ''
    if len(gpus) > 0:
        device = "/gpu:0"
    else:
        device = "/cpu:0"

    with tf.device(device):
        if args.predict:
            data = pd.read_csv('molecules_to_predict.csv')

            mu_s_NLR = []
            if args.explicitH:
                data['total_atoms'] = [ len(rdkit.Chem.AddHs(rdkit.Chem.MolFromSmiles(smi)).GetAtoms()) for smi in data.smiles]
            else:
                data['total_atoms'] = [ rdkit.Chem.MolFromSmiles(smi).GetNumHeavyAtoms() for smi in data.smiles]

            INPUT = create_input(data, device, args)

            data, num_mols, T, Graphs, seg = INPUT
            atom_feat_dim = Graphs.ndata['feat'].shape[-1]

            model, model_name = create_model(args, atom_feat_dim, equation)

            if args.watsoneq:
                Predicted_HoV, A,B,C = predict(model, Graphs.ndata['feat'], Graphs, \
                                                                        seg, args.maxatoms, T, equation, num_mols, mu_s_NLR)
                data['predicted'] = Predicted_HoV.numpy()
                data['A'] = A.numpy()
                data['B'] = B.numpy()
                data['C'] = C.numpy()
            else:
                Predicted_HoV, atom_features_each_layer, Attention_each_layer, T_part, updated_T = predict(model, Graphs.ndata['feat'], Graphs, \
                                                                    seg, args.maxatoms, T, equation, num_mols, mu_s_NLR)
                if len(Predicted_HoV.shape) == 2: # mean+std learning
                    pred_mean, pred_stddev = Predicted_HoV[:, 0], Predicted_HoV[:, 1]
                    data['predicted'] = pred_mean.numpy()
                    data['predicted_stddev'] = pred_stddev.numpy()
                else:
                    data['predicted'] = Predicted_HoV.numpy()

            #data['atom_features_each_layer'] = atom_features_each_layer
            data.to_csv('molecules_to_predict_results.csv', index=False)
         
        else: # train or K-fold
            if args.K_fold:
                data = pd.read_csv('data/Data_for_kfold.csv')
                #data = pd.read_csv('data/Data_210903.csv')
            else:
                data = pd.read_csv('data/Data.csv')
                #data = pd.read_csv('data/Data_211005.csv')
                #data = pd.read_csv('data/HoV_lit1.csv')
                #data = pd.read_csv('data/HoV_lit2.csv')
                #data = pd.read_csv('data/HoV_lit3.csv')

            if args.explicitH:
                data = data.rename(columns={"total_num_atoms":"total_atoms"})
            else:
                data = data.rename(columns={"num_heavy_atoms":"total_atoms"})

            if args.K_fold:
                data = data[['smiles','temperature','HoV (kJ/mol)','total_atoms','Error']]
            else:
                data = data[['smiles','temperature','HoV (kJ/mol)','total_atoms','Error','Train/Valid/Test']]

            if args.sw_thr == 0.0:
                data['sample_weight'] = [ 1.0 ] * len(data)
            else:
                data['sample_weight'] = [ 1.0 if (x < args.sw_thr) else (1.0 / (x ** (args.sw_decay))  ) for x in data.Error ]

            target_molecules = pd.Series(data.smiles.unique())
            if args.train_only:
                train = target_molecules.sample(frac=1.0, random_state=1)
                TRAIN = create_input(data, device, args)
                VALID = ''
                TEST = ''
            else: 
                if args.K_fold: 
                    train = target_molecules.sample(frac=.8, random_state=1).sort_values()
                    valid = target_molecules[~target_molecules.index.isin(train.index)].sample(frac=.5, random_state=1).sort_values()
                    test = target_molecules[~target_molecules.index.isin(train.index) & ~target_molecules.index.isin(valid.index)].sort_values()
                    train_data = data.loc[data['smiles'].isin(train)]
                    valid_data = data.loc[data['smiles'].isin(valid)]
                    test_data =  data.loc[data['smiles'].isin(test)] 
                else:
                    train_data, valid_data, test_data = data[ data['Train/Valid/Test'] == 'Train' ],\
                                                        data[ data['Train/Valid/Test'] == 'Valid' ],\
                                                        data[ data['Train/Valid/Test'] == 'Test' ]

                    # to obtain a learning curve
                    # 600,1200,1800, ..., 5400, 5994 (original)
                    #train_data = train_data[train_data.smiles.isin(train_data.smiles.unique()[:600*args.fraction_for_learning_curve])]

                    if args.watsoneq:
                        NLR_results = pd.read_csv('nonlinear_regression/reg_with_unc_results.csv')
                        NLR_results = NLR_results[NLR_results.smiles.isin(train_data.smiles.unique())]
                        NLR_results = NLR_results[ (NLR_results.logA_det > 0) & (NLR_results.C_det > 0) & (NLR_results.B_det > 0)]
                        logA_NLR, C_NLR, B_NLR = NLR_results['logA_det'], NLR_results['C_det'], NLR_results['B_det'] 
                        mu_s_NLR = [np.mean(logA_NLR), np.std(logA_NLR), 
                                    np.mean(B_NLR), np.std(B_NLR),   
                                    np.mean(C_NLR), np.std(C_NLR)   ]
                    else:
                        mu_s_NLR = []

                TRAIN, VALID, TEST = create_input(train_data, device, args), \
                                     create_input(valid_data, device, args), \
                                     create_input(test_data, device, args)
                if args.K_fold:
                    r2 = []
                    kfold = KFold(n_splits=10, shuffle = True, random_state = 1)
                    molecules_for_KFold = pd.concat([train,valid]).sort_values()

                    fold_count = 0
                    for train_index, valid_index in list(kfold.split(molecules_for_KFold)) :
                        print('Fold Count ',fold_count)
                        train = molecules_for_KFold.iloc[train_index]
                        valid = molecules_for_KFold.iloc[valid_index]

                        train_data = data.loc[data['smiles'].isin(train)]
                        valid_data = data.loc[data['smiles'].isin(valid)]

                        if args.watsoneq:
                            NLR_results = pd.read_csv('nonlinear_regression/reg_with_unc_results.csv')
                            NLR_results = NLR_results[NLR_results.smiles.isin(train_data.smiles.unique())]
                            NLR_results = NLR_results[ (NLR_results.logA_det > 0) & (NLR_results.C_det > 0) & (NLR_results.B_det > 0)]
                            logA_NLR, C_NLR, B_NLR = NLR_results['logA_det'], NLR_results['C_det'], NLR_results['B_det'] 
                            mu_s_NLR = [np.mean(logA_NLR), np.std(logA_NLR), 
                                        np.mean(B_NLR), np.std(B_NLR),   
                                        np.mean(C_NLR), np.std(C_NLR)   ]
                            print(mu_s_NLR)
                        else:
                            mu_s_NLR = []

                        TRAIN = create_input(train_data, device, args)
                        VALID = create_input(valid_data, device, args)
                        TEST = create_input(test_data, device, args)

                        train_molecule_molgraphs_dict = {} # for finding molgraphs after the batch being shuffled
                        for smi in sorted(list(train_data.smiles.unique())):
                            one_mol_graph = dgl_molgraph_one_molecule(smi, args.maxatoms, device, args.explicitH) 
                            train_molecule_molgraphs_dict[smi] = one_mol_graph

                        atom_feat_dim = VALID[-4].ndata['feat'].shape[-1] # -4: molgraphs
                        model, model_name = create_model(args, atom_feat_dim, equation)

                        _Y_train, _Y_valid, _Y_test, hov_results = train_model( model, model_name, args, device, equation, TRAIN, VALID, TEST,train_molecule_molgraphs_dict, mu_s_NLR  )
                        pd.DataFrame(hov_results).to_csv('results_'+equation+'/' +model_name+ '/HoV_results'+ str(fold_count) +'.csv')

                        os.system('mkdir '+'results_'+equation+'/' +model_name+ '/model'+str(fold_count))
                        os.system('mv '+'results_'+equation+'/' +model_name+ '/my_model* '+  'results_'+equation+'/' +model_name+ '/model'+str(fold_count)+'/.' )
                        os.system('mv '+'results_'+equation+'/' +model_name+ '/checkpoint '+  'results_'+equation+'/' +model_name+ '/model'+str(fold_count)+'/.' )
                        
                        Y_valid = VALID[2]
                        r1 = r2_score(list(_Y_valid), list(Y_valid))
                        r2.append(r1)

                        fold_count += 1
                else:
                    train_molecule_molgraphs_dict = {}
                    train = pd.Series(train_data.smiles.unique())
                    for smi in train:
                        one_mol_graph = dgl_molgraph_one_molecule(smi, args.maxatoms, device, args.explicitH) 
                        train_molecule_molgraphs_dict[smi] = one_mol_graph

                    atom_feat_dim = TRAIN[-4].ndata['feat'].shape[-1] # -4: molgraphs

                    model, model_name = create_model(args, atom_feat_dim, equation)
                    _Y_train, _Y_valid, _Y_test, hov_results = train_model( model, model_name, args, device, equation, TRAIN, VALID, TEST, train_molecule_molgraphs_dict, mu_s_NLR )
                    pd.DataFrame(hov_results).to_csv('results_'+equation+'/' +model_name+ '/HoV_results.csv')


if __name__ == '__main__':
    main()
