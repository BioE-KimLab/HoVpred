import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'
from tensorflow import keras
from tensorflow.keras import layers
from dgl.nn.tensorflow import GATConv
import gc
import sys
import numpy as np

class GAT_unc(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 equation):
        super(GAT_unc, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = []
        self.dense_layers = []
        self.activation = activation

        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation, allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation,  allow_zero_in_degree=True))
        # output projection
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None,  allow_zero_in_degree=True))

        # Dense layers for updating atom and global feature vectors
        # These dense layers are applied ...
        # For atom feature update - 0-1: Atom feature term, 2-3: Global feature term
        # For global feature update - 4-5: Averaged atom feature term, 6-7: Global feature term
        for l in range(8):
            self.dense_layers.append(layers.Dense(num_hidden))

        # Readout layers after concatenating atom and global feature vectors
        if   equation == '':
            self.readout_layers = [layers.Dense(2*num_hidden, activation='relu'),\
                                   layers.Dense(num_hidden, activation='relu'), \
                                   layers.Dense(2, activation='relu',\
                                                kernel_initializer = tf.keras.initializers.Ones(), \
                                                bias_initializer = tf.keras.initializers.GlorotNormal())  ]

        # A dense layer for embedding of temperature as a global feature. (batch_size, 1) -> (batch_size, num_hidden)
        self.T_embedding = layers.Dense(num_hidden)

    def call(self, features, g, segment, Max_atoms, T, equation, num_mols, training, verbose=False, mu_s_NLR=[]): 
        # Shape of T: (batch_size), T_part: (batch_size, 1)
        if len(T.shape) == 1:
            T_part = tf.reshape(T, (-1, 1))
        else:
            T_part = T
        
        # (batch_size, 1) -> (batch_size, embed_size)
        T_part = self.T_embedding(T_part)

        # One-hot atom feature vectors as initial values
        # Note: The DGL graph 'g' contains ALL molecular graphs in the batch in one variable.
        # Thus the shape of 'features' is: (batch_size * Max_atoms, dim_one_hot_vector)
        h = features

        atom_features_each_layer = []
        Attention_each_layer = []
        for l in range(self.num_layers):
            #To extract atom feature vector and attention score matrix
            if verbose:
                print(l)
                a_feature_lth, Attention_lth = self.gat_layers[l](g, h, True)
                a_feature_lth = tf.reduce_mean(a_feature_lth, axis = 1)
                Attention_lth = tf.reduce_mean(Attention_lth, axis = 1)

                atom_features_each_layer.append(a_feature_lth)
                Attention_each_layer.append(Attention_lth)

            # 'heads' dimension added after passing a GAT layer
            h = self.gat_layers[l](g, h)
            # (batch_size * Max_atoms, num_heads, dim_hidden_layer) -> (batch_size * Max_atoms, num_heads * dim_hidden_layer)
            h = tf.reshape(h, (h.shape[0], -1))

        # Averaging over all heads
        # (batch_size * Max_atoms, num_heads, dim_hidden_layer) -> (batch_size * Max_atoms, dim_hidden_layer)
        h = tf.reduce_mean(self.gat_layers[-1](g, h), axis=1)
        
        # (batch_size * Max_atoms, dim_hidden_layer) -> (batch_size, Max_atoms, dim_hidden_layer) 
        updated_atom_features = tf.reshape(h, (-1, Max_atoms, h.shape[-1]))

        #reshape for tf.repeat: (batch_size, Max_atoms, dim_hidden_layer) -> (batch_size, Max_atoms*dim_hidden_layer)
        updated_atom_features = tf.reshape(updated_atom_features, (updated_atom_features.shape[0], -1))
        updated_atom_features = tf.repeat(input = updated_atom_features, repeats = num_mols, axis = 0)

        #(batch_size in terms of each data point, Max_atoms*dim_hidden_layer) -> (batch_size, Max_atoms, dim_hidden_layer)
        updated_atom_features = tf.reshape(updated_atom_features, (updated_atom_features.shape[0], Max_atoms, -1))

        if equation == '':
            update1 = self.dense_layers[1](self.dense_layers[0](updated_atom_features))

            # (batch_size, embed_size) -> (batch_size, Max_atoms, embed_size) 
            update2 = layers.RepeatVector(Max_atoms)(self.dense_layers[3](self.dense_layers[2](T_part)))

            # embed_size should equals to dim_hidden_layer
            updated_atom_features = updated_atom_features + layers.ReLU()( update1+update2 )

            # (batch_size, Max_atoms, dim_hidden_layer) -> (batch_size * Max_atoms, dim_hidden_layer)
            # for removing dummy atoms by using unsorted_segment_mean
            updated_atom_features = tf.reshape(updated_atom_features, ( updated_atom_features.shape[0] * updated_atom_features.shape[1], -1 )  )

            # (batch_size * Max_atoms, dim_hidden_layer) -> (batch_size, dim_hidden_layer)
            mean_atom_features = tf.math.unsorted_segment_mean( updated_atom_features, segment_ids = segment, num_segments = T_part.shape[0])

            # (batch_size, Max_atoms, dim_hidden_layer) -> (batch_size, dim_hidden_layer)
            #mean_atom_features = tf.reduce_mean(updated_atom_features, axis=1)

            # (batch_size, embed_size) 
            update3 = self.dense_layers[5](self.dense_layers[4](mean_atom_features))
            update4 = self.dense_layers[7](self.dense_layers[6](T_part))
            updated_T = T_part + layers.ReLU()(update3 + update4)

            # (batch_size, embed_size + dim_hidden_layer) 
            concat_vec = tf.concat([updated_T,mean_atom_features], -1)

        # readout
        h = self.readout_layers[0](concat_vec)
        for l in range(1,len(self.readout_layers)):
            h = self.readout_layers[l](h)

        if   equation == '':
            pred = h 

        if verbose:
            return pred, atom_features_each_layer, Attention_each_layer, T_part, updated_T 
        else:
            return pred

    def save_model(self, name):
        super(GAT_unc, self).save_weights(name)
    
    def load_model(self, name):
        super(GAT_unc, self).load_weights(name)

