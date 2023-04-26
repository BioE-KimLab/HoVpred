import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_MKL_REUSE_PRIMITIVE_MEMORY'] = '0'
from tensorflow.keras import layers
from dgl.nn.tensorflow import GATConv
import gc
import sys

class GAT(tf.keras.Model):
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
        super(GAT, self).__init__()
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

        self.readout_layers = [layers.Dense(32, activation='relu'), layers.Dense(16, activation='relu'), layers.Dense(1,activation='relu')  ]
        #self.readout_layers = [layers.Dense(32, activation='relu'), layers.Dense(16, activation='relu'), layers.Dense(1)  ]


    def call(self, features, g, segment, Max_atoms, equation, num_mols, training, verbose=False): 
        # One-hot atom feature vectors as initial values
        # Note: The DGL graph 'g' contains ALL molecular graphs in the batch in one variable.
        # Thus the shape of 'features' is: (batch_size * Max_atoms, dim_one_hot_vector)
        h = features

        for l in range(self.num_layers):
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

        num_seg = updated_atom_features.shape[0]

        updated_atom_features = tf.repeat(input = updated_atom_features, repeats = num_mols, axis = 0)

        updated_atom_features = tf.reshape(updated_atom_features, (updated_atom_features.shape[0], Max_atoms, -1))
        updated_atom_features = tf.reshape(updated_atom_features, ( updated_atom_features.shape[0] * updated_atom_features.shape[1], -1 )  )
        concat_vec = tf.math.unsorted_segment_mean( updated_atom_features, segment_ids = segment, num_segments = num_seg )

        # readout
        h = self.readout_layers[0](concat_vec)
        for l in range(1,len(self.readout_layers)):
            h = self.readout_layers[l](h)

        logits = tf.reshape(h, [-1])
        return logits

    def save_model(self, name):
        super(GAT, self).save_weights(name)
        #self.save(name)
    
    def load_model(self, name):
        super(GAT, self).load_weights(name)



