"""
Custom Keras layers for Writer Identification Model
"""
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class SpatialPyramidPooling(layers.Layer):
    """Spatial Pyramid Pooling layer"""
    
    def __init__(self, pool_sizes, **kwargs):
        super(SpatialPyramidPooling, self).__init__(**kwargs)
        self.pool_sizes = pool_sizes
        self.pool_layers = [
            tf.keras.layers.MaxPooling2D(
                pool_size=(size, size),
                strides=(size, size),
                padding='same'
            ) for size in self.pool_sizes
        ]

    def call(self, inputs):
        h = tf.shape(inputs)[1]
        w = tf.shape(inputs)[2]

        pooled_outputs = [pool(inputs) for pool in self.pool_layers]
        # Upsample all pooled outputs back to (h, w)
        upsampled = [tf.image.resize(pooled, (h, w), method='bilinear') 
                     for pooled in pooled_outputs]
        # Concatenate along the channel dimension
        output = tf.concat(upsampled, axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, h, w, c)
        if input_shape[3] is None:
            c_new = None
        else:
            c_new = input_shape[3] * len(self.pool_sizes)
        return (input_shape[0], input_shape[1], input_shape[2], c_new)

    def get_config(self):
        config = super(SpatialPyramidPooling, self).get_config()
        config.update({
            'pool_sizes': self.pool_sizes,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class NetVLADLayer(layers.Layer):
    """NetVLAD layer for aggregation"""
    
    def __init__(self, num_clusters, **kwargs):
        super(NetVLADLayer, self).__init__(**kwargs)
        self.num_clusters = num_clusters

    def build(self, input_shape):
        self.feature_dim = input_shape[-1]
        # Initialize cluster centers with Xavier/Glorot initialization for better convergence
        self.cluster_centers = self.add_weight(
            shape=(self.num_clusters, self.feature_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            name="cluster_centers"
        )
        super(NetVLADLayer, self).build(input_shape)

    def call(self, inputs):
        # Inputs shape: (batch_size, H, W, D)
        batch_size = tf.shape(inputs)[0]
        H = tf.shape(inputs)[1]
        W = tf.shape(inputs)[2]
        D = self.feature_dim

        # Reshape inputs to (batch_size, H*W, D)
        inputs_reshaped = tf.reshape(inputs, [batch_size, -1, D])

        # L2-normalize along the feature dimension
        inputs_norm = tf.nn.l2_normalize(inputs_reshaped, axis=-1)

        # Compute similarities (assignments) to cluster centers
        similarities = tf.matmul(inputs_norm, self.cluster_centers, transpose_b=True)
        assignments = tf.nn.softmax(similarities, axis=-1)

        # Compute residuals
        residuals = tf.expand_dims(inputs_reshaped, axis=2) - self.cluster_centers
        residuals_weighted = residuals * tf.expand_dims(assignments, axis=-1)

        # Aggregate residuals
        vlad = tf.reduce_sum(residuals_weighted, axis=1)

        # Flatten and L2-normalize the VLAD descriptors
        vlad = tf.reshape(vlad, [batch_size, -1])
        vlad = tf.nn.l2_normalize(vlad, axis=-1)

        return vlad

    def compute_output_shape(self, input_shape):
        # input_shape: (batch_size, H, W, D)
        batch_size = input_shape[0]
        output_dim = self.num_clusters * input_shape[3]
        return (batch_size, output_dim)

    def get_config(self):
        config = super(NetVLADLayer, self).get_config()
        config.update({
            'num_clusters': self.num_clusters,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class L2Normalization(layers.Layer):
    """L2 Normalization layer"""
    
    def __init__(self, axis=-1, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(L2Normalization, self).get_config()
        config.update({
            'axis': self.axis,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class SelfAttentionBlock(layers.Layer):
    """
    Self-Attention block using Multi-Head Attention
    Matches the original implementation with Reshape -> LayerNorm -> MultiHeadAttention -> LayerNorm -> Reshape
    """
    
    def __init__(self, num_heads=6, key_dim=32, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
    
    def build(self, input_shape):
        # input_shape: (batch_size, H, W, channels)
        self.H = input_shape[1]
        self.W = input_shape[2]
        self.channels = input_shape[3]
        
        # Create layers
        self.reshape_to_seq = layers.Reshape((self.H * self.W, self.channels))
        self.layer_norm_1 = layers.LayerNormalization()
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )
        self.layer_norm_2 = layers.LayerNormalization()
        self.reshape_back = layers.Reshape((self.H, self.W, self.channels))
        
        super(SelfAttentionBlock, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, H, W, channels)
        
        # Reshape to sequence format: (batch_size, H*W, channels)
        x = self.reshape_to_seq(inputs)
        
        # Apply Layer Normalization before attention
        x = self.layer_norm_1(x)
        
        # Apply Multi-Head Self-Attention
        x_attn = self.multi_head_attention(query=x, key=x, value=x)
        
        # Apply Layer Normalization after attention
        x_attn = self.layer_norm_2(x_attn)
        
        # Reshape back to spatial format: (batch_size, H, W, channels)
        output = self.reshape_back(x_attn)
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(SelfAttentionBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
        })
        return config
