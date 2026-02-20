"""
Model builder for Writer Identification
Supports multiple backbone architectures and training strategies
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Dropout
from tensorflow.keras.regularizers import l2

from custom_layers import (
    SpatialPyramidPooling,
    NetVLADLayer,
    L2Normalization,
    SelfAttentionBlock
)


def get_backbone_model(backbone_name, input_shape, weights='imagenet'):
    """
    Get the specified backbone model.
    
    Args:
        backbone_name: Name of the backbone ('resnet50', 'densenet201', 'xception', 'mobilenetv3')
        input_shape: Input shape for the model
        weights: 'imagenet', None, or path to weights file
        
    Returns:
        base_model: The backbone model
        preprocess_fn: Corresponding preprocessing function
    """
    backbone_name = backbone_name.lower()
    
    if backbone_name == 'resnet50':
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base_model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
        
    elif backbone_name == 'densenet201':
        from tensorflow.keras.applications import DenseNet201
        from tensorflow.keras.applications.densenet import preprocess_input
        base_model = DenseNet201(include_top=False, weights=weights, input_shape=input_shape)
        
    elif backbone_name == 'xception':
        from tensorflow.keras.applications import Xception
        from tensorflow.keras.applications.xception import preprocess_input
        base_model = Xception(include_top=False, weights=weights, input_shape=input_shape)
        
    elif backbone_name == 'mobilenetv3':
        from tensorflow.keras.applications import MobileNetV3Large
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
        base_model = MobileNetV3Large(
            include_top=False, 
            weights=weights, 
            input_shape=input_shape, 
            pooling=None, 
            alpha=1.0
        )
        
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    return base_model, preprocess_input


def set_trainability(base_model, training_mode, num_trainable_layers=None):
    """
    Set trainability of backbone layers based on training mode.
    
    Args:
        base_model: The backbone model
        training_mode: 'frozen', 'scratch', 'finetune_last_n', 'finetune_all'
        num_trainable_layers: Number of layers to make trainable (for 'finetune_last_n')
    """
    if training_mode == 'frozen':
        # Freeze all layers
        for layer in base_model.layers:
            layer.trainable = False
            
    elif training_mode == 'scratch':
        # Train all layers
        for layer in base_model.layers:
            layer.trainable = True
            
    elif training_mode == 'finetune_all':
        # Train all layers
        for layer in base_model.layers:
            layer.trainable = True
            
    elif training_mode == 'finetune_last_n':
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
        
        # Make last N layers trainable
        if num_trainable_layers is None:
            raise ValueError("num_trainable_layers must be specified for 'finetune_last_n' mode")
        
        total_layers = len(base_model.layers)
        start_idx = max(0, total_layers - num_trainable_layers)
        
        for layer in base_model.layers[start_idx:]:
            layer.trainable = True
            
    else:
        raise ValueError(f"Unsupported training mode: {training_mode}")


def build_writer_identification_model(
    backbone_name='resnet50',
    input_shape=(224, 224, 3),
    num_clusters=64,
    num_classes=179,
    training_mode='frozen',
    num_trainable_layers=None,
    use_attention=False,
    weights='imagenet'
):
    """
    Build the writer identification model with specified configuration.
    
    Args:
        backbone_name: Name of the backbone architecture
        input_shape: Input image shape
        num_clusters: Number of clusters for NetVLAD
        num_classes: Number of output classes
        training_mode: Training strategy ('frozen', 'scratch', 'finetune_last_n', 'finetune_all')
        num_trainable_layers: Number of layers to finetune (for 'finetune_last_n')
        use_attention: Whether to use attention mechanism
        weights: Backbone weights ('imagenet' or None)
        
    Returns:
        model: Compiled Keras model
        preprocess_fn: Preprocessing function for the backbone
    """
    # Handle weights based on training mode
    if training_mode == 'scratch':
        weights = None
    
    # Get backbone model
    base_model, preprocess_fn = get_backbone_model(backbone_name, input_shape, weights)
    
    # Set trainability
    set_trainability(base_model, training_mode, num_trainable_layers)
    
    # Build the model
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    
    print(f'Shape after base_model: {x.shape}')
    
    # 1x1 Convolution to reduce channels to 64
    x = Conv2D(64, kernel_size=(1, 1), activation='relu', kernel_regularizer=l2(1e-4))(x)
    print(f'Shape after Conv2D: {x.shape}')
    
    # L2-normalize to obtain compact local descriptors
    x = L2Normalization(axis=-1)(x)
    print(f'Shape after L2-normalization: {x.shape}')
    
    # Optional attention mechanism
    if use_attention:
        # Store the backbone features for cross-attention later
        x_backbone = x  # Shape: (batch_size, H, W, 64)
        H_shape = x.shape[1] if x.shape[1] is not None else 7
        W_shape = x.shape[2] if x.shape[2] is not None else 7
        
        # Flatten for self-attention
        from tensorflow.keras.layers import Reshape
        x_backbone_flat = Reshape((H_shape * W_shape, 64))(x_backbone)
        print(f'Shape after reshaping for self-attention (backbone): {x_backbone_flat.shape}')
        
        # First self-attention block (after backbone features)
        x = SelfAttentionBlock(num_heads=6, key_dim=32)(x)
        print(f'Shape after first Self-Attention: {x.shape}')
    
    # Spatial Pyramid Pooling
    x = SpatialPyramidPooling(pool_sizes=[1, 2, 4])(x)
    print(f'Shape after SpatialPyramidPooling: {x.shape}')
    
    # Optional attention mechanism (applied after SPP)
    if use_attention:
        # Second self-attention block (after SPP features)
        x = SelfAttentionBlock(num_heads=6, key_dim=32)(x)
        print(f'Shape after second Self-Attention: {x.shape}')
    
    # NetVLAD Layer
    x_vlad = NetVLADLayer(num_clusters=num_clusters)(x)
    print(f'Shape after NetVLADLayer: {x_vlad.shape}')
    
    # Optional cross-attention (between VLAD and backbone features)
    if use_attention:
        from tensorflow.keras.layers import Reshape
        
        # Calculate D_vlad based on SPP output (64 * 3 = 192 channels after SPP)
        D_vlad = 192
        
        # Reshape x_vlad for cross-attention: (batch_size, num_clusters, D_vlad)
        x_vlad_reshaped = Reshape((num_clusters, D_vlad))(x_vlad)
        print(f'Shape after reshaping x_vlad for cross-attention: {x_vlad_reshaped.shape}')
        
        # Project backbone features to match D_vlad dimensions
        x_backbone_flat_proj = Dense(D_vlad, activation='relu', name='proj_backbone_flat')(x_backbone_flat)
        print(f'Shape after projecting backbone features for cross-attention: {x_backbone_flat_proj.shape}')
        
        # Project VLAD features for query alignment
        x_vlad_proj = Dense(D_vlad, activation='relu', name='proj_vlad')(x_vlad_reshaped)
        
        # Cross-Attention: VLAD queries attend to backbone features
        cross_attn_layer = layers.MultiHeadAttention(num_heads=6, key_dim=32)
        x_cross_attn = cross_attn_layer(
            query=x_vlad_proj, 
            key=x_backbone_flat_proj, 
            value=x_backbone_flat_proj
        )
        print(f'Shape after cross-attention: {x_cross_attn.shape}')
        
        # Flatten the cross-attended features
        x_cross_attn_flat = Reshape((num_clusters * D_vlad,))(x_cross_attn)
        print(f'Shape after flattening cross-attention output: {x_cross_attn_flat.shape}')
        
        # L2-normalize the cross-attention output
        x = L2Normalization(axis=-1)(x_cross_attn_flat)
        print(f'Shape after L2-normalization (cross-attention): {x.shape}')
    else:
        # No attention: just L2-normalize the NetVLAD output
        x = L2Normalization(axis=-1)(x_vlad)
        print(f'Shape after final L2-normalization: {x.shape}')
    
    # Fully connected layers to produce the embedding vector
    x = Dense(512, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = L2Normalization(axis=-1)(x)
    
    # Classification head
    # Use float32 for the output layer (required for mixed precision training)
    classification_output = Dense(
        num_classes, 
        activation='softmax', 
        name='classification_output',
        dtype='float32'  # Force float32 for numerical stability
    )(x)
    print(f'Shape after classification_output: {classification_output.shape}')
    
    # Define the model
    model = Model(inputs, classification_output)
    
    return model, preprocess_fn


if __name__ == '__main__':
    # Test the model builder
    print("Testing model builder...")
    
    model, preprocess_fn = build_writer_identification_model(
        backbone_name='resnet50',
        training_mode='frozen',
        use_attention=False
    )
    
    model.summary()
    print("\nModel built successfully!")
