import keras
from keras.layers import *
from keras.models import Model

conv_params = {
    'kernel_size': (3, 3, 3),
    'strides': (1, 1, 1),
    'padding': 'same',  # TODO: consider 'valid' (no padding),
                        # as suggested by paper.
    'activation': 'relu',
    # TODO: experiment with regularizers and initializers.
    'kernel_initializer': 'he_normal',
    'kernel_regularizer': keras.regularizers.l2(.001)
}

def build_unet(n_classes, depth=4, base_filters=32):
    inputs = Input((None, None, None, 1))
    x = inputs

    n_filters = base_filters

    # Layers tat will be used in up-convolution
    layer_outputs = []
    # TODO: try adding batch normalization layers.

    # Convolution layers
    for layer in range(depth - 1):
        x = Conv3D(filters=n_filters, **conv_params)(x)
        x = Conv3D(filters=n_filters, **conv_params)(x)
        layer_outputs.append(x)
        x = MaxPooling3D(pool_size=(2,2,2))(x)
        n_filters *= 2

    # Bottom layers
    x = Conv3D(filters=n_filters, **conv_params)(x)
    x = Conv3D(filters=n_filters, **conv_params)(x)

    # Transposed Convolution layers (up-convolution)
    for layer in reversed(range(depth-1)):
        n_filters //= 2
        x = Conv3DTranspose(filters=n_filters, kernel_size=(2,2,2),
                            strides=(2,2,2))(x)
        x = concatenate([x, layer_outputs.pop()])
        x = Conv3D(filters=n_filters, **conv_params)(x)
        x = Conv3D(filters=n_filters, **conv_params)(x)

    # Final layer
    x = Conv3D(filters=n_classes, kernel_size=(1,1,1), padding='same',
               activation='softmax')(x)


    model = Model(inputs, x)
    return model