# Batch norm note: training = True @ training mode, training = False @ inference model, nothing changed

"""U-Net computer vision architecture."""

from tensorflow.keras import Input, layers, activations
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import sys
sys.path.append('../')
import model.test_cases as test_cases

model_input = 'images'
model_output = 'label' #probabilities

def conv2d_norm(x, filters, bn_position, bn_momentum, regularizer):
    """Initializes a TensorFlow 2D convolution layer with batch normalization.

    Args:
        inputs: Tensor input.
        filters: Int, the dimensionality of the output space (i.e. the number of filters in the convolution).

    Returns:
        Output tensor.

    """
    if bn_position == 'before':
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer)(x)
    else:
        x = layers.Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer)(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)

    return x


def contracting_block(x, filters, pooling, res_layers, bn_position,
                      bn_momentum, first_layer, regularizer, trainable):
    """Initializes a U-Net contracting block:
        - One max pooling layer
        - Two separable 2D convolution layers followed by batch normalization

    Args:
        inputs: Tensor input.
        filters: Int, the dimensionality of the output space (i.e. the number of filters in the convolution).

    Returns:
        Output tensor.

    """
    if not first_layer:
        if pooling == 'max':
            x = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid',trainable = trainable)(x)
        elif pooling == 'ave':
            x = layers.AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid',trainable = trainable)(x)

    if res_layers:
        y = layers.Conv2D(filters=filters,
                          kernel_size=(1, 1),
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,trainable = trainable)(x)
        y = layers.BatchNormalization(momentum=bn_momentum)(y)

    if bn_position == 'before':
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,trainable = trainable)(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,trainable = trainable)(x)
        if res_layers:
            x = layers.Add()([x, y])
        x = activations.relu(x)
    else:
        x = layers.Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,trainable = trainable)(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)
        x = layers.Conv2D(filters=filters,
                          kernel_size=(3, 3),
                          padding='same',
                          activation='relu',
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer, trainable = trainable)(x)
        if res_layers:
            x = layers.Add()([x, y])
        x = activations.relu(x)
        x = layers.BatchNormalization(momentum=bn_momentum)(x)

    return x


def expanding_block(x, x_stack, filters, bn_position, bn_momentum, expansion, regularizer, merge = True):
    """Initializes a U-Net expanding block:
        - One transpose 2D convolution layer
        - Two separable 2D convolution layers followed by batch normalization

    Args:
        inputs: Tensor input.
        stack_inputs: Tensor input to be stacked after transpose 2D convolution.
        filters: Int, the dimensionality of the output space (i.e. the number of filters in the convolution).

    Returns:
        Output tensor.

    """

    if expansion == 'conv2dtranspose':
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2),
                                   strides=2, padding='valid')(x)
    elif expansion == 'upsampling':
        x = layers.UpSampling2D(size=(2, 2), interpolation='nearest')(x)

    if (merge):
        x = layers.Concatenate(axis=3)([x_stack, x])

    x = conv2d_norm(x, filters, bn_position, bn_momentum, regularizer)
    return conv2d_norm(x, filters, bn_position, bn_momentum, regularizer)


def model(c, k, activation = 'sigmoid', test_case = 'test_UNet_supervised'):
    """Initializes a U-Net model for image segmentation:
        - One contracting path with 4 contraction / convolution blocks
        - One expanding path with 4 expansion / convolution blocks

    Args:
        c: Int, the dimensionality of the input space (i.e. the number of channels in the input images).
        k: Int, the dimensionality of the output space (i.e. the number of classes in the segmentation).
        activation: Str, the activation method (example: relu, softmax, ...)
        test_case: Str, the configuration test case of the model

    Returns:
        Keras implementation of U-Net model.

    """

    # define model features
    model_dict = test_cases.get_test_cases(test_case)

    depth = model_dict.get('depth', 5)
    base_filter = model_dict.get('base_filter', 64)
    res_layers = model_dict.get('res_layers', False)
    bn_position = model_dict.get('bn_position', 'before')
    bn_momentum = model_dict.get('bn_momentum', 0.9)
    pooling = model_dict.get('pooling', 'max')
    expansion = model_dict.get('expansion', 'conv2dtranspose') #conv3dtranspose
    regularizer = model_dict.get('regularizer', None)
    dropout = model_dict.get('dropout', False)
    dropout_rate = model_dict.get('dropout_rate', 0.2)
    trainable = model_dict.get('trainable', True)
    
    
    downsample_layers = ['down'+str(i) for i in range(depth)]
    upsample_layers = ['up'+str(i) for i in range(depth-1)]
    
    
    
    # input layer
    with K.name_scope('input'):
        X = Input(shape=(None, None, c), name=model_input)

    # build contraction branch
    first_layer = True
    down_blocks = []
    filter_list = []
    for i, layer in enumerate(downsample_layers):
        with K.name_scope(layer):
            filters = base_filter*2**i
            filter_list.append(filters)
            if dropout:
                down_blocks.append(layers.Dropout(rate=dropout_rate)(contracting_block(X if i == 0 else down_blocks[-1],
                                                                                       filters=filters,
                                                                                       pooling=pooling,
                                                                                       res_layers=res_layers,
                                                                                       bn_position=bn_position,
                                                                                       bn_momentum=bn_momentum,
                                                                                       first_layer=first_layer,
                                                                                       regularizer=regularizer,
                                                                                       trainable = trainable)))
            else:
                down_blocks.append(contracting_block(X if i == 0 else down_blocks[-1],
                                                     filters=filters,
                                                     pooling=pooling,
                                                     res_layers=res_layers,
                                                     bn_position=bn_position,
                                                     bn_momentum=bn_momentum,
                                                     first_layer=first_layer,
                                                     regularizer=regularizer,
                                                     trainable = trainable))
            first_layer = False


    # expanding branch
    up_block = down_blocks.pop()
    filter_list = filter_list[:-1]
    for i, layer in enumerate(upsample_layers):
        with K.name_scope(layer):
            filters = filter_list.pop()
            if i < depth-3: 
                up_block = expanding_block(up_block,
                                           down_blocks.pop(),
                                           filters=filters,
                                           bn_position=bn_position,
                                           bn_momentum=bn_momentum,
                                           expansion=expansion,
                                           regularizer=regularizer, merge = True)                
            else: # Not connect the low resolution layers
                up_block = expanding_block(up_block,
                                           down_blocks.pop(),
                                           filters=filters,
                                           bn_position=bn_position,
                                           bn_momentum=bn_momentum,
                                           expansion=expansion,
                                           regularizer=regularizer, merge = False)
            
            if dropout:
                up_block = layers.Dropout(rate=dropout_rate)(up_block)

                
    # output layer
    with K.name_scope('output'):
        prob = layers.Conv2D(filters=k,
                             kernel_size=(1, 1),
                             strides=1,
                             padding='same',
                             activation=activation,
                             name=model_output)(up_block)
    return Model(inputs=X, outputs=prob)
