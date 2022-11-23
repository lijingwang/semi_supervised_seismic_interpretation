"""Module provides test cases for configuring module architectures for experimentation."""

from tensorflow.keras import regularizers


def get_test_cases(test):
    """This function returns configurations for model testing

    This function is designed to enable the systematic testing of the network
    configuration to aid in experimentation.  Within the experimental network
    with different flags and settings whos parameters are set here.  The following
    are the possible settings"

    test_dict = {
        'depth': any positive int value, but there are limitations on max depth due to image size
        'base_filter': an positive int value and sets the base number of filters with
                       which to build the network.  The default is considered 64
        'res_layers': True/False boolean.  When true, this flag will build the network with
                      the standard resnet skip connections detailed in He et al. 2015
                      (https://arxiv.org/pdf/1512.03385.pdf).
        'bn_position': valid flags are 'before' or 'after'.  This set the batch norm at the beginning
                       or end of the conv block.
        'bn_momentum': The momentum factor to use in the batch norm calculation.  Default is 0.99.
        'pooling': valid settings are 'max' or 'ave'.  'max' will do max pooling and 'ave' will do
                   average pooling.
        'expansion': valid settings are 'conv2dtranspose' or 'upsampling'. 'conv2dtranspose' will set
                     the expansion to use a transposed convolutional layer, i.e. deconvolution, while
                     'upsampling' will use an upsampling layer with the default nearest neighbors
                     interpolation.
        'time_shift': True/False boolean.  This flag will activate a data augmentation in the data
                      feeding that adds random noise and a random time shift to the data signal.
        'use_limits': True/False boolean.  When True, this flag will use a limit formulation of the
                      problem and do the segmentation on the fascia interfaces rather than training
                      to learn the fascias themselves.  ***If True, n_class also needs to be set to
                      n_class=1.***
        'regularizer': a Keras function from tensorflow.keras.regularizers to pass to the convolutions.
    }
    Args:
        'test': str, a string value corresponding to the test number.
    Returns:
        'test_dict': dict, a dictionary containing all of the test parameters.
    """
    if test == 'test_UNet_semi_supervised':
        test_dict = {
            'depth': 5,
            'base_filter': 16,
            'res_layers': False,
            'bn_position': 'before',
            'bn_momentum': 0.9,
            'pooling': 'ave',
            'expansion': 'conv2dtranspose',
            'time_shift': False,
            'use_limits': False,
            'trainable': False, ## freeze the encoder
            'dropout': True,
            'dropout_rate': 0.3
        }
    elif test == 'test_UNet_supervised':
        test_dict = {
            'depth': 5,
            'base_filter': 16,
            'res_layers': False,
            'bn_position': 'before',
            'bn_momentum': 0.9,
            'pooling': 'ave',
            'expansion': 'conv2dtranspose',
            'time_shift': False,
            'use_limits': False,
            'trainable': True,
            'dropout': True,
            'dropout_rate': 0.3
        }
    return test_dict
