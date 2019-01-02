from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, MaxPooling1D, Dense, Input,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, activation, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    # Add first recurrent layer with batch normalization
    simp_rnn = GRU(units, activation=activation,
                   return_sequences=True, implementation=2, name='rnn-0')(input_data)
    bn_rnn = BatchNormalization()(simp_rnn)
    # Add the rest of the recurrent layers
    for i in range(1, recur_layers):
        simp_rnn = GRU(units, activation=activation,
                       return_sequences=True, implementation=2, name='rnn-' + str(i))(bn_rnn)
        bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation=activation,
                                  return_sequences=True, implementation=2, name='rnn'),
                              merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def batch_norm(cur_layer, inp_layer=None):
    """Return Batch Normalized layer cur_layer connected to previous inp_layer"""

    output = cur_layer(inp_layer)
    return BatchNormalization()(output)


def final_model(input_dim, conv_layers, filters, kernel_size, pooling_size, conv_stride,
    conv_border_mode, units, activation, recur_layers, dropout_w=0.0, dropout_u=0.0, output_dim=29):
    """ Build a deep network for speech
    :param input_dim: number of features in each time step of the input (13 for MFCC, 161 for Spectrogram)
    :param conv_layers: number of 1D convolutional layers followed by Max Pooling
    :param filters: number of filters in each convolutional layer
    :param kernel_size: temporal window size of each filter in 1D convolutional layers
    :param pooling_size: temporal window size and window stride in Max Pooling layer
    :param conv_stride: stride of the window in 1D convolutional layers (must be 1 when using dilated convolutions)
    :param conv_border_mode: 'valid' for standard convolution, 'same' for padding to same size as input with zeros,
    'causal' for dilated convolutions
    :param units: number of hidden units in recurrent cells
    :param activation: activation function applied to the output of the RNNs
    :param recur_layers: number of stacked RNN layers
    :param dropout_w: RNN layer input dropout
    :param dropout_u: RNN layer recurrent dropout
    :param output_dim: number of classes at the output (default: 26 characters + 1 apostrophe, 1 space, 1 blank)
    """
    dilation = 1

    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    # Add first 1D convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     dilation_rate=dilation,
                     name='conv1d-0')(input_data)
    pool_1d = MaxPooling1D(pool_size=pooling_size)(conv_1d)
    # Add the rest of the 1D convolutional layers
    for i in range(1, conv_layers):
        if conv_border_mode == 'causal':
            dilation *= 2  # dilation is doubled with every layer
        conv_1d = Conv1D(filters, kernel_size,
                         strides=conv_stride,
                         padding=conv_border_mode,
                         activation='relu',
                         dilation_rate=dilation,
                         name='conv1d-' + str(i))(pool_1d)
        pool_1d = MaxPooling1D(pool_size=pooling_size)(conv_1d)

    # Add first recurrent layer with batch normalization
    bidir_rnn = batch_norm(Bidirectional(GRU(units, activation=activation,
                                              return_sequences=True, implementation=2,
                                              dropout=dropout_w, recurrent_dropout=dropout_u),
                                         merge_mode='concat', name='birnn-0'),
                           pool_1d)
    # Add the rest of the recurrent layers with batch normalization
    for i in range(1, recur_layers):
        bidir_rnn = batch_norm(Bidirectional(GRU(units, activation=activation,
                                                  return_sequences=True, implementation=2,
                                                  dropout=dropout_w, recurrent_dropout=dropout_u),
                                             merge_mode='concat', name='birnn-' + str(i)),
                               bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: final_output_length(conv_layers, x, kernel_size, pooling_size,
                                                        conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_pooling_output_length(input_length, filter_size, pooling_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        pooling_size (int): Size of the 1D max pooling layer kernel
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'valid':
        output_length = (input_length - dilated_filter_size + stride) // stride // pooling_size
    else:
        output_length = input_length // pooling_size
    return output_length


def final_output_length(conv_layers, input_length, filter_size, pooling_size, border_mode, stride, dilation=1):
    """ Apply cnn_output_length function conv_layers-times in order to calculate output size when using more
        convolutional layers
    """

    output_length = input_length

    for i in range(conv_layers):
        output_length = cnn_pooling_output_length(output_length, filter_size, pooling_size,
                                                  border_mode, stride, dilation)

    return output_length
