from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

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

def final_model(input_dim, conv_layers, filters, kernel_size, conv_stride,
    conv_border_mode, units, activation, recur_layers, dropout_w=0.0, dropout_u=0.0, output_dim=29):
    """ Build a deep network for speech 
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
    # Add the rest of the 1D convolutional layers
    for i in range(1, conv_layers):
        dilation *= 2  # dilation is doubled with every layer
        conv_1d = Conv1D(filters, kernel_size,
                         strides=conv_stride,
                         padding=conv_border_mode,
                         activation='relu',
                         dilation_rate=dilation if conv_border_mode == 'casual' else 1,
                         name='conv1d-' + str(i))(conv_1d)

    # Add first recurrent layer with batch normalization
    bidir_rnn = Bidirectional(GRU(units, activation=activation,
                                  return_sequences=True, implementation=2,
                                  dropout=dropout_w, recurrent_dropout=dropout_u),
                              merge_mode='concat', name='birnn-0')(conv_1d)
    bn_rnn = BatchNormalization()(bidir_rnn)
    # Add the rest of the recurrent layers with batch normalization
    for i in range(1, recur_layers):
        bidir_rnn = Bidirectional(GRU(units, activation=activation,
                                      return_sequences=True, implementation=2,
                                      dropout=dropout_w, recurrent_dropout=dropout_u),
                                  merge_mode='concat', name='birnn-' + str(i))(bn_rnn)
        bn_rnn = BatchNormalization()(bidir_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
#    model.output_length = lambda x: final_output_length(conv_layers, x, kernel_size, conv_border_mode, conv_stride)
    # with dilated convolutions, the input is padded so that the output size is same as the original input size
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_output_length(conv_layers, input_length, filter_size, border_mode, stride, dilation=1):
    """ Apply cnn_output_length function conv_layers-times in order to calculate output size when using more
        convolutional layers
    """

    output_length = input_length

    for i in range(conv_layers):
        output_length = cnn_output_length(output_length, filter_size, border_mode, stride, dilation)

    return output_length
