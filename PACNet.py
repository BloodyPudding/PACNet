from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Convolution2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import SpatialDropout2D, LSTM, Reshape
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

def PACNet(num_classes,
            k_length=[10,10,10,10],
            F1_nums=[4,4,4,4],
            Chans=64,
            Samples=1000,
            dropoutRate=0.25,
            D=2,
            F2=8,
            norm_rate=0.25,
            dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D or Dropout, passed as a string.')
    inputs = []
    for i in range(len(k_length)):
        inputs.append(Input((Chans, Samples, 1)))
    raw_conv = []
    for i in range(len(k_length)):
        broadband_block = Conv2D(F1_nums[i], (1, k_length[i]), padding='same',
                                 use_bias=False,name=f'raw_conv_{i+1}')(inputs[i])
        broadband_block = BatchNormalization(axis=-1)(broadband_block)
        raw_conv.append(broadband_block)
    raw_conv = Concatenate(axis=-1)(raw_conv)
    depth_conv = DepthwiseConv2D((Chans, 1), use_bias=False,
                         depth_multiplier=D,name='depth_conv',
                         depthwise_constraint=max_norm(1.))(raw_conv)
    depth_conv = BatchNormalization(axis=1)(depth_conv)
    depth_conv = Activation('elu')(depth_conv)
    depth_conv = AveragePooling2D((1, 4))(depth_conv)
    depth_conv = dropoutType(dropoutRate)(depth_conv)
    separable_conv = SeparableConv2D(F2, (1, 16),name='separable_conv',
                             use_bias=False, padding='same')(depth_conv)
    separable_conv = BatchNormalization(axis=1)(separable_conv)
    separable_conv = Activation('elu')(separable_conv)
    separable_conv = AveragePooling2D((1, 8))(separable_conv)
    separable_conv = dropoutType(dropoutRate)(separable_conv)
    flatten = Flatten(name='flatten')(separable_conv)
    dense = Dense(num_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    return Model(inputs=inputs, outputs=softmax)