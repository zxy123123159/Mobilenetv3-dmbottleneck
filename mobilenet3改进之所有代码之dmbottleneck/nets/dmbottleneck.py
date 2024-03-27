from keras import backend as K
from keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dense,
                          DepthwiseConv2D, GlobalAveragePooling2D, Input,
                          Multiply, Reshape)
from keras.models import Model

from keras import layers
import numpy as np

#---------------------------------------#
#   激活函数 relu6
#---------------------------------------#
def relu6(x):
    return K.relu(x, max_value=6)

#---------------------------------------#
#   利用relu函数乘上x模拟sigmoid
#---------------------------------------#
def hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

#---------------------------------------#
#   用于判断使用哪个激活函数
#---------------------------------------#
def return_activation(x, activation):
    if activation == 'HS':
        x = Activation(hard_swish)(x)
    if activation == 'RE':
        x = Activation(relu6)(x)
    return x

#---------------------------------------#
#   卷积块
#   卷积 + 标准化 + 激活函数
#---------------------------------------#
def conv_block(inputs, filters, kernel, strides, activation):
    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization()(x)
    return return_activation(x, activation)

#---------------------------------------#
#   通道注意力机制单元
#   利用两次全连接算出每个通道的比重
#---------------------------------------#
def squeeze(inputs):
    input_channels = int(inputs.shape[-1])
    x = GlobalAveragePooling2D()(inputs)

    x = Dense(int(input_channels/4))(x)
    x = Activation(relu6)(x)

    x = Dense(input_channels)(x)
    x = Activation(hard_swish)(x)

    x = Reshape((1, 1, input_channels))(x)
    x = Multiply()([inputs, x])
    print(np.shape(inputs), np.shape(x))
    return x

#---------------------------------------#
#   逆瓶颈结构
#---------------------------------------#
def bottleneck(inputs, filters, kernel, up_dim, stride, attention, activation, alpha = 1):
    input_shape = K.int_shape(inputs)
    skip_flag = stride == 1 and input_shape[3] == filters

    #---------------------------------#
    #   part1 利用1x1卷积进行通道上升
    #---------------------------------#
    x = conv_block(inputs, int(up_dim), (1, 1), (1, 1), activation)

    #---------------------------------#
    #   part2 进行3x3的深度可分离卷积
    #---------------------------------#
    x = DepthwiseConv2D(kernel, strides=(stride, stride), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = return_activation(x, activation)

    #---------------------------------#
    #   引入注意力机制
    #---------------------------------#
    if attention:
        x = squeeze(x)

    #------------------------------------------#   
    #   part3 利用1x1卷积进行通道的下降
    #------------------------------------------#
    x = Conv2D(int(alpha * filters), (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if skip_flag:
        x = Add()([x, inputs])

    return x

def Dmbottleneck(input_shape = (224,224,3),classes = 1000):
    inputs = Input(input_shape)
    # 224,224,3 -> 112,112,16
    x = conv_block(inputs, 16, (3, 3), strides=(2, 2), activation='HS')
    x1 = bottleneck(x, 16, (3, 3), up_dim=16, stride=1, attention=False, activation='RE')

    # 112,112,16 -> 56,56,24
    x2 = bottleneck(x1, 24, (3, 3), up_dim=64, stride=2, attention=False, activation='RE')
    x3 = bottleneck(x2, 24, (3, 3), up_dim=72, stride=1, attention=False, activation='RE')
    
    # 56,56,24 -> 28,28,40
    x4 = bottleneck(x3, 40, (5, 5), up_dim=72, stride=2, attention=True, activation='RE')
    x5 = bottleneck(x4, 40, (5, 5), up_dim=120, stride=1, attention=True, activation='RE')
    x6 = bottleneck(x5, 40, (5, 5), up_dim=120, stride=1, attention=True, activation='RE')

    # 28,28,40 -> 14,14,80
    # x7 = bottleneck(x6, 80, (3, 3), up_dim=240, stride=2, attention=False, activation='HS')
    # x8 = bottleneck(x7, 80, (3, 3), up_dim=200, stride=1, attention=False, activation='HS')
    # x9 = bottleneck(x8, 80, (3, 3), up_dim=184, stride=1, attention=False, activation='HS')
    # x10 = bottleneck(x9, 80, (3, 3), up_dim=184, stride=1, attention=False, activation='HS')
    # 14,14,80 -> 14,14,112
    x11 = bottleneck(x6, 112, (3, 3), up_dim=480, stride=2, attention=True, activation='HS')
    x12 = bottleneck(x11, 112, (3, 3), up_dim=672, stride=1, attention=True, activation='HS')

    # 14,14,112 -> 7,7,160
    x13 = bottleneck(x12, 160, (5, 5), up_dim=672, stride=2, attention=True, activation='HS')
    x14 = bottleneck(x13, 160, (5, 5), up_dim=960, stride=1, attention=True, activation='HS')
    x15 = bottleneck(x14, 160, (5, 5), up_dim=960, stride=1, attention=True, activation='HS')
    x16 = bottleneck(x15, 160, (5, 5), up_dim=960, stride=1, attention=True, activation='HS')

    ############
    x17 = x16
    x1 = layers.GlobalMaxPooling2D(data_format='channels_last')(x1)
    #x1 = layers.Flatten()(x1)
    x2 = layers.GlobalMaxPooling2D(data_format='channels_last')(x2)
    #x2 = layers.Flatten()(x2)
    x3 = layers.GlobalMaxPooling2D(data_format='channels_last')(x3)
    #x3 = layers.Flatten()(x3)
    x4 = layers.GlobalMaxPooling2D(data_format='channels_last')(x4)
    #x4 = layers.Flatten()(x4)
    x5 = layers.GlobalMaxPooling2D(data_format='channels_last')(x5)
    #x5 = layers.Flatten()(x5)
    x6 = layers.GlobalMaxPooling2D(data_format='channels_last')(x6)
    #x6 = layers.Flatten()(x6)
    #7 8 9 10都被去掉了
    x11 = layers.GlobalAveragePooling2D(data_format='channels_last')(x11)
    #x11 = layers.Flatten()(x11)
    x12 = layers.GlobalAveragePooling2D(data_format='channels_last')(x12)
    #x12 = layers.Flatten()(x12)
    x13 = layers.GlobalAveragePooling2D(data_format='channels_last')(x13)
    #x13 = layers.Flatten()(x13)
    x14 = layers.GlobalAveragePooling2D(data_format='channels_last')(x14)
    #x14 = layers.Flatten()(x14)
    x15 = layers.GlobalAveragePooling2D(data_format='channels_last')(x15)
    #x15 = layers.Flatten()(x15)
    x16 = layers.GlobalAveragePooling2D(data_format='channels_last')(x16)
    #x16 = layers.Flatten()(x16)
    denseblock = layers.concatenate(   # 拼接
        [x1,x2,x3,x4,x5,x6,x11,x12,x13,x14,x15,x16],axis=-1
    )
    # print(np.shape(denseblock))  # 1048
    # 加入注意力机制
    def att(att_dim, name, inputs):
        V = inputs
        QK = Dense(att_dim)(inputs)
        QK = Activation("softmax", name=name)(QK)  # 激活函数可修改 softmax hard_swish relu6
        # print(np.shape(V), np.shape(QK))
        MV = Multiply()([V, QK])
        return (MV)
    denseblock = att(int(denseblock.shape[1]),"dense_attention",denseblock)  # 7840
    # 法一
    # denseblock = Dense(7840)(denseblock)
    # x = denseblock
    # x = layers.Reshape((7, 7, 160))(x)
    # 法二
    denseblock = Dense(1176)(denseblock)
    denseblock = layers.Reshape((7,7,24))(denseblock)
    denseblock = layers.Conv2D(int(160), (1, 1), strides=(1, 1), padding='same')(denseblock)
    x = denseblock
    ##
    x = layers.concatenate([x17, denseblock])
    ############

    # 7,7,160 -> 7,7,960
    x = conv_block(x, 960, (1, 1), strides=(1, 1), activation='HS')

    # 7,7,960 -> 1,1,960
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 960))(x)

    # 1,1,960 -> 1,1,1280
    x = Conv2D(1280, (1, 1), padding='same')(x)
    x = return_activation(x, 'HS')

    # 1,1,1280 -> classes
    x = Conv2D(classes, (1, 1), padding='same', activation='softmax')(x)
    x = Reshape((classes,))(x)

    model = Model(inputs, x)
    return model
    
if __name__ == "__main__":
    model = MobileNetv3_large_merge2()
    for i,layer in enumerate(model.layers):
        print(i,layer.name)
    print(model.summary())


