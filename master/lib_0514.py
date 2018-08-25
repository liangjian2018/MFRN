# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:19:24 2017

@author: Liang Jian
"""

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D,Conv2DTranspose,concatenate, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

def bn_relu_conv2d(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x

  
def denseblockC(x, nb_layers, fmnum,
               dropout_rate=None):

    list_feat = [x]
#    size =3
    for i in range(nb_layers):
#        x = BatchNormalization(axis=-1)(x)
#        if i>0:
#            size = 2
#        else:
#            size = 3
        x = Conv2D(fmnum, (3, 3), activation='relu', padding='same',kernel_initializer = 'he_normal')(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        list_feat.append(x)
        x = concatenate(list_feat, axis=-1)
#        nb_filter += growth_rate
    return x
    
def denseblockup(x, nb_layers, growth_rate,time,name):

    list_feat = []
#    filternumber
    for i in range(nb_layers):
        x = conv_block_up(x,growth_rate,time,name=name + '_block' + str(i + 1) )
        list_feat.append(x)
        if i > 0 : 
            x = concatenate(list_feat, axis=-1)
#        nb_filter += growth_rate
    return x
    
    
def dense_block(x, blocks, growth_rate,time, name):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, time,name=name + '_block' + str(i + 1))
    return x
    
def conv_block_up(x, growth_rate,time, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
#    x1 = BatchNormalization(axis=bn_axis, scale=False, name=name + '_0_bn')(x)
#    x1 = Activation('relu', name=name + '_0_relu')(x1)
#    x1 = Conv2D(4 * growth_rate, 1,padding='same', use_bias=False,
#                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, name=name + '_1_bn')(x)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(time*growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x1 = Dropout(0.2)(x1)
    return x1
    
def conv_block(x, growth_rate, time, name):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
#    x1 = BatchNormalization(axis=bn_axis, scale=False, name=name + '_0_bn')(x)
#    x1 = Activation('relu', name=name + '_0_relu')(x1)
#    x1 = Conv2D(4 * growth_rate, 1,padding='same', use_bias=False,
#                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis,  name=name + '_1_bn')(x)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(time*growth_rate, 3, padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x1 = Dropout(0.2)(x1)
    x = concatenate([x, x1],axis=bn_axis, name=name + '_concat')
    return x
    
def transition_down_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis,  name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,padding='same', use_bias=False,
               name=name + '_conv')(x)
    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x
    
def transition_up_block(x, reduction, name):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv2DTranspose(int(K.int_shape(x)[bn_axis] * reduction),(2,2),strides = (2,2),padding = 'same')(x)
#    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 1,padding='same', use_bias=False,
#               name=name + '_conv')(x)
#    x = UpSampling2D(size = (2,2),name = name + '_upsample')(x)
#    x = AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x
    
def filter_block(x, reduction,name):
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis,  name=name + '_0_bn')(x)
    x = Activation('relu', name=name + '_0_relu')(x)
    x = Conv2D(int(K.int_shape(x)[bn_axis] * reduction), 3,padding='same', use_bias=False,
                name=name + '_0_conv')(x)
#    x = BatchNormalization(axis=bn_axis, scale=False, name=name + '_1bn')(x)
#    x = Activation('relu', name=name + '_relu')(x)
#    x = Conv2D(filter_number, 3,padding='same', use_bias=False,
#               name=name + '_1conv')(x)
    x = Dropout(0.2)(x)
    return x
    
    
def dilated_dense_block(x, blocks, growth_rate,time, name):
#    count = 0
    for i in range(blocks):
#        count +=1
        x = dilated_conv_block(x, growth_rate, time,2,name=name + '_block' + str(i + 1))
    return x
    
def dilated_conv_block(x, growth_rate, time, d_rate,name):

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
#    x1 = BatchNormalization(axis=bn_axis, scale=False, name=name + '_0_bn')(x)
#    x1 = Activation('relu', name=name + '_0_relu')(x1)
#    x1 = Conv2D(4 * growth_rate, 1,padding='same', use_bias=False,
#                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis,  name=name + '_1_bn')(x)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv2D(time*growth_rate, 3, padding='same', use_bias=False, dilation_rate=(d_rate, d_rate),name=name + '_2_conv')(x1)
    x1 = Dropout(0.2)(x1)
    x = concatenate([x, x1],axis=bn_axis, name=name + '_concat')
    return x    
