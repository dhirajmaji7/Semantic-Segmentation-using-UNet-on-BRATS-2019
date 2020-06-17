from keras.layers import Conv2D,Activation,Input,Dropout,Reshape,MaxPooling2D,UpSampling2D,Conv2DTranspose,Concatenate,flatten
from keras.layers.normalization import BatchNormalization 
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.utils import *
from keras import regularizers
from keras import backend as K
import numpy as np


def UNet(n_classes=5, im_sz=240, n_channels=4, n_filters_start=64, growth_factor=2):
    
    droprate=0.2
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels))
    #inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #pool1 = Dropout(droprate)(pool1)

    n_filters *= growth_factor
    pool1 = BatchNormalization()(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #pool2 = Dropout(droprate)(pool2)

    n_filters *= growth_factor
    pool2 = BatchNormalization()(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #pool3 = Dropout(droprate)(pool3)

    n_filters *= growth_factor
    pool3 = BatchNormalization()(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #pool4 = Dropout(droprate)(pool4)

    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv5)


    n_filters //= growth_factor
    up6 = Concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    up6 = BatchNormalization()(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv6)
    #conv6 = Dropout(droprate)(conv6)

    n_filters //= growth_factor
    up7 = Concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    up7 = BatchNormalization()(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv7)
    #conv7 = Dropout(droprate)(conv7)

    n_filters //= growth_factor
    up8 = Concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    up8 = BatchNormalization()(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv8)
    #conv8 = Dropout(droprate)(conv8)

    n_filters //= growth_factor
    up9 = Concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(n_filters, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model

# Dice loss for each slice and then calculating the mean dice score

def dice_coef_0(y_true, y_pred,smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,0])
    y_pred_f = K.flatten(y_pred[:,:,:,0])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_1(y_true, y_pred,smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,1])
    y_pred_f = K.flatten(y_pred[:,:,:,1])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_2(y_true, y_pred,smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,2])
    y_pred_f = K.flatten(y_pred[:,:,:,2])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_3(y_true, y_pred,smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,3])
    y_pred_f = K.flatten(y_pred[:,:,:,3])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_4(y_true, y_pred,smooth=0.000001):
    y_true_f = K.flatten(y_true[:,:,:,4])
    y_pred_f = K.flatten(y_pred[:,:,:,4])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_score(y_true, y_pred):
    d0 = dice_coef_0(y_true, y_pred,smooth=0.000001)
    d1 = dice_coef_1(y_true, y_pred,smooth=0.000001)
    d2 = dice_coef_2(y_true, y_pred,smooth=0.000001)
    #d3 = dice_coef_3(y_true, y_pred,smooth=0.000001)
    d4 = dice_coef_4(y_true, y_pred,smooth=0.000001)

    dice_mean = (d0+d1+d2+d4)/4
    return dice_mean

def dice_loss(y_true, y_pred):
    return 1-dice_score(y_true, y_pred)

model = UNet()
model.compile(optimizer=Adam(lr=0.0001), loss= dice_loss,  
              metrics= [dice_coef_0, dice_coef_1, dice_coef_2, dice_coef_4, dice_score])
model.summary()
