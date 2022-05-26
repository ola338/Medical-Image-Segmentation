from keras.models import Model
from keras.layers import Conv2DTranspose, Input, concatenate, Dropout, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Lambda, add, multiply
from keras import backend as K
from keras.optimizers import RMSprop
from losses import bce_dice_loss, dice_loss
from metrics import iou_score


def gatingsignal(input, features):
    x = Conv2D(features, (1, 1), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def attention_block(x, gating, features):
    theta_x = Conv2D(features, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(x)

    X_shape = K.int_shape(x)
    g_shape = K.int_shape(gating) 
    theta_shape = K.int_shape(theta_x)

    phi_g = Conv2D(features, (1, 1), kernel_initializer='he_normal', padding='same')(gating)

    up_g_stride_shape = (theta_shape[1] // g_shape[1], theta_shape[2] // g_shape[2])
    upsample_g = Conv2DTranspose(features, (3, 3), strides=up_g_stride_shape, kernel_initializer='he_normal', padding='same')(phi_g)
    
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), kernel_initializer='he_normal', padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)

    sigmoid_shape = K.int_shape(sigmoid_xg)
    up_psi_stride_shape = (X_shape[1] // sigmoid_shape[1], X_shape[2] // sigmoid_shape[2])

    upsample_psi = UpSampling2D(size=up_psi_stride_shape)(sigmoid_xg) 
    upsample_psi = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': X_shape[3]})(upsample_psi)                          
    
    y = multiply([upsample_psi, x])
    result = Conv2D(X_shape[3], (1, 1), kernel_initializer='he_normal', padding='same')(y)
    attention_block = BatchNormalization()(result)
    return attention_block


def att_unet(input_shape=(128, 128, 3), num_classes=1):

    input_size = input_shape[0]
    inputs = Input(shape=input_shape)
    down = inputs

    depth = 4
    features = 32
    downs = []

    # ENCODER
    for _ in range(depth):
        down = Conv2D(features, (3, 3), padding='same')(down)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        down = Conv2D(features, (3, 3), padding='same')(down)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        downs.append(down)
        down = MaxPooling2D((2, 2), strides=(2, 2))(down)
        features = features * 2

    # CENTER
    center = Conv2D(features, (3, 3), padding='same')(down)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(features, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)

    # DECODER
    up = center
    for i in reversed(range(depth)):
        features = features // 2
        gating = gatingsignal(up, features)
        att = attention_block(downs[i], gating, features)        
        up = UpSampling2D((2, 2), data_format="channels_last")(up)
        up = concatenate([up, att], axis=3)
        up = Conv2D(features, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv2D(features, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv2D(features, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up)
    model = Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=RMSprop(lr=0.001), loss=bce_dice_loss, metrics=[dice_loss])

    return input_size, model

