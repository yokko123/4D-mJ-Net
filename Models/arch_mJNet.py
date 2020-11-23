import constants
from Utils import general_utils, spatial_pyramid

from tensorflow.keras import layers, models, regularizers, initializers
from tensorflow.keras.constraints import max_norm
# from keras.layers.core import Dropout
# from keras.layers.convolutional import Conv3D, Conv3DTranspose
from tensorflow.keras.layers import Conv3D, Conv3DTranspose, Dropout, Concatenate
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG16


################################################################################
# mJ-Net model
def mJNet(params, to_categ, drop=False, longJ=False, v2=False):
    # from (30,M,N) to (1,M,N)

    size_two = (2,2,1)  # (1,2,2)
    kernel_size = (3,3,1)
    activ_func = 'relu'
    l1_l2_reg = None
    channels = [16,32,16,32,16,32,16,32,64,64,128,128,256,-1,-1,-1,-1,128,128,64,64,32,16]
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)
    kernel_init = "glorot_uniform"  # Xavier uniform initializer.
    kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)

    if v2:  # version 2
        # size_two = (2,2,1)
        activ_func = None
        l1_l2_reg = regularizers.l1_l2(l1=1e-6, l2=1e-5)
        # Hu initializer
        kernel_init = initializers.VarianceScaling(scale=(9/5), mode='fan_in', distribution='normal', seed=None)
        kernel_constraint, bias_constraint = max_norm(2.), None

        # channels = [16,32,32,64,64,128,128,32,64,128,256,512,1024,512,1024,512,1024,-1,512,256,-1,128,64]
        channels = [16,32,32,64,64,128,128,16,32,32,64,64,128,128,128,256,128,-1,128,64,-1,64,32]
        # channels = [int(ch/2) for ch in channels] # implemented due to memory issues

        # input_shape = (None,constants.getM(),constants.getN(),1)
        # TODO: input_shape = (constants.NUMBER_OF_IMAGE_PER_SECTION,None,None,1)

    input_x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(input_x)  # (None, 30, M, N, 1)

    if longJ:
        conv_01 = Conv3D(channels[0], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_x)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01)  # (None, 30, M, N, 16)
        conv_01 = Conv3D(channels[1], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_01)
        if v2: conv_01 = layers.LeakyReLU(alpha=0.33)(conv_01)
        conv_01 = layers.BatchNormalization()(conv_01)
        general_utils.print_int_shape(conv_01)  # (None, 30, M, N, 32)

        # pool_drop_01 = layers.MaxPooling3D((params["max_pool"]["long.1"],1,1))(conv_01)
        pool_drop_01 = layers.MaxPooling3D((1,1,params["max_pool"]["long.1"]))(conv_01)
        general_utils.print_int_shape(pool_drop_01) # (None, 15, M, N, 32)
        conv_02 = Conv3D(channels[2], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_01)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02)  # (None, 15, M, N, 32)
        conv_02 = Conv3D(channels[3], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_02)
        if v2: conv_02 = layers.LeakyReLU(alpha=0.33)(conv_02)
        conv_02 = layers.BatchNormalization()(conv_02)
        general_utils.print_int_shape(conv_02)  # (None, 15, M, N, 64)

        # pool_drop_02 = layers.MaxPooling3D((params["max_pool"]["long.2"],1,1))(conv_02)
        pool_drop_02 = layers.MaxPooling3D((1,1,params["max_pool"]["long.2"]))(conv_02)
        general_utils.print_int_shape(pool_drop_02)  # (None, 5, M, N, 64)
        conv_03 = Conv3D(channels[4], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_02)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03)  # (None, 5, M, N, 64)
        conv_03 = Conv3D(channels[5], kernel_size=kernel_size, activation=activ_func, padding='same',
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_03)
        if v2: conv_03 = layers.LeakyReLU(alpha=0.33)(conv_03)
        conv_03 = layers.BatchNormalization()(conv_03)
        general_utils.print_int_shape(conv_03)  # (None, 5, M, N, 128)

        # pool_drop_1 = layers.MaxPooling3D((params["max_pool"]["long.3"],1,1))(conv_03)
        pool_drop_1 = layers.MaxPooling3D((1,1,params["max_pool"]["long.3"]))(conv_03)
        general_utils.print_int_shape(pool_drop_1)  # (None, 1, M, N, 128)
        if drop: pool_drop_1 = Dropout(params["dropout"]["long.1"])(pool_drop_1)
    else:
        # conv_1 = Conv3D(channels[6], kernel_size=(constants.NUMBER_OF_IMAGE_PER_SECTION,3,3), activation=activ_func, padding='same', kernel_regularizer=l1_l2_reg)(input_x)
        conv_1 = Conv3D(channels[6], kernel_size=(3,3,constants.NUMBER_OF_IMAGE_PER_SECTION), activation=activ_func,
                        padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                        kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(input_x)
        if v2: conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
        conv_1 = layers.BatchNormalization()(conv_1)
        general_utils.print_int_shape(conv_1)  # (None, 30, M, N, 128)
        # TODO: make this dynamic based on the original flag
        # pool_drop_1 = layers.AveragePooling3D((constants.NUMBER_OF_IMAGE_PER_SECTION,1,1))(conv_1)
        pool_drop_1 = layers.AveragePooling3D((1,1,constants.NUMBER_OF_IMAGE_PER_SECTION))(conv_1)
        # pool_drop_1 = spatial_pyramid.SPP3D([1,2,4], input_shape=(channels[6],None,None,None))(conv_1)
        general_utils.print_int_shape(pool_drop_1)  # (None, 1, M, N, 128)
        if drop: pool_drop_1 = Dropout(params["dropout"]["1"])(pool_drop_1)

    # from (1,M,N) to (1,M/2,N/2)
    conv_2 = Conv3D(channels[7], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_1)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, 1, M, N, 32)
    conv_2 = Conv3D(channels[8], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_2)
    if v2: conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, 1, M, N, 64)
    pool_drop_2 = layers.MaxPooling3D(size_two)(conv_2)
    general_utils.print_int_shape(pool_drop_2)  # (None, 1, M/2, N/2, 64)
    if drop: pool_drop_2 = Dropout(params["dropout"]["2"])(pool_drop_2)

    # from (1,M/2,N/2) to (1,M/4,N/4)
    conv_3 = Conv3D(channels[9], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_2)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)  # (None, 1, M/2, N/2, 128)
    conv_3 = Conv3D(channels[10], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_3)
    if v2: conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)  # (None, 1, M/2, N/2, 256)
    pool_drop_3 = layers.MaxPooling3D(size_two)(conv_3)
    general_utils.print_int_shape(pool_drop_3)  # (None, 1, M/4, N/4, 256)
    if drop: pool_drop_3 = Dropout(params["dropout"]["3"])(pool_drop_3)

    # from (1,M/4,N/4) to (1,M/8,N/8)
    conv_4 = Conv3D(channels[11], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_3)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, 1, M/4, N/4, 512)
    conv_4 = Conv3D(channels[12], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4)
    if v2: conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, 1, M/4, N/4, 1024)

    if v2:
        pool_drop_3_1 = layers.MaxPooling3D(size_two)(conv_4)
        general_utils.print_int_shape(pool_drop_3_1)  # (None, 1, M/8, N/8, 1024)
        if drop: pool_drop_3_1 = Dropout(params["dropout"]["3.1"])(pool_drop_3_1)
        conv_4_1 = Conv3D(channels[13], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_3_1)
        conv_4_1 = layers.LeakyReLU(alpha=0.33)(conv_4_1)
        conv_4_1 = layers.BatchNormalization()(conv_4_1)
        general_utils.print_int_shape(conv_4_1)  # (None, 1, M/8, N/8, 512)
        if drop: conv_4_1 = Dropout(params["dropout"]["3.2"])(conv_4_1)
        conv_5_1 = Conv3D(channels[14], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4_1)
        conv_5_1 = layers.LeakyReLU(alpha=0.33)(conv_5_1)
        conv_5_1 = layers.BatchNormalization()(conv_5_1)
        general_utils.print_int_shape(conv_5_1)  # (None, 1, M/8, N/8, 1024)
        add_1 = layers.add([pool_drop_3_1, conv_5_1])
        general_utils.print_int_shape(add_1)  # (None, 1, M/8, N/8, 1024)
        up_01 = layers.UpSampling3D(size=size_two)(add_1)
        general_utils.print_int_shape(up_01)  # (None, 1, M/4, N/4, 1024)

        conc_1 = layers.concatenate([up_01, conv_4], axis=-1)
        general_utils.print_int_shape(conc_1)  # (None, 1, M/4, N/4, 1024)
        if drop: conc_1 = Dropout(params["dropout"]["3.3"])(conc_1)
        conv_6_1 = Conv3D(channels[15], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conc_1)
        conv_6_1 = layers.LeakyReLU(alpha=0.33)(conv_6_1)
        conv_6_1 = layers.BatchNormalization()(conv_6_1)
        general_utils.print_int_shape(conv_6_1)  # (None, 1, M/4, N/4, 512)
        if drop: conv_6_1 = Dropout(params["dropout"]["3.4"])(conv_6_1)
        conv_7_1 = Conv3D(channels[16], (3,3,3), activation=activ_func, padding='same',
                          kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_6_1)
        conv_7_1 = layers.LeakyReLU(alpha=0.33)(conv_7_1)
        conv_7_1 = layers.BatchNormalization()(conv_7_1)
        general_utils.print_int_shape(conv_7_1)  # (None, 1, M/4, N/4, 1024)
        add_2 = layers.add([conv_4, conv_7_1])
        general_utils.print_int_shape(add_2)  # (None, 1, M/4, N/4, 1024)
        up_02 = layers.UpSampling3D(size=size_two)(add_2)
        general_utils.print_int_shape(up_02)  # (None, 1, M/2, N/2, 1024)

        addconv_3 = layers.concatenate([conv_3, conv_3])
        while K.int_shape(up_02)[-1] != K.int_shape(addconv_3)[-1]:
            addconv_3 = layers.concatenate([addconv_3, addconv_3])
        up_1 = layers.concatenate([up_02, addconv_3])
    else:
        # first UP-convolutional layer: from (1,M/4,N/4) to (2M/2,N/2)
        up_1 = layers.concatenate([
            Conv3DTranspose(channels[17], kernel_size=size_two, strides=size_two, activation=activ_func,
                            padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_4),
            conv_3], axis=3)

    general_utils.print_int_shape(up_1)  # (None, 1, M/2, N/2, 1024)
    conv_5 = Conv3D(channels[18], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_1)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5)  # (None, 1, M/2, N/2, 512)
    conv_5 = Conv3D(channels[19], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_5)
    if v2: conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5)  # (None, 1, M/2, N/2, 256)

    if v2:
        addconv_5 = layers.concatenate([conv_5, conv_5])
        while K.int_shape(addconv_5)[-1] != K.int_shape(up_1)[-1]:
            addconv_5 = layers.concatenate([addconv_5, addconv_5])
        add_3 = layers.add([up_1, addconv_5])
        general_utils.print_int_shape(add_3)  # (None, 1, M/2, N/4, 1024)
        up_03 = layers.UpSampling3D(size=size_two)(add_3)
        general_utils.print_int_shape(up_03)  # (None, 1, M, N, 1024)

        addconv_2 = layers.concatenate([conv_2, conv_2])
        while K.int_shape(addconv_2)[-1] != K.int_shape(up_03)[-1]:
            addconv_2 = layers.concatenate([addconv_2, addconv_2])
        up_2 = layers.concatenate([up_03, addconv_2])
    else:
        pool_drop_4 = layers.MaxPooling3D((1,1,2))(conv_5)
        general_utils.print_int_shape(pool_drop_4)  # (None, 1, M/2, N/2, 512)
        if drop: pool_drop_4 = Dropout(params["dropout"]["4"])(pool_drop_4)
        # second UP-convolutional layer: from (2,M/2,N/2,2) to (2,M,N)
        up_2 = layers.concatenate([
            Conv3DTranspose(channels[20], kernel_size=size_two, strides=size_two, activation=activ_func,
                            padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                            kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_4),
            conv_2], axis=3)

    general_utils.print_int_shape(up_2)  # (None, X, M, N, 1024)
    conv_6 = Conv3D(channels[21], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_2)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    conv_6 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(conv_6)  # (None, X, M, N, 128)
    conv_6 = Conv3D(channels[22], kernel_size=kernel_size, activation=activ_func, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv_6)
    if v2: conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    pool_drop_5 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(pool_drop_5)  # (None, X, M, N, 64)

    if not v2:
        # from (2,M,N)  to (1,M,N)
        pool_drop_5 = layers.MaxPooling3D((1,1,2))(pool_drop_5)
        general_utils.print_int_shape(pool_drop_5)  # (None, 1, M, N, 16)
        if drop: pool_drop_5 = Dropout(params["dropout"]["5"])(pool_drop_5)

    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(),constants.getN())

    # set the softmax activation function if the flag is set
    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(),constants.getN(),n_chann)

    # last convolutional layer; plus reshape from (1,M,N) to (M,N)
    conv_7 = Conv3D(n_chann, (1,1,1), activation=act_name, padding='same',
                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(pool_drop_5)
    general_utils.print_int_shape(conv_7)  # (None, 1, M, N, 1)
    y = layers.Reshape(shape_output)(conv_7)
    general_utils.print_int_shape(y)  # (None, M, N)
    model = models.Model(inputs=input_x, outputs=y)

    return model


################################################################################
# mJ-Net model version 3D ?
def mJNet_3D(params, to_categ):
    l1_l2_reg = None
    input_shape = (constants.getM(), constants.getN(), constants.NUMBER_OF_IMAGE_PER_SECTION, 1)
    kernel_init = "glorot_uniform"  # Xavier uniform initializer.

    # # Create base model
    # base_model = NASNetLarge(
    #     include_top=False,
    #     weights='imagenet'
    # )
    # # Freeze base model
    # base_model.trainable = False

    x = layers.Input(shape=input_shape, sparse=False)
    general_utils.print_int_shape(x)  # (None, M, N, 3, 1)
    # transfer_x = base_model(x, training=False)
    # general_utils.print_int_shape(transfer_x) # (None, 16, 16, 4032)
    # print([layers.name for layer in base_model.layers])


    conv_1 = layers.Conv2D(16, kernel_size=(3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(x)
    conv_1 = layers.LeakyReLU(alpha=0.33)(conv_1)
    conv_1 = layers.BatchNormalization()(conv_1)
    general_utils.print_int_shape(conv_1)  # (None, M, N, 3, 16)
    pool_drop_1 = layers.AveragePooling2D((2,2), padding='same')(conv_1)
    general_utils.print_int_shape(pool_drop_1)  # (None, M/2, N/2, 3, 16)

    conv_2 = layers.Conv2D(16, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_1)
    conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, M/2, N/2, 3, 16)
    conv_2 = layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_2)
    conv_2 = layers.LeakyReLU(alpha=0.33)(conv_2)
    conv_2 = layers.BatchNormalization()(conv_2)
    general_utils.print_int_shape(conv_2)  # (None, M/2, N/2, 3, 32)
    pool_drop_2 = layers.MaxPooling2D((2,2))(conv_2)
    general_utils.print_int_shape(pool_drop_2)  # (None, M/4, N/4, 3, 32)

    conv_3 = layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_2)
    conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)  # (None, M/4, N/4, 3, 32)
    conv_3 = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_3)
    conv_3 = layers.LeakyReLU(alpha=0.33)(conv_3)
    conv_3 = layers.BatchNormalization()(conv_3)
    general_utils.print_int_shape(conv_3)  # (None, M/4, N/4, 3, 64)
    pool_drop_3 = layers.MaxPooling2D((2,2))(conv_3)
    general_utils.print_int_shape(pool_drop_3)  # (None, M/8, N/8, 3, 64)

    conv_4 = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_3)
    conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, M/8, N/8, 3, 64)
    conv_4 = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_4)
    conv_4 = layers.LeakyReLU(alpha=0.33)(conv_4)
    conv_4 = layers.BatchNormalization()(conv_4)
    general_utils.print_int_shape(conv_4)  # (None, M/8, N/8, 3, 128)

    up_1 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv_4), conv_3], axis=-1)
    general_utils.print_int_shape(up_1)  # (None, M/4, N/4, 3, 128)
    conv_5 = layers.Conv2D(128, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(up_1)
    conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5)  # (None, M/4, N/4, 3, 128)
    conv_5 = layers.Conv2D(64, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_5)
    conv_5 = layers.LeakyReLU(alpha=0.33)(conv_5)
    conv_5 = layers.BatchNormalization()(conv_5)
    general_utils.print_int_shape(conv_5)  # (None, M/4, N/4, 3, 64)

    up_2 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(conv_5), conv_2], axis=-1)
    general_utils.print_int_shape(up_2)  # (None, M/2, N/2, 3, 32)
    conv_6 = layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(up_2)
    conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    conv_6 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(conv_6)  # (None, M/2, N/2, 3, 32)
    conv_6 = layers.Conv2D(16, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_6)
    conv_6 = layers.LeakyReLU(alpha=0.33)(conv_6)
    pool_drop_5 = layers.BatchNormalization()(conv_6)
    general_utils.print_int_shape(pool_drop_5)  # (None, M/2, N/2, 3, 16)

    up_3 = layers.concatenate([layers.UpSampling2D(size=(2, 2))(pool_drop_5), conv_1], axis=-1)
    general_utils.print_int_shape(up_2)  # (None, M, N, 3, 32)
    conv_7 = layers.Conv2D(32, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(up_3)
    conv_7 = layers.LeakyReLU(alpha=0.33)(conv_7)
    conv_7 = layers.BatchNormalization()(conv_7)
    general_utils.print_int_shape(conv_7)  # (None, M, N, 3, 16)
    conv_8 = layers.Conv2D(16, (3,3), padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(conv_7)
    conv_7 = layers.LeakyReLU(alpha=0.33)(conv_7)
    pool_drop_6 = layers.BatchNormalization()(conv_8)
    general_utils.print_int_shape(pool_drop_5)  # (None, M, N, 3, 16)

    conv_9 = layers.Conv2D(1, (1,1), activation="sigmoid", padding='same', kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init)(pool_drop_6)
    general_utils.print_int_shape(conv_9)  # (None, M, N, 1)
    y = layers.Reshape((constants.getM(),constants.getN()))(conv_9)
    general_utils.print_int_shape(y)  # (None, M, N)

    model = models.Model(inputs=x, outputs=y)
    return model


################################################################################
# Class that define a PM object
class PM_obj(object):

    def __init__(self, name, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint):
        self.name = ("_" + name)
        self.input_shape = (constants.getM(), constants.getN(), 3)

        # Create base model
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        self.base_model._name += self.name
        for layer in self.base_model.layers: layer._name += self.name
        # Freeze base model
        self.base_model.trainable = False

        self.input = self.base_model.input

        # Creating dictionary that maps layer names to the layers
        self.layer_dict = dict([(layer.name, layer) for layer in self.base_model.layers])

        # Conv layers after the VGG16
        self.conv_1 = layers.Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.base_model.output)
        self.conv_2 = layers.Conv2D(128, kernel_size=(3, 3), padding='same',activation=activ_func,
                                    kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(self.conv_1)
        self.conv_2 = layers.BatchNormalization()(self.conv_2)

################################################################################
# mJ-Net model version for the parametric maps as input
def mJNet_PM(params, to_categ):

    activ_func = 'relu'
    l1_l2_reg = regularizers.l1_l2(l1=1e-6, l2=1e-5)
    kernel_init = "glorot_uniform"  # Xavier uniform initializer.
    kernel_constraint, bias_constraint = max_norm(2.), max_norm(2.)

    cbf = PM_obj("cbf", activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    cbv = PM_obj("cbv", activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    ttp = PM_obj("ttp", activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    tmax = PM_obj("tmax", activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)

    PMS = [cbf, cbv, ttp, tmax]

    layersAfterTransferLearning, inputs, block5_conv3, block4_conv3, block3_conv3, block2_conv2, block1_conv2 = [], [], [], [], [], [], []

    for pm in PMS:
        layersAfterTransferLearning.append(pm.conv_2)
        inputs.append(pm.input)
        block5_conv3.append(pm.layer_dict["block5_conv3" + pm.name].output)
        block4_conv3.append(pm.layer_dict["block4_conv3" + pm.name].output)
        block3_conv3.append(pm.layer_dict["block3_conv3" + pm.name].output)
        block2_conv2.append(pm.layer_dict["block2_conv2" + pm.name].output)
        block1_conv2.append(pm.layer_dict["block1_conv2" + pm.name].output)

    conc_layer = Concatenate(-1)(layersAfterTransferLearning)
    general_utils.print_int_shape(conc_layer)

    transp_1 = layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding='same',activation=activ_func,
                                      kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                      kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conc_layer)
    general_utils.print_int_shape(transp_1)

    block5_conv3_conc = Concatenate(-1)(block5_conv3)
    up_1 = Concatenate(-1)([transp_1,block5_conv3_conc])
    general_utils.print_int_shape(up_1)

    # going up with the layers
    up_2 = upLayer(up_1, 128, block4_conv3, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    up_3 = upLayer(up_2, 64, block3_conv3, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    up_4 = upLayer(up_3, 32, block2_conv2, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)
    up_5 = upLayer(up_4, 16, block1_conv2, len(PMS), activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint)

    final_conv_1 = layers.Conv2D(16, kernel_size=(3, 3), padding='same',activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(up_5)
    final_conv_1 = layers.BatchNormalization()(final_conv_1)
    general_utils.print_int_shape(final_conv_1)
    final_conv_2 = layers.Conv2D(16, kernel_size=(3, 3), padding='same',activation=activ_func,
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_1)
    final_conv_2 = layers.BatchNormalization()(final_conv_2)
    general_utils.print_int_shape(final_conv_2)

    act_name = "sigmoid"
    n_chann = 1
    shape_output = (constants.getM(), constants.getN())

    # set the softmax activation function if the flag is set
    if to_categ:
        act_name = "softmax"
        n_chann = len(constants.LABELS)
        shape_output = (constants.getM(), constants.getN(), n_chann)

    final_conv_3 = layers.Conv2D(n_chann, kernel_size=(1, 1), activation=act_name, padding='same',
                                 kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                 kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(final_conv_2)
    general_utils.print_int_shape(final_conv_3)
    y = layers.Reshape(shape_output)(final_conv_3)
    general_utils.print_int_shape(y)

    model = models.Model(inputs=inputs, outputs=[y])
    return model


################################################################################
# Helpful function to define up-layers based on the previous layer
def upLayer(prev_up, filters, block, howmanypms, activ_func, l1_l2_reg, kernel_init, kernel_constraint, bias_constraint):
    conv = layers.Conv2D(filters * howmanypms, kernel_size=(3, 3), padding='same',activation=activ_func,
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(prev_up)
    conv = layers.Conv2D(filters * howmanypms, kernel_size=(3, 3), padding='same',activation=activ_func,
                         kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                         kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv)
    transp = layers.Conv2DTranspose(filters * howmanypms, kernel_size=(2, 2), strides=(2, 2), padding='same',
                                    activation=activ_func,kernel_regularizer=l1_l2_reg, kernel_initializer=kernel_init,
                                    kernel_constraint=kernel_constraint, bias_constraint=bias_constraint)(conv)

    block_conc = Concatenate(-1)(block)
    up = Concatenate(-1)([transp, block_conc])

    return up
