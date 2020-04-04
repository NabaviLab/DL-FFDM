"""
This file is for testing proformance of PNG and bitmap on same model
Model employed : VGG16
"""


from __future__ import absolute_import
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop, Adam
from keras.applications import VGG16, MobileNet
import keras
from keras import backend as K
import sys

from PIL import Image, ImageFile
from keras_preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""""""""""""""""""""""""""""""""""""""""some hyper parameters """""""""""""""""""""""""""""""""""""""""""""
num_classes = 2
EPOCHS = 20
DEBUGMODE = False
IMGshape = 224
LRP = 0.001
LR = 0.001
BATCH_SIZE = 128
MODELFLAG = True
"""""""""""""""""""""""""""""""""""""""""some hyper parameters """""""""""""""""""""""""""""""""""""""""""""

def create_base_network(inputShape, modelFlagt):
    '''Base network to be shared .
    '''
    input = Input(shape=inputShape)

    K.set_learning_phase(0)
    base_model = VGG16(weights='imagenet', include_top=True)
    K.set_learning_phase(1)
    for layer in base_model.layers[:16]:
        layer.trainable = False
        print(layer.name)
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    x = layer_dict['fc2'].output

    if modelFlagt:

        x = keras.layers.Dense(2048,activation='relu',kernel_regularizer=keras.regularizers.l2(0.0001))(x)
        x = keras.layers.BatchNormalization(name='bn_fc_01')(x)
        x = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(2,activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
    else:
        x = keras.layers.Dense(2, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=x)
    model.summary()

    return model


def recall_m(y_true, y_pred):
    pass

def precision_m(y_true, y_pred):
    pass

def f1_m(y_true, y_pred):
    pass

def pretrain():
    input_shape = [IMGshape, IMGshape, 3]

    print("start data loading...")




    trainPath = sys.argv[1]
    validationPath = sys.argv[2]
    testPath = sys.argv[3]

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train = train_datagen.flow_from_directory(trainPath, class_mode='categorical',
                                              batch_size=BATCH_SIZE,
                                              target_size=(IMGshape, IMGshape),
                                              subset='training', shuffle=True
                                              , seed=200)
    validation = validation_datagen.flow_from_directory(validationPath, class_mode='categorical',
                                                  batch_size=BATCH_SIZE,
                                                  target_size=(IMGshape, IMGshape)
                                                  , shuffle=True, seed=200)

    test = test_datagen.flow_from_directory(testPath, class_mode='categorical',
                                       batch_size=BATCH_SIZE,
                                       target_size=(IMGshape, IMGshape),
                                       shuffle=True,seed=200)

    print("done data loading...")
    print("VGG model with {}training, {} validation, {} testing, epoch = {}, lr = {}, batch size= {} , fileName = TLvggModelTest.py".format(len(train), len(validation), len(test), EPOCHS, LRP, BATCH_SIZE))
    base_network = create_base_network(input_shape, MODELFLAG)
    # train
    rms = RMSprop()
    adam = Adam(lr=LRP)
    base_network.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print('start training...')
    base_network.fit_generator(train,steps_per_epoch = len(train),
                            validation_data = validation,
                            validation_steps = len(validation),
                            epochs = 50,
                            shuffle=True)

    preformance = base_network.evaluate(test)

    print('test loss : ', preformance[0])
    print('test accuracy : ', preformance[1])

    base_network.save("pretrainVGG.h5")
    base_network.save_weights('pretrain_wVGG.h5')



if __name__ == "__main__":

    pretrain()


