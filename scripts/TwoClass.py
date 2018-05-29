from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201
from keras.applications.densenet import DenseNet121

from keras.preprocessing import image
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from skimage import data, exposure, img_as_float

"""Fairly basic set of tools for real-time data augmentation on image data.

Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


fileName = 'InceptionV3E25'
batch_size = 32
test_batch_size = 16

#train_data_size = 18688
#val_data_size = 5184
#test_data_size = 5184

train_data_size = 4032
val_data_size = 448
test_data_size = 448

#train_data_size = 64
#val_data_size = 32
#test_data_size = 32

num_classes = 2
epochs = 1
num_predictions = 20
sizeOfImage = 256



##############################################################################################################
#The data is loaded by using keras flow_from_directory, change the path to make it work on different computer#
##############################################################################################################

train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True
)

test_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True
)

train_generator = train_datagen.flow_from_directory(
        #'/media/lasse/DATA/RetinaBilleder/Train4000',
        '/home/lnoejgaard/data/Train4000',
        target_size=(sizeOfImage, sizeOfImage),
        batch_size=batch_size,
        #save_to_dir='dataPreview'
) #18.701

validation_generator = test_datagen.flow_from_directory(
        #'/media/lasse/DATA/RetinaBilleder/Val4000',
        '/home/lnoejgaard/data/Val4000',
        target_size=(sizeOfImage, sizeOfImage),
        batch_size=batch_size
)#5204

test_generator = test_datagen.flow_from_directory(
        #'/media/lasse/DATA/RetinaBilleder/Test4000',
        '/home/lnoejgaard/data/Test4000',
        target_size=(sizeOfImage, sizeOfImage),
        batch_size=test_batch_size
)#5204
#learningRates = [0.001, 0.01, 0.0001]
learningRates = [0.001, 0.0001]
#learningRates = [0.0001]

for learn in learningRates:
    file = open(fileName + ':'+format(learn)+'.txt', 'w')
    ###########################################
    #Change which model is going to be trained#
    ###########################################

    #base_model = Xception(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    #base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    #base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    #base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    #base_model = VGG19(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    #base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    #base_model = VGG16(weights='imagenet', include_top=False, input_shape=(sizeOfImage, sizeOfImage, 3))

    # add a global spatial average pooling layer
    x = base_model.output
    x = Flatten()(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)

    #this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # initiate RMSprop optimizer
    opt = keras.optimizers.Adam(lr=learn, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss='categorical_crossentropy',
                   optimizer=opt,
                   metrics=['accuracy'])

    #plot_model(model, to_file='model.png')
    #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=0, verbose=0, mode='auto')

    history = model.fit_generator(train_generator, steps_per_epoch=train_data_size/batch_size, epochs=epochs, validation_data=validation_generator, validation_steps=val_data_size/batch_size)

    scores = model.evaluate_generator(test_generator, steps=test_data_size / test_batch_size)

    file.write('Learning rate: ' + format(learn) + '\n')
    file.write('training loss: ' + format(history.history['loss'][epochs - 1]) + '\n')
    file.write('training accuracy: ' + format(history.history['acc'][epochs - 1]) + '\n')
    file.write('validation loss: ' + format(history.history['val_loss'][epochs - 1]) + '\n')
    file.write('validation accuracy: ' + format(history.history['val_acc'][epochs - 1]) + '\n')
    file.write('Test loss: ' + format(scores[0]) + '\n')
    file.write('Test accuracy: ' + format(scores[1]) + '\n' + '\n')
    file.close()

    model.save(fileName + format(learn) + '.h5')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(fileName + ':' + format(learn) + ' Accuracy.png', bbox_inches='tight')
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(fileName + ':' + format(learn) + ' Loss.png', bbox_inches='tight')
    plt.clf()

    test_labels = []
    predictions_labels = []

    for i in range(0, int(test_data_size / test_batch_size)):
        print(format(i) + " out off: " + format(test_data_size / test_batch_size))
        test_imgs, test_labelsTemp = next(test_generator)

        for ii in range(0, test_batch_size):
            test_labels = np.append(test_labels, np.argmax(test_labelsTemp[ii]))

        predictionsTemp = model.predict_on_batch(test_imgs)
        for ii in range(0, test_batch_size):
            predictions_labels = np.append(predictions_labels, np.argmax(predictionsTemp[ii]))

    cm = confusion_matrix(test_labels, predictions_labels)
    class_names = ['a', 'b']

    plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig(fileName + ':' + format(learn) + '.png', bbox_inches='tight')
    plt.clf()


