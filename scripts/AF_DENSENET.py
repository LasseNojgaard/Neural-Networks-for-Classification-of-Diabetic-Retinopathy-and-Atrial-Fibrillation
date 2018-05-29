from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import keras
import matplotlib as mpl
mpl.use('Agg')


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Input
from keras.layers.normalization import BatchNormalization
import gc
from keras import backend as K
import numpy as np
from keras.models import Model
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
import scipy.io as spio
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_validate
import itertools
from sklearn.metrics import confusion_matrix
from keras.utils.vis_utils import plot_model

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

def create_model(X_train):
    ##################################################
    #The desired number of hidden layers are set here#
    ##################################################

    input_layer = Input(X_train.shape[1:], name="base_input")
    x1 = Dense(1024 , name='DENSE_1', activation="relu")(input_layer)
    x1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x1)
   # x1 = Dropout(0.1)(x1)

    x2 = Dense(1024 , name='DENSE_2')(x1)
    x2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x2)
    concentrated12 = Concatenate(name="ConX1X2")([x1,x2])
    #concentrated12 = Dropout(0.1)(concentrated12)

    x3 = Dense(1024, name='DENSE_3', activation="relu")(concentrated12)
    x3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x3)
    concentrated123 = Concatenate(name="ConX1X2X3")([concentrated12, x3])
    #concentrated123 = Dropout(0.1)(concentrated123)

    x4 = Dense(1024, name="Dense_4")(concentrated123)
    x4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x4)
    concentrated1234 = Concatenate(name="ConX1X2X3X4")([concentrated123, x4])
    #concentrated1234 = Dropout(0.1)(concentrated1234)

    x5 = Dense(1024, name='DENSE_5', activation="relu")(concentrated1234)
    x5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x5)
    concentrated12345 = Concatenate(name="ConX1X2X3X4X5")([concentrated1234, x5])
    #concentrated12345 = Dropout(0.1)(concentrated12345)

    x6 = Dense(1024, name='DENSE_6', activation="relu")(concentrated12345)
    x6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x6)
    concentrated123456 = Concatenate(name="ConX1X2X3X4X5X6")([concentrated12345, x6])
    #concentrated123456 = Dropout(0.1)(concentrated123456)

    x7 = Dense(1024, name='DENSE_7', activation="relu")(concentrated123456)
    x7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x7)
    concentrated1234567 = Concatenate(name="ConX1X2X3X4X5X6X7")([concentrated123456, x7])
    #concentrated1234567 = Dropout(0.1)(concentrated1234567)

    x8 = Dense(1024, name='DENSE_8', activation="relu")(concentrated1234567)
    x8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x8)
    concentrated12345678 = Concatenate(name="ConX1X2X3X4X5X6X7X8")([concentrated1234567, x8])
    #concentrated12345678 = Dropout(0.1)(concentrated12345678)

    x9 = Dense(1024, name='DENSE_9', activation="relu")(concentrated12345678)
    x9 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x9)
    concentrated123456789 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9")([concentrated12345678, x9])
    #concentrated123456789 = Dropout(0.1)(concentrated123456789)

    x10 = Dense(1024, name='DENSE_10', activation="relu")(concentrated123456789)
    x10 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x10)
    concentrated12345678910 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9X10")([concentrated123456789, x10])
    #concentrated12345678910 = Dropout(0.1)(concentrated12345678910)

    x11 = Dense(1024, name='DENSE_11', activation="relu")(concentrated12345678910)
    x11 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x11)
    concentrated1234567891011 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9X10X11")([concentrated12345678910, x11])
    #concentrated1234567891011 = Dropout(0.1)(concentrated1234567891011)

    x12 = Dense(1024, name='DENSE_12', activation="relu")(concentrated1234567891011)
    x12 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x12)
    concentrated123456789101112 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9X10X11X12")([concentrated1234567891011, x12])
    #concentrated123456789101112 = Dropout(0.1)(concentrated123456789101112)

    x13 = Dense(1024, name='DENSE_13', activation="relu")(concentrated123456789101112)
    x13 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x13)
    concentrated12345678910111213 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9X10X11X12X13")([concentrated123456789101112, x13])
    #concentrated12345678910111213 = Dropout(0.1)(concentrated12345678910111213)

    x14 = Dense(1024, name='DENSE_14', activation="relu")(concentrated12345678910111213)
    x14 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x14)
    concentrated1234567891011121314 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9X10X11X12X13X14")([concentrated12345678910111213, x14])
    #concentrated1234567891011121314 = Dropout(0.1)(concentrated1234567891011121314)

    x15 = Dense(1024, name='DENSE_15', activation="relu")(concentrated1234567891011121314)
    x15 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x15)
    concentrated123456789101112131415 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9X10X11X12X13X14X15")([concentrated1234567891011121314, x15])
    #concentrated123456789101112131415 = Dropout(0.1)(concentrated123456789101112131415)

    out = Dense(2, name="output")(concentrated123456789101112131415)
    out = Activation('softmax', name="SOFTMAX")(out)

    return Model(inputs=input_layer, outputs=out, name="base_model")


###############################################
#Loads the data from same folder as the script#
###############################################
matFull = spio.loadmat('FullData.mat')
matRed = spio.loadmat('DataReduced.mat')
mat49 = spio.loadmat('DataArray49.mat')
matLabel = spio.loadmat('LabelsLogi.mat')


dataFull = matFull['DataArray']
dataRed = matRed['DataArrayReduced']
data49 = mat49['DataArray49']
dataLabel = matLabel['DataClassVec']

dataFull = np.array(dataFull)
dataRed = np.array(dataRed)
data49 = np.array(data49)
dataLabel = np.array(dataLabel)

#print('Full: ', dataFull.shape)
#print('Label: ', dataLabel.shape)

dataSets = [0,1,2]
#dataSets = [2]

for datasets in dataSets:
    if (datasets==0):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(data49, dataLabel, test_size=0.1)
    if (datasets == 1):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(dataRed, dataLabel, test_size=0.1)
    if (datasets == 2):
        X_train, X_test, y_train, y_test = model_selection.train_test_split(dataFull, dataLabel, test_size=0.1)


    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)


    class1 = np.sum(y_train, axis=0)


    class_weight = {0 : 1.,
        1: (len(X_train)/class1),
        }


    batch_size = 32
    num_classes = 2
    epochs = 50
    k_folds = 10

    # Convert class vectors to binary class matrices.
    y_train_temp = y_train
    y_test_temp = y_test
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std

    filename = "DENSENET_15HLayer"

    if datasets==0:
        filename = filename +"_49"
    if datasets == 1:
        filename = filename + "_Red"
    if datasets == 2:
        filename = filename + "_Full"


    file = open('AF_' + filename + '.txt', 'w')

    # define 10-fold cross validation test harness
    #kf = KFold(n_splits=k_folds, shuffle=False, random_state=None)
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    num_fold = 1
    cvTrainACCscores = []
    cvTrainLOSSscores = []
    cvValACCscores = []
    cvValLOSSscores = []
    cvTestACCscores = []
    cvTestLOSSscores = []
    for train_index, val_index in kf.split(X_train, y_train_temp):
        #print("TRAIN:", train_index, "VAL:", val_index)
        X1_train, X1_val = X_train[train_index], X_train[val_index]
        y1_train, y1_val = y_train[train_index], y_train[val_index]

        model = create_model(X_train)

        opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        #opt = keras.optimizers.RMSprop(lr=0.00001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        history = model.fit(X1_train, y1_train, batch_size=batch_size, epochs=epochs, validation_data=(X1_val, y1_val), shuffle=True, class_weight=class_weight)



        # evaluate the model
        #scores = model.evaluate_generator(X_test, y_test, verbose=1)
        scores = model.evaluate(X_test, y_test, verbose=1)

        file.write(filename + format(num_fold) + '\n')
        file.write('training loss: ' + format(history.history['loss'][epochs - 1]) + '\n')
        file.write('training accuracy: ' + format(history.history['acc'][epochs - 1]) + '\n')
        file.write('validation loss: ' + format(history.history['val_loss'][epochs - 1]) + '\n')
        file.write('validation accuracy: ' + format(history.history['val_acc'][epochs - 1]) + '\n')
        file.write('Test loss: ' + format(scores[0]) + '\n')
        file.write('Test accuracy: ' + format(scores[1]) + '\n' + '\n')

        cvTrainLOSSscores.append(history.history['loss'][epochs - 1])
        cvTrainACCscores.append(history.history['acc'][epochs - 1])
        cvValLOSSscores.append(history.history['val_loss'][epochs - 1])
        cvValACCscores.append(history.history['val_acc'][epochs - 1])
        cvTestLOSSscores.append(scores[0])
        cvTestACCscores.append(scores[1])

        test_labels = []
        predictions_labels = []
        predictionsTemp = []

        predictionsTemp = model.predict(X_test)
        for ii in range(0, len(predictionsTemp)):
            predictions_labels = np.append(predictions_labels, np.argmax(predictionsTemp[ii]))

        cm = confusion_matrix(y_test_temp, predictions_labels)
        class_names = ['a', 'b']
        plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')
        #plt.savefig(filename + ':' + format(num_fold) + ' CM.png', bbox_inches='tight')
        plt.clf()
        num_fold += 1
        del history
        del model
        del scores
        gc.collect()
        K.clear_session()

    file.write('Average Results' + '\n')
    file.write('training loss: mean: ' + format(np.mean(cvTrainLOSSscores)) + ' std: ' + format(np.std(cvTrainLOSSscores)) + '\n')
    file.write('training accuracy: mean: ' + format(np.mean(cvTrainACCscores)) + ' std: ' + format(np.std(cvTrainACCscores)) + '\n')
    file.write('validation loss: mean: ' + format(np.mean(cvValLOSSscores)) + ' std: ' + format(np.std(cvValLOSSscores)) + '\n')
    file.write('validation accuracy: mean: ' + format(np.mean(cvValACCscores)) + ' std: ' + format(np.std(cvValACCscores)) + '\n')
    file.write('Test loss: mean: ' + format(np.mean(cvTestLOSSscores)) + ' std: ' + format(np.std(cvTestLOSSscores)) + '\n')
    file.write('Test accuracy: mean: ' + format(np.mean(cvTestACCscores)) + ' std: ' + format(np.std(cvTestACCscores)) + '\n' + '\n')

    file.close()



