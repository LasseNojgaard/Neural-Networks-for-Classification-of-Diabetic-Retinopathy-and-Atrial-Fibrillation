from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import keras
import matplotlib as mpl
mpl.use('Agg')


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Input, Add
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
from keras.utils.vis_utils import plot_model
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support,cohen_kappa_score,classification_report

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


def create_model(X_train, hidden_layer=32, dropout=0):
    input_layer = Input(X_train.shape[1:], name="base_input")
    x1 = Dense(hidden_layer, name='DENSE_1', activation="relu")(input_layer)
    x1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x1)
    x1 = Dropout(dropout)(x1)

    x2 = Dense(hidden_layer, name='DENSE_2')(x1)
    x2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x2)
    added12 = Add(name="ADDX1X2")([x1, x2])
    added12 = Dropout(dropout)(added12)

    x3 = Dense(hidden_layer, name='DENSE_3', activation="relu")(added12)
    x3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x3)
    added23 = Add(name="ADDX2X3")([x2, x3])
    added23 = Dropout(dropout)(added23)

    x4 = Dense(hidden_layer, name="Dense_4")(added23)
    x4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x4)
    added34 = Add(name="ADDX3X4")([x3, x4])
    added34 = Dropout(dropout)(added34)

    x5 = Dense(hidden_layer, name='DENSE_5', activation="relu")(added34)
    x5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x5)
    added45 = Add(name="ADDX4X5")([x4, x5])
    added45 = Dropout(dropout)(added45)

    x6 = Dense(hidden_layer, name='DENSE_6', activation="relu")(added45)
    x6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x6)
    added56 = Add(name="ADDX5X6")([x5, x6])
    added56 = Dropout(dropout)(added56)

    out = Dense(2, name="output")(added56)
    out = Activation('softmax', name="SOFTMAX")(out)
    return Model(inputs=input_layer, outputs=out, name="base_model")




###############################################
#Loads the data from same folder as the script#
###############################################
matFull = spio.loadmat('dataFullTrainOver.mat')
matRed = spio.loadmat('dataRedTrainOver.mat')
mat49 = spio.loadmat('data49TrainOver.mat')
matLabel = spio.loadmat('labelTrainOver.mat')

matFullTest = spio.loadmat('dataFullTest.mat')
matRedTest = spio.loadmat('dataRedTest.mat')
mat49Test = spio.loadmat('data49Test.mat')
matLabelTest = spio.loadmat('labelTest.mat')


dataFull = matFull['arr']
dataRed = matRed['arr']
data49 = mat49['arr']
dataLabel = matLabel['arr']

dataFullTest = matFullTest['arr']
dataRedTest = matRedTest['arr']
data49Test = mat49Test['arr']
dataLabelTest = matLabelTest['arr']

dataFull = np.array(dataFull)
dataRed = np.array(dataRed)
data49 = np.array(data49)
dataLabel = np.array(dataLabel)

dataFullTest = np.array(dataFullTest)
dataRedTest = np.array(dataRedTest)
data49Test = np.array(data49Test)
dataLabelTest = np.array(dataLabelTest)



#print('Full: ', dataFull.shape)
#print('Label: ', dataLabel.shape)

dataSets = [0,1,2]
#dataSets = [1]
for datasets in dataSets:
    if (datasets==0):
        X_train = data49
        X_test = data49Test
        y_train = dataLabel
        y_test = dataLabelTest
    if (datasets == 1):
        X_train = dataRed
        X_test = dataRedTest
        y_train = dataLabel
        y_test = dataLabelTest
    if (datasets == 2):
        X_train = dataFull
        X_test = dataFullTest
        y_train = dataLabel
        y_test = dataLabelTest

    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)


    batch_size = 64
    num_classes = 2
    epochs = 100
    k_folds = 5

    # Convert class vectors to binary class matrices.
    y_train_temp = y_train
    y_test_temp = y_test
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std

    #filename = "ResNet_GS_DropoutEpochs"
    filename = "ResNet_GS_Hidden"

    if datasets==0:
        filename = filename +"_49"
    if datasets == 1:
        filename = filename + "_Red"
    if datasets == 2:
        filename = filename + "_Full"

    # define 10-fold cross validation test harness
    #kf = KFold(n_splits=k_folds, shuffle=False, random_state=None)
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    num_fold = 1

    ####################################################
    #The desired hidden parameters to tune are set here#
    ####################################################
    learning_rateList = [0.0001, 0.00001, 0.000001]
    hidden_layerList = [128, 512, 1024, 1536, 2048]
    learning_decayList = [0, 0.00001, 0.000001]
    dropoutList = [0, 0.1]
    optimizer = ['adam', 'RMS', 'SGD']    
    epochList = [50]

    numberOfComb = len(learning_rateList) * len(learning_decayList) * len(hidden_layerList) * len(dropoutList) * len(optimizer) * len(epochList)
    cvNumber = 1

    bestSensSpeci = 0
    bestresultList = [None] * 9

    file = open('AF_' + filename + '.txt', 'w')
    for learning_rate in learning_rateList:
        for learning_decay in learning_decayList:
            for hidden_layer in hidden_layerList:
                for dropout in dropoutList:
                    for opti in optimizer:
                        for epochs in epochList:
                            cvTrainACCscores = []
                            cvTrainLOSSscores = []
                            cvValACCscores = []
                            cvValLOSSscores = []
                            cvTestACCscores = []
                            cvTestLOSSscores = []
                            cvSensSpeci = []
                            print('Mulighed:' + format(cvNumber) + ' ud af:' + format(numberOfComb))
                            cvNumber += 1
                            for train_index, val_index in kf.split(X_train, y_train_temp):
                                # print("TRAIN:", train_index, "VAL:", val_index)
                                X1_train, X1_val = X_train[train_index], X_train[val_index]
                                y1_train, y1_val = y_train[train_index], y_train[val_index]

                                model = create_model(X_train, hidden_layer, dropout)
                                if opti == 'adam':
                                    opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None,
                                                                decay=learning_decay, amsgrad=False)
                                if opti == 'RMS':
                                    opt = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=learning_decay)
                                if opti == 'SGD':
                                    opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=learning_decay, nesterov=True)

                                model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
                                history = model.fit(X1_train, y1_train, batch_size=batch_size, epochs=epochs,
                                                    validation_data=(X1_val, y1_val), shuffle=True, verbose=0)
                                scores = model.evaluate(X_test, y_test, verbose=0)

                                # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
                                cvTrainLOSSscores.append(history.history['loss'][epochs - 1])
                                cvTrainACCscores.append(history.history['acc'][epochs - 1])
                                cvValLOSSscores.append(history.history['val_loss'][epochs - 1])
                                cvValACCscores.append(history.history['val_acc'][epochs - 1])
                                cvTestLOSSscores.append(scores[0])
                                cvTestACCscores.append(scores[1])
                                predictions_labels = []
                                predictionsTemp = []
                                predictionsTemp = model.predict(X_test, batch_size=16)
                                for ii in range(0, len(predictionsTemp)):
                                    predictions_labels = np.append(predictions_labels, np.argmax(predictionsTemp[ii]))

                                #cm = confusion_matrix(y_test_temp, predictions_labels)
                                #class_names = ['No_Redo', 'Redo']
                                #plot_confusion_matrix(cm, classes=class_names,
                                #                      title='Confusion matrix, without normalization')
                                #plt.savefig(filename + "_" + format(dropout) +"_" + format(epochs) + '_CM' + '.png')
                                #plt.clf()


                                target_names = ['class 0', 'class 1']
                                report = precision_recall_fscore_support(y_test_temp, predictions_labels)
                                cvSensSpeci.append(report[1][0] + report[1][1])
                                del model
                                del history
                                del scores
                                gc.collect()
                                K.clear_session()

                            file.write('learning_rate:' + format(learning_rate) + ', ' + 'learning_decay:' + format(learning_decay) + ', ' + 'hidden_layer:' + format(hidden_layer) + ', ' + 'dropout:' + format(dropout) + ', ' + 'opti:' + format(opti))
                            file.write(', training accuracy:' + format(np.mean(cvTrainACCscores)) + '(' + format(
                                np.std(cvTrainACCscores)) + ')')
                            file.write(', validation accuracy:' + format(np.mean(cvValACCscores)) + '(' + format(
                                np.std(cvValACCscores)) + ')')
                            file.write(', Test accuracy:' + format(np.mean(cvTestACCscores)) + '(' + format(
                                np.std(cvTestACCscores)) + ')')
                            file.write(', SensSpeci:' + format(np.mean(cvSensSpeci)) + '(' + format(
                                np.std(cvSensSpeci)) + ')' + '\n')
                            if (np.mean(cvSensSpeci) > bestSensSpeci):
                                bestSensSpeci = np.mean(cvSensSpeci)
                                bestresultList[0] = learning_rate
                                bestresultList[1] = learning_decay
                                bestresultList[2] = hidden_layer
                                bestresultList[3] = dropout
                                bestresultList[4] = optimizer
                                bestresultList[5] = np.mean(cvTrainACCscores)
                                bestresultList[6] = np.mean(cvValACCscores)
                                bestresultList[7] = np.mean(cvTestACCscores)
                                bestresultList[8] = np.mean(cvSensSpeci)

    file.write('\n')
    file.write('Best Results:' + '\n')
    file.write('learning_rate:' + format(bestresultList[0]) + ', ' + 'learning_decay:' + format(bestresultList[1]) + ', ' + 'hidden_layer:' + format(bestresultList[2]) + ', ' + 'dropout:' + format(bestresultList[3]) + ', ' + 'opti:' + format(bestresultList[4]))
    file.write(', training accuracy:' + format(bestresultList[5]))
    file.write(', validation accuracy:' + format(bestresultList[6]))
    file.write(', Test accuracy:' + format(bestresultList[7]))
    file.write(', SensSpeci accuracy:' + format(bestresultList[8]))
    file.close()