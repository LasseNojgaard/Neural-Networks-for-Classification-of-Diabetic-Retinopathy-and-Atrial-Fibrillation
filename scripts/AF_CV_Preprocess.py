from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
import numpy as np
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


def create_model():
	# create model
    model = Sequential()
    model.add(Dense(1024, input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model



batch_size = 32
num_classes = 2
epochs = 50
#num_predictions = 20
k_folds = 5



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


##########################################################################
#These three variables choses which dataset is used for training and test#
##########################################################################
X_train, X_test, y_train, y_test = model_selection.train_test_split(data49, dataLabel, test_size=0.1)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(dataRed, dataLabel, test_size=0.1)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(dataFull, dataLabel, test_size=0.1)

mean = np.mean(X_train, axis = 0)
std = np.std(X_train, axis = 0)


class1 = np.sum(y_train, axis=0)
#print(y_train)
#print(class1)
#print(len(X_train))

class_weight = {0 : 1.,
    1: (len(X_train)/class1),
    }

# Convert class vectors to binary class matrices.
y_train_temp = y_train
y_test_temp = y_test
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#########################################################################
#These four variables are shifted depending on what preprocess is tested#
#########################################################################
X_train -= mean 
X_train /= std

X_test -= mean
X_test /= std

filename = 'NoMeanSTD49'
#filename = 'OnlySTD49'
#filename = 'OnlyMean49'
#filename = 'BothMeanSTD49'

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

    model = None
    model = Sequential()

    model.add(Dense(1024, input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.25))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001))
    model.add(Dropout(0.25))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))


    opt = keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
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

    #model.save(filename + ':' + format(num_fold)+'.h5')

    #print('History: ' + history)
    #print('scores: ' + scores)

#    plt.plot(history.history['acc'])
#    plt.plot(history.history['val_acc'])
#    plt.title('model accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'validation'], loc='upper left')
#    plt.savefig(filename + ':' + format(num_fold) + ' Accuracy.png', bbox_inches='tight')
#    plt.clf()

    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'validation'], loc='upper left')
#    plt.savefig(filename + ':' + format(num_fold) + ' Loss.png', bbox_inches='tight')
#    plt.clf()

    #print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
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

#print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))

file.write('Average Results' + '\n')
file.write('training loss: mean: ' + format(np.mean(cvTrainLOSSscores)) + ' std: ' + format(np.std(cvTrainLOSSscores)) + '\n')
file.write('training accuracy: mean: ' + format(np.mean(cvTrainACCscores)) + ' std: ' + format(np.std(cvTrainACCscores)) + '\n')
file.write('validation loss: mean: ' + format(np.mean(cvValLOSSscores)) + ' std: ' + format(np.std(cvValLOSSscores)) + '\n')
file.write('validation accuracy: mean: ' + format(np.mean(cvValACCscores)) + ' std: ' + format(np.std(cvValACCscores)) + '\n')
file.write('Test loss: mean: ' + format(np.mean(cvTestLOSSscores)) + ' std: ' + format(np.std(cvTestLOSSscores)) + '\n')
file.write('Test accuracy: mean: ' + format(np.mean(cvTestACCscores)) + ' std: ' + format(np.std(cvTestACCscores)) + '\n' + '\n')

file.close()