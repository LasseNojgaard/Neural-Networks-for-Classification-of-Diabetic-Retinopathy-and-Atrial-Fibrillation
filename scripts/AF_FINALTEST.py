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
from keras.callbacks import ModelCheckpoint
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
        print("Confusion matrix")
    else:
        print('Confusion matrix')

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


def create_model_FullyCon(X_train, hidden_layer=32, dropout=0):

    input_layer = Input(X_train.shape[1:], name="base_input")
    x1 = Dense(hidden_layer, name='DENSE_1', activation="relu")(input_layer)
    x1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x1)
    x1 = Dropout(dropout)(x1)

    x2 = Dense(hidden_layer, name='DENSE_2')(x1)
    x2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x2)
    x2 = Dropout(dropout)(x2)

    x3 = Dense(hidden_layer, name='DENSE_3')(x2)
    x3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x3)
    x3 = Dropout(dropout)(x3)

    x4 = Dense(hidden_layer, name='DENSE_4')(x3)
    x4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x4)
    x4 = Dropout(dropout)(x4)

    x5 = Dense(hidden_layer, name='DENSE_5')(x4)
    x5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x5)
    x5 = Dropout(dropout)(x5)

    x6 = Dense(hidden_layer, name='DENSE_6')(x5)
    x6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x6)
    x6 = Dropout(dropout)(x6)

    out = Dense(2, name="output")(x6)
    out = Activation('softmax', name="SOFTMAX")(out)
    return Model(inputs=input_layer, outputs=out, name="base_model")



def create_model_DenseNet(X_train, hidden_layer=32, dropout=0):
    input_layer = Input(X_train.shape[1:], name="base_input")
    x1 = Dense(hidden_layer, name='DENSE_1', activation="relu")(input_layer)
    x1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x1)
    x1 = Dropout(dropout)(x1)

    x2 = Dense(hidden_layer, name='DENSE_2')(x1)
    x2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x2)
    concentrated12 = Concatenate(name="ConX1X2")([x1, x2])
    concentrated12 = Dropout(dropout)(concentrated12)

    x3 = Dense(hidden_layer, name='DENSE_3', activation="relu")(concentrated12)
    x3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x3)
    concentrated123 = Concatenate(name="ConX1X2X3")([concentrated12, x3])
    concentrated123 = Dropout(dropout)(concentrated123)

    x4 = Dense(hidden_layer, name="Dense_4")(concentrated123)
    x4 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x4)
    concentrated1234 = Concatenate(name="ConX1X2X3X4")([concentrated123, x4])
    concentrated1234 = Dropout(dropout)(concentrated1234)

    x5 = Dense(hidden_layer, name='DENSE_5', activation="relu")(concentrated1234)
    x5 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x5)
    concentrated12345 = Concatenate(name="ConX1X2X3X4X5")([concentrated1234, x5])
    concentrated12345 = Dropout(dropout)(concentrated12345)

    x6 = Dense(hidden_layer, name='DENSE_6', activation="relu")(concentrated12345)
    x6 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x6)
    concentrated123456 = Concatenate(name="ConX1X2X3X4X5X6")([concentrated12345, x6])
    concentrated123456 = Dropout(dropout)(concentrated123456)

    x7 = Dense(hidden_layer, name='DENSE_7', activation="relu")(concentrated123456)
    x7 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x7)
    concentrated1234567 = Concatenate(name="ConX1X2X3X4X5X6X7")([concentrated123456, x7])
    concentrated1234567 = Dropout(dropout)(concentrated1234567)

    x8 = Dense(hidden_layer, name='DENSE_8', activation="relu")(concentrated1234567)
    x8 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x8)
    concentrated12345678 = Concatenate(name="ConX1X2X3X4X5X6X7X8")([concentrated1234567, x8])
    concentrated12345678 = Dropout(dropout)(concentrated12345678)

    x9 = Dense(hidden_layer, name='DENSE_9', activation="relu")(concentrated12345678)
    x9 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x9)
    concentrated123456789 = Concatenate(name="ConX1X2X3X4X5X6X7X8X9")([concentrated12345678, x9])
    concentrated123456789 = Dropout(dropout)(concentrated123456789)

    out = Dense(2, name="output")(concentrated123456789)
    out = Activation('softmax', name="SOFTMAX")(out)
    return Model(inputs=input_layer, outputs=out, name="base_model")



def create_model_ResNet(X_train, hidden_layer=32, dropout=0):
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

X_train = dataRed
X_test = dataRedTest
y_train = dataLabel
y_test = dataLabelTest

mean = np.mean(X_train, axis = 0)
std = np.std(X_train, axis = 0)


batch_size = 64
num_classes = 2

# Convert class vectors to binary class matrices.
y_train_temp = y_train
y_test_temp = y_test
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


X_train -= mean
X_train /= std
X_test -= mean
X_test /= std

epochs = 150
files = "FinalTest"
file = open('AF_' + files + '.txt', 'w')
archList = ["Fully", "ResNet", "DenseNet"]

for arch in archList:
    filename = files + "_" + arch
    if arch == "Fully":
        learning_rate=0.0001
        hidden_layer=1024
        learning_decay=0
        dropout = 0
        opti ='adam'
        model = create_model_FullyCon(X_train, hidden_layer, dropout)
    if arch == "DenseNet":
        learning_rate=0.00001
        hidden_layer= 2048
        learning_decay= 0.00001
        dropout = 0.1
        opti = "SGD"
        model = create_model_DenseNet(X_train, hidden_layer, dropout)
    if arch == "ResNet":
        learning_rate=0.0001
        hidden_layer=1024
        learning_decay=0.0001
        dropout = 0.1
        opti = "adam"
        model = create_model_ResNet(X_train, hidden_layer, dropout)

    if opti == 'adam':
        opt = keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=learning_decay, amsgrad=False)
    if opti == 'RMS':
        opt = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=learning_decay)
    if opti == 'SGD':
        opt = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=learning_decay, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), shuffle=True, verbose=1)
    scores = model.evaluate(X_test, y_test, verbose=0)
    file.write("\n"+ "ARCH " + arch + ":\n" )
    file.write('training loss: ' + format(history.history['loss'][epochs - 1]) + '\n')
    file.write('training accuracy: ' + format(history.history['acc'][epochs - 1]) + '\n')
    file.write('test loss: ' + format(history.history['val_loss'][epochs - 1]) + '\n')
    file.write('test accuracy: ' + format(history.history['val_acc'][epochs - 1]) + '\n')
    file.write('Test loss: ' + format(scores[0]) + '\n')
    file.write('Test accuracy: ' + format(scores[1]) + '\n' + '\n')

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename + '_Acc.png', bbox_inches='tight')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(filename + '_Loss.png', bbox_inches='tight')
    plt.clf()

    predictions_labels = []
    predictionsTemp = []
    predictionsTemp = model.predict(X_test, batch_size=16)
    for ii in range(0, len(predictionsTemp)):
        predictions_labels = np.append(predictions_labels, np.argmax(predictionsTemp[ii]))

    cm = confusion_matrix(y_test_temp, predictions_labels)
    class_names = ['No_Redo', 'Redo']
    plot_confusion_matrix(cm, classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig(filename + '_CM' + '.png')
    plt.clf()

    target_names = ['class 0', 'class 1']
    report_TwoClass = precision_recall_fscore_support(y_test_temp, predictions_labels)
    reportAvarage_TwoClass = precision_recall_fscore_support(y_test_temp, predictions_labels, average="macro")
    kappa_TwoClass = cohen_kappa_score(y_test_temp, predictions_labels, weights='quadratic')
    file.write("\n" + "Two Class:" + "\n")
    file.write(
        "Class" + "  " + "Precision" + "  " + "Recall" + "  " + "F-score" + "  " + "Support" + "  " + "Kappa" + '\n')
    for x in range(0, 3):
        if (x >= 2):
            file.write(
                "Aerage" + "  " + format(reportAvarage_TwoClass[0]) + "  " + format(
                    reportAvarage_TwoClass[1]) + "  " + format(
                    reportAvarage_TwoClass[2]) + "  " + "  " + format(kappa_TwoClass) + '\n')
        else:
            file.write(
                "Class" + format(x) + "  " + format(report_TwoClass[0][x]) + "  " + format(
                    report_TwoClass[1][x]) + "  " + format(
                    report_TwoClass[2][x]) + "  " + format(report_TwoClass[3][x]) + '\n')

    del model
    del history
    del scores
    gc.collect()
    K.clear_session()
file.close()