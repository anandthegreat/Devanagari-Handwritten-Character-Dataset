import numpy as np
import pandas as pd
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, AveragePooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, History
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import winsound
import pickle

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# exit()

train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

X_train = train_data.drop(columns='character').to_numpy()
Y_train = train_data[['character']].to_numpy().flatten()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)

X_test = test_data.drop(columns='character').to_numpy()
Y_test = test_data[['character']].to_numpy().flatten()
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

print("X_test shape: ", X_test.shape)
print("Y_test shape: ", Y_test.shape)

encoder = LabelEncoder()
Y = np.append(Y_train, Y_test)
Y = encoder.fit_transform(Y)
Y_train = Y[:len(Y_train)]
Y_test = Y[len(Y_train):]
# print("Y_train categorical shape: ", Y_train.shape, Y_train)
# print("Y_test categorical shape: ", Y_test.shape, Y_test)


model = load_model('adadelta/adadelta_6.h5')
# print("Accuracy: ", model.evaluate(X_test, to_categorical(Y_test), verbose=0)[1])
# feature extraction layer
getFeature = K.function([model.layers[0].input, K.learning_phase()], [model.layers[7].output])

X_train_svm = np.empty((0, 1000))
for i in range(78200//200):
    fv = getFeature([X_train[200*i:200*(i+1)], 0])[0]
    X_train_svm = np.append(X_train_svm, fv, axis=0)

print("Training")
model_svm = svm.SVC(kernel='rbf', decision_function_shape="ova", gamma="auto")
model_svm.fit(X_train_svm, Y_train)

with open("pkl_svm", 'wb') as file:
    pickle.dump(model_svm, file)

winsound.MessageBeep()
X_test_svm = np.empty((0, 1000))
for i in range(13800//200):
    fv = getFeature([X_test[200*i:200*(i+1)], 0])[0]
    X_test_svm = np.append(X_test_svm, fv, axis=0)

print("Testing")

y_predict_train = model_svm.predict(X_train_svm)
as_train = accuracy_score(y_predict_train, Y_train)
print("Train accuracy score (SVM): ", as_train*100)

y_predict_test = model_svm.predict(X_test_svm)
as_train = accuracy_score(y_predict_test, Y_test)
print("Test accuracy score (SVM): ", as_train*100)
exit()


# model = load_model('cnn_models/model_sgd.h5')
# print("Accuracy: ", model.evaluate(X_train, to_categorical(Y_train), verbose=0)[1])
# exit()

############################# Part 1: Finding the best optimizer #############################
'''
# https://keras.io/optimizers/

# create model
model = Sequential()

# add model layers
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(46, activation='softmax'))


# Add a checkpoint
csv_logger = CSVLogger('logs/sgd.log')
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5)
saveModel = ModelCheckpoint(filepath="sgd.h5", monitor='val_accuracy', verbose=2, save_best_only=True)

# compile model using accuracy to measure model performance
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(X_train, to_categorical(Y_train), validation_data=(X_test, to_categorical(Y_test)), epochs=15, verbose=2,
          batch_size=200, callbacks=[csv_logger, earlyStopping, saveModel])

winsound.MessageBeep()
'''
############################# Part 2: Finding best set of parameters #############################
'''
# Running with one layer less
model = Sequential()
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(46, activation='softmax'))

# Add a checkpoint
csv_logger = CSVLogger('adadelta/adadelta_1.log')
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5)
saveModel = ModelCheckpoint(filepath="adadelta/adadelta_1.h5", monitor='val_accuracy', verbose=2, save_best_only=True)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, to_categorical(Y_train), validation_data=(X_test, to_categorical(Y_test)), epochs=50, verbose=2,
          batch_size=200, callbacks=[csv_logger, earlyStopping, saveModel])

winsound.MessageBeep()
'''

'''
model = Sequential()
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(AveragePooling2D(pool_size=2))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(AveragePooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(46, activation='softmax'))

csv_logger = CSVLogger('adadelta/adadelta_2.log')
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=5)
saveModel = ModelCheckpoint(filepath="adadelta/adadelta_2.h5", monitor='val_accuracy', verbose=2, save_best_only=True)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, to_categorical(Y_train), validation_data=(X_test, to_categorical(Y_test)), epochs=50, verbose=2,
          batch_size=200, callbacks=[csv_logger, earlyStopping, saveModel])

winsound.MessageBeep()
'''

'''
# Dropout - Average pooling
# Dropout: 0.1 = 0.98101, 0.2 = 0.98486, 0.3 = 0.98457, 0.4: 0.98464, 0.5: 0.98609
model = Sequential()
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(AveragePooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(AveragePooling2D(pool_size=2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(46, activation='softmax'))

csv_logger = CSVLogger('adadelta/adadelta_3.log')
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=6)
saveModel = ModelCheckpoint(filepath="adadelta/adadelta_3.h5", monitor='val_accuracy', verbose=2, save_best_only=True)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, to_categorical(Y_train), validation_data=(X_test, to_categorical(Y_test)), epochs=50, verbose=2,
          batch_size=200, callbacks=[csv_logger, earlyStopping, saveModel])

winsound.MessageBeep()
'''
# https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7
# https://ruder.io/optimizing-gradient-descent/
# https://www.kdnuggets.com/2018/09/dropout-convolutional-networks.html

'''
# Dropout - Max Pooling
model = Sequential()
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(46, activation='softmax'))

csv_logger = CSVLogger('adadelta/adadelta_5.log')
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=6)
saveModel = ModelCheckpoint(filepath="adadelta/adadelta_5.h5", monitor='val_accuracy', verbose=2, save_best_only=True)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, to_categorical(Y_train), validation_data=(X_test, to_categorical(Y_test)), epochs=50, verbose=2,
          batch_size=200, callbacks=[csv_logger, earlyStopping, saveModel])

winsound.MessageBeep()
'''
'''
# Batch Normalization
model = Sequential()
model.add(Conv2D(256, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=2))
model.add(Conv2D(64, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(46, activation='softmax'))

csv_logger = CSVLogger('adadelta/adadelta_7.log')
earlyStopping = EarlyStopping(monitor='val_accuracy', patience=6)
saveModel = ModelCheckpoint(filepath="adadelta/adadelta_7.h5", monitor='val_accuracy', verbose=2, save_best_only=True)

model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, to_categorical(Y_train), validation_data=(X_test, to_categorical(Y_test)), epochs=50,
                    verbose=2, batch_size=200, callbacks=[csv_logger, earlyStopping, saveModel])

winsound.MessageBeep()

with open("pkl_history", 'wb') as file:
    pickle.dump(history, file)


with open("pkl_history", 'rb') as file:
    history = pickle.load(file)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy (Batch Normalization and Average Pooling)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss (Batch Normalization and Average Pooling)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()
'''
# https://github.com/shibuiwilliam/Keras_Sklearn/blob/master/Cifar_CNN_SKLearn.ipynb
