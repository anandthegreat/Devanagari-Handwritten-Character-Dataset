#!/usr/bin/env python
# coding: utf-8

# In[29]:


import tensorflow
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.models import model_from_json
from keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Deconvolution2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam
from keras.optimizers import RMSprop
from sklearn.metrics import accuracy_score
from keras import backend as kb
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


# In[2]:


from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# In[4]:
train_data = pd.read_csv('data.csv')
test_data = pd.read_csv('data_test.csv')

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
print("Y_train categorical shape: ", Y_train.shape, Y_train)
print("Y_test categorical shape: ", Y_test.shape, Y_test)
X_train, X_test = X_train / 255.0, X_test / 255.0




#AUTOENCODER
inp = Input((28, 28,1))                         #28x28x1 : wide and thin input images
conv1 = Conv2D(32, (3, 3), activation='relu')(inp)      #32 filters of size  3x3   => 28x28x32 images
pool1 = MaxPooling2D((2, 2))(conv1)                     #downsampling              => 14x14x32 images
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)    #64 filters of size  3x3   => 14x14x64 images
pool2 = MaxPooling2D((2, 2))(conv2)                     #downsampling              => 7x7x64   images
conv3 = Conv2D(64, (3, 3), activation='relu')(pool2)    #128 filters of size 3x3   => 7x7x128  images (small and more channels)
flat = Flatten()(conv3)
dense = Dense(49, activation='softmax')(flat)

#AUTODECODER
# re = Reshape((7,7,1))(dense)
# conv5 = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(re)
# conv5 = BatchNormalization()(conv5)
# conv6 = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(conv5)
# conv6 = BatchNormalization()(conv6)
# conv7 = Conv2DTranspose(32,(3, 3), activation='relu', padding='same')(conv6)
# decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)


out = Dense(46, activation='softmax')(dense)
autoencoder = Model(inp, out)
autoencoder.summary()


# In[11]:

train_Y_one_hot = to_categorical(Y_train)[]
test_Y_one_hot = to_categorical(Y_test)

nepochs = 2
# compile it using adam optimizer
autoencoder.compile(optimizer="adam", loss="mse",metrics=['accuracy'])     #Train it by providing training images
print(train_Y_one_hot.shape)
print(X_train.shape)
classify_train = autoencoder.fit(X_train, train_Y_one_hot,epochs=nepochs)

# saveModel = ModelCheckpoint(filepath="sgd.h5", monitor='val_accuracy', verbose=2, save_best_only=True)


getFeature = kb.function([autoencoder.layers[0].input, kb.learning_phase()],[autoencoder.layers[7].output])
train = getFeature([X_train, 0])[0]
test = getFeature([X_test, 0])[0]

X_train_svm = np.empty((0, 49))
for i in range(78200//200):
    fv = getFeature([X_train[200*i:200*(i+1)], 0])[0]
    X_train_svm = np.append(X_train_svm, fv, axis=0)

print("Training")
model_svm = svm.SVC(kernel='rbf', decision_function_shape="ova", gamma="auto")
model_svm.fit(X_train_svm, Y_train)


X_test_svm = np.empty((0, 49))
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




loss = classify_train.history['loss']
epochs = range(nepochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('Training loss vs Epoch')
plt.legend()
plt.show()

with open("pkl_svm", 'wb') as file:
    pickle.dump(model_svm, file)