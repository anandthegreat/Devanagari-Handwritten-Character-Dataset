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
re = Reshape((7,7,1))(dense)
conv5 = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(re)
conv5 = BatchNormalization()(conv5)
conv6 = Conv2DTranspose(64,(3, 3), strides=2, activation='relu', padding='same')(conv5)
conv6 = BatchNormalization()(conv6)
conv7 = Conv2DTranspose(32,(3, 3), activation='relu', padding='same')(conv6)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7)
autoencoder = Model(inp, decoded)
autoencoder.summary()


# In[11]:


# In[25]:


# compile it using adam optimizer
# autoencoder.compile(optimizer="adam", loss="mse")     #Train it by providing training images

# fitted = autoencoder.fit(X_train, X_train, epochs=5)  
# model_json = autoencoder.to_json()
# with open("cae.json", "w") as json_file:
#     json_file.write(model_json)

# autoencoder.save_weights("cae.h5")
# print("Saved model")


# In[27]:


# loss = fitted.history['loss']
# epochs = [0,1,2,3,4]
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.xlabel('Epoch')
# plt.ylabel('Training loss')
# plt.title('Training loss vs Epoch')
# plt.legend()
# plt.show()


# In[49]:


train_Y_one_hot = to_categorical(Y_train)
test_Y_one_hot = to_categorical(Y_test)

out = Dense(46, activation='softmax')(dense)
final_model = Model(inp,out)
for l1,l2 in zip(final_model.layers[:8],autoencoder.layers[0:8]):
    l1.set_weights(l2.get_weights())
# autoencoder.get_weights()[0][1]
# final_model.get_weights()[0][1]
# for layer in final_model.layers[0:8]:
#     layer.trainable = False  #encoder part is already trained, only train fc part


# In[51]:


final_model.compile(loss="mse", optimizer="RMSprop",metrics=['accuracy'])
final_model.summary()


# In[58]:


classify_train = final_model.fit(X_train, train_Y_one_hot,epochs=20)

test_eval = final_model.evaluate(X_test,test_Y_one_hot,verbose = 0)
print('Test accuracy:', test_eval[1])

model_json = autoencoder.to_json()
with open("cae_RMSProp.json", "w") as json_file:
    json_file.write(model_json)

autoencoder.save_weights("cae_RMSProp.h5")

# In[50]:


# autoencoder.load_weights("cae.h5")


# In[ ]:


# getFeature = kb.function([autoencoder.layers[0].input, kb.learning_phase()],[autoencoder.layers[14].output])
# train = getFeature([x_train[:3000], 0])[0]

