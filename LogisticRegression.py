# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 12:53:56 2019

@author: Akash
"""
#!/usr/bin/env python
# coding: utf-8
# In[ ]:

import os
import numpy as np
import pandas as pd
from scipy.misc import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings(action="ignore",category=DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# In[ ]:
# ##PLAGIARIZED CELL
# 
# root_dir = os.getcwd()
# img_dir = os.path.join(root_dir, 'train')
# 
# pixels = np.array(['pixel_{:04d}'.format(x) for x in range(1024)])
# flag = True
# 
# for char_name in sorted(os.listdir(img_dir)):
#     char_dir = os.path.join(img_dir, char_name)
#     img_df = pd.DataFrame(columns=pixels)
#     
#     for img_file in sorted(os.listdir(char_dir)):
#         image = pd.Series(imread(os.path.join(char_dir, img_file)).flatten(), index=pixels)
#         img_df = img_df.append(image.T, ignore_index=True)
#         
#     img_df = img_df.astype(np.uint8)
#     img_df['character'] = char_name
#     
#     img_df.to_csv('data.csv', index=False, mode='a', header=flag)
#     flag=False
# 
# 
# =============================================================================
# In[ ]:
print('Reading data.csv into a dataframe...')
df = pd.read_csv('data.csv')
df['character_class'] = LabelEncoder().fit_transform(df.character)
df.drop('character', axis=1, inplace=True)
df = df.astype(np.uint8)
print('reading data.csv completed ')

# =============================================================================
print('Reading data_test.csv into a dataframe...')
dftest = pd.read_csv('data_test.csv')
dftest['character_class'] = LabelEncoder().fit_transform(dftest.character)
dftest.drop('character', axis=1, inplace=True)
dftest = dftest.astype(np.uint8)
print('reading data_test.csv completed ')
 
# =============================================================================
# Dimension of df is: [78200 x 1025]
# 1700  =  (85%) of 2000 (each class has 2000 samples)
# 78200 =  1700*46 characters 
# 46    =  number of classes
# 1025  = 32*32 = 1024 feature columns, one output column

# In[ ]:
print('converting into numpy and shuffling datasets...')

dataset = df.to_numpy()
np.random.shuffle(dataset)
print(dataset)
print(dataset.shape)
X_train = dataset[0:15000,:-1]
Y_train = dataset[0:15000, -1]
X_validation = dataset[15000:18000,:-1]
Y_validation = dataset[15000:18000, -1]
print(X_train.shape)
print(Y_train.shape)
print(X_validation.shape)
print(Y_validation.shape)

dataset_Test = dftest.to_numpy()
np.random.shuffle(dataset_Test)
print(dataset_Test)
print(dataset_Test.shape)

X_test = dataset_Test[0:3000,:-1]
Y_test = dataset_Test[0:3000, -1]
print(X_test.shape)
print(Y_test.shape)

print('converting into numpy and shuffling done.')

# In[ ]:
def accuracy(actual,predicted):
    count=0
    for i in range(len(predicted)):
        diff=abs( (int)(actual[i]) - (int)(predicted[i]))
        #diff=abs(Y_validation[i]-predictions[i])
        if(diff==0):
            count=count+1
    accuracy = (count*100.0)/(len(predicted))
    return accuracy

# In[ ]: 
def Logistic(typ,subtyp):
    logistic = LogisticRegression(solver=typ,multi_class=subtyp)
    logistic.fit(X_train,Y_train)
    predictions1=logistic.predict(X_validation)
    predictions2=logistic.predict(X_test)
    print(predictions1.shape)
    print(predictions2.shape)
    acc1=accuracy(Y_validation,predictions1)
    acc2=accuracy(Y_test,predictions2)
    print('For Type : ',typ)
    print('Accuracy with Logistic Regression (One Vs Rest) Validation: ',acc1,'%.')
    print('Accuracy with Logistic Regression (One Vs Rest) Test      : ',acc2,'%.')
    #return acc2

# In[ ]:
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
#------------------------------------------------------------------------------
Logistic('sag','multinomial')
#types=(['newton-cg','lbfgs', 'sag', 'saga'])
##
#
#part1=[]
#for i in range (len(types)):
#    a=Logistic(types[i],'multinomial')
#    part1.append(a)
#part1.append(50)
#
#types.append('liblinear')
#part2=[]
#for i in range (len(types)):
#    a=Logistic(types[i],'ovr')
#    part2.append(a)
#
#print('For validation data...')
#xAxis=np.arange(1,6,1)
#plt.plot(xAxis,part1,label="Multinomial",)
#plt.plot(xAxis,part2,label="OVR")
#plt.title("Variation of accuracy with type of logistic regression algorithms on ovr and multinomial scheme")
#plt.xlabel("Algorithm (See Key)")
#plt.ylabel("Accuracy")
##plt.grid(True)
#plt.legend()
#plt.show()
