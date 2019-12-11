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
warnings.filterwarnings(action="ignore",category=DeprecationWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# In[ ]:
# =============================================================================
# # In[ ]:
# ## Ref : 
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
# RANDOM FOREST
# from sklearn.ensemble import RandomForestClassifier

def RFC(trees,depth):
    randomforest=RandomForestClassifier(n_estimators=trees,max_depth=depth,random_state=0)
    randomforest.fit(X_train,Y_train)
    predictions=randomforest.predict(X_validation)
    return accuracy(Y_validation,predictions)
#------------------------------------------------------------------------------
# In[ ]:
def RFC2(trees,depth):
    return (trees+depth)/300

# In[ ]:
#optimalDepth=0
#optimalTrees=0
#maxAccuracy=0
#
#depthArray=[1,2,3,4,5,6,7]
##depthArray=[2,4,8,16,32,64,128] Actual Used Value
#treesArray=[5,7.5,10,12.5,15.0,17.5,20]
#
#fig= plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#xs=treesArray
#y=depthArray # consider value of xs on y=1 , y=2 and so on  
#
#i=0    
#for c,y in zip (['r', 'g', 'b', 'y','c','m','k'],y):
#    currDepth=depthArray[i]
#    z=[]
#    for j in range (len(treesArray)):
#        currTree=treesArray[j]
#        acc=RFC((int)(10*currTree),pow(2,currDepth))
#        if(acc>=maxAccuracy):
#            maxAccuracy=acc
#            optimalDepth=pow(2,currDepth)
#            optimalTrees=10*currTree
#                
#        print('Trees :',(10*currTree),'Depth :',pow(2,currDepth),'Accuracy :',acc)
#        z.append(acc)
#    
#    cs = [c] * len(xs)
#    ax.bar(xs, z, zs=y, zdir='y', color=cs, alpha=0.8)
#    i=i+1
#    
#ax.set_xlabel('  trees count/10 ')
#ax.set_ylabel('  log-base-2(depth count)    ')
#ax.set_zlabel('  accuracy (%)   ')
#
#plt.savefig('GraphRFFinal.pdf')
#plt.show()
#
#print('Accuracy with Random Forest : ',maxAccuracy,'%. With Optimal Depth of each tree :',optimalDepth,' and Optimal no. of Trees :',optimalTrees)
# In[ ]:
print('Model will be trained on Complete Training Set...')
randomforest=RandomForestClassifier(n_estimators=200,max_depth=16,random_state=0)
#randomforest=RandomForestClassifier(n_estimators=4,max_depth=2,random_state=0)
randomforest.fit(X_train,Y_train)
print('Training Complete.')
print('Checking Accuracy    on Validation Set...')
predictions=randomforest.predict(X_validation)
acc1=accuracy(Y_validation,predictions)
print('Accuracy on Validation Set :',acc1)
print('Checking Accuracy on Test Set...')
predictions2=randomforest.predict(X_test)
acc2=accuracy(Y_test,predictions2)
print('Accuracy on Test Set :',acc2)
