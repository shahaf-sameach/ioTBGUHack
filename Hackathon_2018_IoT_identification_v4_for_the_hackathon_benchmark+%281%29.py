
# coding: utf-8

# # imports

# In[1]:


from __future__ import division
import os # to set working directory
import csv # to read/write csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from IPython.core.display import display
from sklearn.preprocessing import Imputer # for imputing missing values
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns # for heatmaps


# # parameters

# In[2]:


prm_dir='C:/Users/home/Dropbox/IoT_Research/hackathon_2018_IoT_device_type_identification'


# ## set working directory

# In[4]:


os.chdir(prm_dir)
# os.listdir(prm_dir)


# ## set random seed (for reproducibility)

# In[5]:


np.random.seed(23)


# # read the data sets

# In[18]:


# training set
df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')

# validation set - anonymized
df_vld = pd.read_csv('hackathon_IoT_validation_set_based_on_01mar2017_ANONYMIZED.csv', low_memory=False, na_values='?')

# validation set - actual types (true labels)
y_vld_ind_and_actual = pd.read_csv('hackathon_IoT_validation_set_based_on_01mar2017_COMPLETE.csv', low_memory=False, na_values='?'
                                  ,usecols = ['session_ind', 'device_category'])


# ## explore the training data

# In[7]:


print(df_trn.shape) # dimensions (rows, columns)
display(df_trn.head()) # overview of first (last) rows
display(df_trn.describe().T) # basic stats per variable


# ## data types

# In[8]:


df_trn.dtypes.value_counts()


# ## data exploration

# In[9]:


print('the value counts of the target are:')
print(df_trn.iloc[:,-1].value_counts())
print(df_trn.iloc[:,-1].value_counts().plot(kind = 'bar'))


# ## prepare data for modeling

# In[10]:


y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column
X_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column


# In[11]:


X_trn.head()


# ## handle missing values

# In[12]:


# summary of missing values
X_trn.isnull().sum().value_counts()


# In[13]:


# elaborate on the number of missing values per feature
with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
    print(df_trn.isnull().sum())


# In[14]:


# fill values that are missing because the respective sessions are
# 1. not HTTP
# 2. not SSL
X_trn.fillna(value=0, inplace=True)


# ## train a classifier

# In[15]:


# train a random forest
RFmodel = RandomForestClassifier(n_estimators=50)
RFmodel.fit(X_trn,y_trn)


# In[16]:


# column names
X_names = list(X_trn) 

# feature importance as per the random forest
featimp = pd.Series(RFmodel.feature_importances_, index=X_names).sort_values(ascending=False)
print(featimp)


# In[17]:


featimp[featimp>0.02].keys


# # validation

# In[23]:


X_vld = df_vld.iloc[:,0:(df_vld.shape[1]-1)].copy() # all but the last column
y_vld = y_vld_ind_and_actual['device_category']


# examine missing values in the validation set

# In[20]:


X_vld.isnull().sum().value_counts()


# In[21]:


# fill missing values
# 1. not HTTP
# 2. not SSL
# X_vld.fillna(value=0, inplace=True)


# ## apply model on the validation model for prediction

# In[22]:


# predict class
y_vld_pred = RFmodel.predict(X_vld)
print(y_vld_pred)


# ## examine classification accuracy

# In[24]:


print(classification_report(y_vld, y_vld_pred))
# precision = tp / (tp + fp) = tp / "p" = the ability of the classifier not to label as positive a sample that is negative
# recall = tp / (tp + fn) = tp / p = the ability of the classifier to find all the positive samples
# f1-score = weighted harmonic mean of the precision and recall
# support = number of occurrences of each class in y_true


# In[25]:


# confusion_matrix
pd.crosstab(y_vld, y_vld_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# ## calculate the classification accuracy on the validation set

# In[31]:


classification_accuracy = sum(y_vld==y_vld_pred)/len(y_vld)
classification_accuracy

