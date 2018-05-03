
# coding: utf-8

# In[1]:


import os # to set working directory
import csv # to read/write csv files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.display import display
from sklearn.preprocessing import Imputer # for imputing missing values
from sklearn import metrics
from sklearn.metrics import accuracy_score,precision_score,f1_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns # for heatmaps
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', '')


# ## Read CSV

# In[2]:


df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')

# validation set - anonymized
df_vld = pd.read_csv('hackathon_IoT_validation_set_based_on_01mar2017_ANONYMIZED.csv', low_memory=False, na_values='?')


# ### Split to X and Y 

# In[18]:


y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column
X_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column
categories = df_trn.device_category.unique()
y_one_hot = [np.where(categories==category)[0][0] for category in y_trn]


# ## Get Dummies to split models

# In[ ]:


y_trn_dummies=pd.get_dummies(y_trn,drop_first=True)


# ## Fill NAs

# In[20]:


X_trn.fillna(value=0,inplace=True)


# In[ ]:


y_trn_dummies.columns


# In[ ]:


Probabilities={}
threshold_dic={}
for y in y_trn_dummies.columns:
    X_train, X_test, y_train, y_test = train_test_split(X_trn, y_trn_dummies[y], test_size=0.3, random_state=1000)
    RFmodel = RandomForestClassifier(n_estimators=10)
    RFmodel.fit(X_train,y_train)
    predictions_rfc_prob = RFmodel.predict_proba(X_test)
    prob_list_rfc= [x[1] for x in predictions_rfc_prob]
    d_rfc={}
    for threshold in np.arange(0.0, 1.0, 0.01):
        list_for_check_rfc=np.int_([z>=threshold for z in prob_list_rfc])
        d_rfc[threshold]=accuracy_score(y_test,list_for_check_rfc)
    df_rfc=pd.DataFrame.from_dict(d_rfc,orient='index')
    Probabilities[y]=prob_list_rfc>df_rfc[df_rfc[0]==df_rfc[0].max()].index.min()
        


# In[ ]:


features=['baby_monitor', 'lights', 'motion_sensor', 'security_camera',
          'smoke_detector', 'socket', 'thermostat', 'watch', 'water_sensor']
features_zero_count={}
for label in features:
    for y in Probabilities[label]:
        if y==False: 
            if label not in features_zero_count.keys():
                features_zero_count[label]=1
            else:
                features_zero_count[label]+=1
features_zero_count
   


# In[ ]:


Classes_df=pd.DataFrame.from_dict(Probabilities,orient='index').transpose()


# In[ ]:


unknown_predictions=Classes_df.eq(False,axis=0).all(axis=1)


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X_trn,y_trn, test_size=0.3, random_state=1000)
RFmodel = RandomForestClassifier()
RFmodel.fit(X_train,y_train)


# In[23]:


predictions_RFC=RFmodel.predict(X_test)


# In[24]:


print(classification_report(y_test,predictions_RFC))


# In[26]:


accuracy_score(y_test,predictions_RFC)

