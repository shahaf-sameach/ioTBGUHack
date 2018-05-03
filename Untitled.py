
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb

np.random.seed(2018)

my_color_map = ['green','aqua','pink','blue','red','black','yellow','teal','orange','grey']


# In[2]:


tr_data = pd.read_csv('../input/train.csv')
te_data = pd.read_csv('../input/test.csv')
print('train shape is: {} \r\n\ test shape is: {}'.format(tr_data.shape, te_data.shape))


# In[3]:


pd.options.display.max_rows = 200
pd.options.display.max_columns = 50
tr_data.describe().T


# In[4]:


print('the value counts of the target are:')
print(tr_data.iloc[:,-1].value_counts())
print(tr_data.iloc[:,-1].value_counts().plot(kind = 'bar'))


# In[5]:


def value_counts_plots(dat,rows = 4, cols = 4):
    _,ax = plt.subplots(rows,cols,sharey='row',sharex='col',figsize = (cols*5,rows*5))
    for i,feat in enumerate(dat.columns[:(rows*cols)]):
        dat[feat].value_counts().iloc[:20].plot(kind = 'bar',ax=ax[int(i/cols), int(i%cols)],title='value_counts {}'.format(feat))

value_counts_plots(tr_data.iloc[:,1:9],2,4)


# In[6]:


tr_data['source'] = 'train'
te_data['source'] = 'test'
all_data = pd.concat([tr_data,te_data],axis=0)
tr_data.drop('source',axis=1,inplace=True)
te_data.drop('source',axis=1,inplace=True)

molten = pd.melt(all_data, id_vars = 'source',value_vars = ['feat_'+str(x) for x in range(46,55)])
plt.subplots(figsize = (20,8))
sns.violinplot(data=molten, x= 'variable',y='value',hue='source',split = True,palette=my_color_map)


# In[7]:


from sklearn.model_selection import train_test_split
tr_data['parsed_target'] = [int(n.split('_')[1]) for n in tr_data.target]
tr_data.drop('target',axis=1,inplace=True)
X_train, X_val, y_train, y_val = train_test_split(tr_data.iloc[:,1:-1],tr_data.parsed_target,test_size = 0.2,random_state =12345)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
knn = KNeighborsClassifier(n_jobs=8,n_neighbors=4)
knn.fit(X_train,y_train)
knn2_pred = knn.predict(X_val)


# In[10]:


print(confusion_matrix(y_pred=knn2_pred,y_true=y_val))
sns.heatmap(xticklabels=range(1,10),yticklabels=range(1,10),
            data = confusion_matrix(y_pred=knn2_pred,y_true=y_val),cmap='Greens')
plt.xlabel('predicted class')
plt.ylabel('actual class')


# In[11]:


from sklearn.metrics import classification_report
print('classification report results:\r\n' + classification_report(y_pred=knn2_pred,y_true=y_val))


# In[13]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=100,max_features=92,min_samples_split=2,random_state=12345)
dtc.fit(X_train,y_train)
tree_pred = dtc.predict(X_val)
print(confusion_matrix(y_pred=tree_pred,y_true=y_val))
sns.heatmap(confusion_matrix(y_pred=tree_pred,y_true=y_val),cmap='Greens',xticklabels=range(1,10),yticklabels=range(1,10))
plt.xlabel('predicted class')
plt.ylabel('actual class')
print('classification report results:\r\n' + classification_report(y_pred=tree_pred,y_true=y_val))


# In[14]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_jobs=4,n_estimators=100)
rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_val)
print(confusion_matrix(y_pred=rfc_pred,y_true=y_val))
sns.heatmap(confusion_matrix(y_pred=rfc_pred,y_true=y_val),cmap='Greens',xticklabels=range(1,10),yticklabels=range(1,10))
plt.xlabel('predicted class')
plt.ylabel('actual class')
print('classification report results:\r\n' + classification_report(y_pred=rfc_pred,y_true=y_val))


# In[15]:


import xgboost as xgb

dtrain = xgb.DMatrix(data=X_train,label=y_train-1) #xgb classes starts from zero
dval = xgb.DMatrix(data=X_val,label=y_val-1) #xgb classes starts from zero
watchlist = [(dval,'eval'), (dtrain,'train')]

xgb_params = {
    'eta': 0.05,
    'max_depth': 7,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'colsample_bylevel': 0.7,
    'alpha':0.1,
    #'objective': 'binary:logistic',
    'objective': 'multi:softmax',
    #'eval_metric': 'auc',
    'eval_metric': 'mlogloss',
    'watchlist':watchlist,
    'print_every_n':5,
    'min_child_weight':2,
    'num_class' : 9
}

bst = xgb.train(params=xgb_params,dtrain=dtrain,num_boost_round=400)
xgb_pred = bst.predict(dval)
print(confusion_matrix(y_pred=xgb_pred,y_true=y_val))
sns.heatmap(confusion_matrix(y_pred=xgb_pred+1,y_true=y_val),cmap='Greens',xticklabels=range(1,10),yticklabels=range(1,10))
plt.xlabel('predicted class')
plt.ylabel('actual class')
print('classification report results:\r\n' + classification_report(y_pred=xgb_pred+1,y_true=y_val))


# In[16]:


test_pred = bst.predict(xgb.DMatrix(te_data.iloc[:,1:]))


# In[21]:


test_pred


# In[20]:


subm = pd.get_dummies(test_pred)
subm.columns = ['class_'+ str(x) for x in range(1,10)]
subm.index = te_data.id
subm.to_csv('../subm/xgboost_classification_submission.csv')

