import pandas as pd
import numpy as np
import pdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

selected_category = 'socket'

df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')
df_trn.loc[df_trn['device_category'] != selected_category ,'device_category'] = 'unk'

watch = df_trn.loc[df_trn['device_category'] == selected_category]
unk = df_trn.loc[df_trn['device_category'] != selected_category]
unk = unk.iloc[:watch.shape[0], :]

tmp_df_trn = pd.concat([unk, watch])

x_trn = tmp_df_trn.iloc[:, 0:(df_trn.shape[1]-1)].copy()
x_trn.fillna(value=0, inplace=True)

y_trn = tmp_df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column

categories = tmp_df_trn.device_category.unique()
y_one_hot = [np.where(categories==category)[0][0] for category in y_trn]

X_train, X_test, y_train, y_test = train_test_split(x_trn, y_one_hot)

#pdb.set_trace()
print("running")
# train logistic regression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
print("test" ,logmodel.score(X_train, y_train))
print("train", logmodel.score(X_test, y_test))

