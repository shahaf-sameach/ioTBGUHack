import pandas as pd
import numpy as np
import pdb
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
# from nn import Network
from Yechiav_TEst import DataImport_and_preProdceesing, Test_Data_Train

np.random.seed(23)

models = dict()
df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')
for category in df_trn.device_category.unique():
    tmp_df_trn = df_trn.copy()
    tmp_df_trn.loc[tmp_df_trn['device_category'] != category ,'device_category'] = 'unk'

    selected = tmp_df_trn.loc[tmp_df_trn['device_category'] == category]
    unk = tmp_df_trn.loc[tmp_df_trn['device_category'] != category]
    unk = unk.iloc[:selected.shape[0], :]

    tmp_df_trn = pd.concat([unk, selected])
    #tmp_df_trn = df_trn.copy()

    x_trn = tmp_df_trn.iloc[:, 0:(tmp_df_trn.shape[1]-1)].copy()
    x_trn.fillna(value=0, inplace=True)

    y_trn = tmp_df_trn.iloc[:,(tmp_df_trn.shape[1]-1)].copy() # last column

    categories = ['unk', category]
    y_one_hot = [categories.index(c) for c in y_trn]

    X_train, X_test, y_train, y_test = train_test_split(x_trn, y_one_hot)
    # X_train, tuple1 = DataImport_and_preProdceesing(X_train)
    # X_test = Test_Data_Train(X_test, tuple1)


    #pdb.set_trace()
    print("running on {}".format(category))
    # train logistic regression
    logmodel = RandomForestClassifier(n_estimators=100)
    logmodel.fit(X_train, y_train)
    models[category] = logmodel
    print("test" ,logmodel.score(X_train, y_train))
    print("train", logmodel.score(X_test, y_test))
    print("---"*10)

x_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column
x_trn.fillna(value=0, inplace=True)

y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column

rf_model = RandomForestClassifier(n_estimators=100, max_depth=4)
rf_model.fit(x_trn,y_trn)

df_trn = pd.read_csv('hackathon_IoT_validation_set_based_on_01mar2017_ANONYMIZED.csv', low_memory=False, na_values='?')
x_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column
x_trn.fillna(value=0, inplace=True)

categories_model = [k for k in models.keys()]
y_pred = []
for category in categories_model:
  pdb.set_trace()
  prediction = models[category].predict_proba(x_trn)
  tmp = []
  for row in prediction:
    if row[1] > 0.9:
      tmp.append(1)
    else:
      tmp.append(0)
  y_pred.append(tmp)

#pdb.set_trace()
yy = np.array(y_pred).T
#pdb.set_trace()
yt = rf_model.predict(x_trn)

# neural net for classifing ambiguous data
# net = Network()
# df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')
# x_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column
# x_trn.fillna(value=0, inplace=True)

# y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column
# y_trn = pd.get_dummies(y_trn)
# net.train(x_trn,y_trn)

# yt = net.predict(x_trn)

preds = []
#pdb.set_trace()
counter = 0
for idx, row in enumerate(yy):
  if(sum(row)) == 1:
    preds.append(categories_model[np.where(row==1)[0][0]])
  elif(sum(row)) == 0:
    preds.append('unknown')
    counter += 1
  else:
    preds.append("{}".format(yt[idx]))

print(counter)
for idx,p in enumerate(preds):
  print("{},{}".format(p,idx + 1))






