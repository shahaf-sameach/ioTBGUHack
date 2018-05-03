import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# training set
df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')

# validation set - anonymized
df_vld = pd.read_csv('hackathon_IoT_validation_set_based_on_01mar2017_ANONYMIZED.csv', low_memory=False, na_values='?')

# validation set - actual types (true labels)
#y_vld_ind_and_actual = pd.read_csv('hackathon_IoT_validation_set_based_on_01mar2017_COMPLETE.csv', low_memory=False, na_values='?'
#                                  ,usecols = ['session_ind', 'device_category'])

print(df_trn.shape) # dimensions (rows, columns)
print(df_trn.head()) # overview of first (last) rows
print(df_trn.describe().T) # basic stats per variable

print(df_trn.dtypes.value_counts())

print('the value counts of the target are:')
print(df_trn.iloc[:,-1].value_counts())
print(df_trn.iloc[:,-1].value_counts().plot(kind = 'bar'))

y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column
X_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column

X_trn.fillna(value=0, inplace=True)

# train a random forest
RFmodel = RandomForestClassifier(n_estimators=50)
RFmodel.fit(X_trn,y_trn)

# column names
X_names = list(X_trn) 

# feature importance as per the random forest
featimp = pd.Series(RFmodel.feature_importances_, index=X_names).sort_values(ascending=False)
print(featimp)


# X_vld = df_vld.iloc[:,0:(df_vld.shape[1]-1)].copy() # all but the last column
# y_vld = y_vld_ind_and_actual['device_category']

# print(X_vld.isnull().sum().value_counts())

# y_vld_pred = RFmodel.predict(X_vld)
# print(y_vld_pred)

# print(classification_report(y_vld, y_vld_pred))

# pd.crosstab(y_vld, y_vld_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# classification_accuracy = sum(y_vld==y_vld_pred)/len(y_vld)
# print(classification_accuracy)


