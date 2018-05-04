import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_handler import get_data

x_trn, y_trn = get_data()
# train a random forest
RFmodel = RandomForestClassifier(n_estimators=50)
RFmodel.fit(x_trn,y_trn)

# column names
X_names = list(X_trn) 

# feature importance as per the random forest
featimp = pd.Series(RFmodel.feature_importances_, index=X_names).sort_values(ascending=False)
print(featimp)
