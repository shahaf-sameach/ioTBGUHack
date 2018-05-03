import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from data_handler import get_data

x_trn, y_trn = get_data()

# # train a random forest
# RFmodel = RandomForestClassifier(n_estimators=50)
# RFmodel.fit(x_trn, y_trn)
print("running")
# train logistic regression
logmodel=LogisticRegression()
logmodel.fit(x_trn, y_trn)

# # train knn
# knn = KNeighborsClassifier(n_neighbors=9)
# knn.fit(x_trn, y_trn)
