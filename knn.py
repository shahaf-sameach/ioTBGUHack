import pandas as pd
import pdb
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from data_handler import get_data

df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')

x_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column
x_trn.fillna(value=0, inplace=True)

y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column

values = df_trn.device_category.unique()
y_one_hot = [np.where(values==category)[0][0] for category in y_trn]
  
print("training")
# train knn
knn = KNeighborsClassifier()
knn.fit(x_trn, y_one_hot)
print(knn.score(x_trn, y_one_hot))