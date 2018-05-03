import pandas as pd


def get_data():
  # training set
  df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')

  x_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column
  y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column
  
  X_trn.fillna(value=0, inplace=True)

  return (x_trn, y_trn)
