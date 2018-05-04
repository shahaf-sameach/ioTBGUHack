from __future__ import division

import os  # to set working directory
import time

import numpy as np
import pandas as pd
from IPython.core.display import display
from sklearn.model_selection import train_test_split

from ioTBGUHack import Preprocess

skew_thershold=0.9
kurt_thershold=0.3

pd.set_option('display.max_rows', 400)
# DataSet = "hackathon_IoT_training_set_based_on_01mar2017.csv"
DataSet = "Sample_2.csv"

file_name="Results"
prm_dir=r"C:\Users\hp\Google Drive\studies\Hackton\DataSet"

def Print_DF_to_file(DF):
    TimeStamp  = time.strftime("%Y%m%d-%H%M%S")
    DF.to_csv(file_name + TimeStamp+ '.csv', encoding='utf-8')

def print_unique(DF):
    for col in DF.columns:
        print(col)
        print(DF[col].unique()[:20])


os.chdir(prm_dir)

np.random.seed(23)

# training set
df_trn = pd.read_csv(DataSet, low_memory=False, na_values='?')

# df_vld = pd.read_csv("hackathon_IoT_ANONYMIZED.csv", low_memory=False, na_values='?')


# (df_trn.iloc[:,-1].value_counts().plot(kind = 'bar'))

y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column
X_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() #

DF_columns = df_trn.columns

PD_Data_tranformation = pd.DataFrame(columns=["Dtype","Remove","To_Bin","BinSize","Get_Dummies"], index=DF_columns)

PD_Data_tranformation.loc["ack_A"] =                                   ["NA",False,False,False,False]
PD_Data_tranformation.loc["ack_B"] =                                   ["NA",False,False,False,False]
PD_Data_tranformation.loc["bytes"] =                                   ["NA",False,True,3,False]
PD_Data_tranformation.loc["ssl_dom_version"] =                         ["NA",False,False,False,True]
PD_Data_tranformation.loc["push"] =                                    ["NA",False,False,False,True]
PD_Data_tranformation.loc["push_A"] =                                  ["NA",False,False,False,True]
PD_Data_tranformation.loc["push_B"] =                                  ["NA",False,False,False,True]
PD_Data_tranformation.loc["ssl_count_client_cipher_algs"] =            ["NA",False,False,False,True]
PD_Data_tranformation.loc["ssl_count_client_ciphersuites"] =           ["NA",False,False,False,True]
PD_Data_tranformation.loc["ssl_count_client_compressions"] =           ["NA",False,False,False,True]
PD_Data_tranformation.loc["http_dom_resp_code"] =                       ["NA",False,False,False,True]
PD_Data_tranformation.loc["ssl_count_server_compression"] =            ["NA",False,False,False,True]

# print(PD_Data_tranformation)


def DataImport_and_preProdceesing(DF,PD_Data_tranformation):
    # print('the value counts of the target are:')
    # print(df_trn.iloc[:, -1].value_counts())

    # print(DF.shape)  # dimensions (rows, columns)
    # display(DF.head())  # overview of first (last) rows
    display(DF.describe().T)  # basic stats per variable

    # DF = Preprocess.PreRemvoe_Low_STD(DF)

    cols_to_bin_sizes = {"bytes": 3}
    # print(DF.columns)

    Cols_to_remvoe = list(PD_Data_tranformation[PD_Data_tranformation["Remove"]==True].index)
    cols_to_dummies = list(PD_Data_tranformation[PD_Data_tranformation["Get_Dummies"]==True].index)

    cols_to_bin = list(PD_Data_tranformation[PD_Data_tranformation["To_Bin"]==True].index)

    for col in cols_to_bin:
        cols_to_bin_sizes["bytes"]=PD_Data_tranformation.loc[col]["BinSize"]

    Dummies_list=[]
    Dummies_dict=[]
    # DF,cols_to_bin_naming =     Preprocess.Df_col_binnining(DF,cols_to_bin,cols_to_bin_sizes)
    # DF, Dummies_list,Dummies_dict=           Preprocess.DF_Get_Dummies(DF,cols_to_dummies)
    DF =                                     Preprocess.fill_nans(DF)
    DF =                                     Preprocess.drop_features(DF, Cols_to_remvoe)
    DF, skewed_kurt_features=                Preprocess.data_preprocess_skewed(DF, Ignore_cols=Dummies_list)
    DF,droped_cols=                          Preprocess.DF_Drop_X_values(DF, 1)



    # DF =                      Preprocess.stadardize_features(DF)



    # print(skewed_features)
    # print(DF.columns)

    # print(droped_cols)



    tupl_lists=(droped_cols,cols_to_dummies,skewed_kurt_features,Dummies_list,Dummies_dict)
    # return DF (droped_cols,cols_to_dummies)
    return DF,tupl_lists


def Test_Data_Train(Test_DF,Tuple_lists,X_train):
    (droped_cols, cols_to_dummies, skewed_kurt_features,Dummies_list,Dummies_dict) = Tuple_lists

    skewed_kurt_features=list(skewed_kurt_features)
    Test_DF[skewed_kurt_features]=           np.log1p(Test_DF[skewed_kurt_features])
    Test_DF =                        Preprocess.drop_features(Test_DF, droped_cols)
    # Test_DF=                        Preprocess.DF_Get_Dummies_Train(Test_DF,cols_to_dummies,X_train)
    # Test_DF=                        Preprocess.DF_Get_Dummies_Train_B(Test_DF,cols_to_dummies,X_train,Dummies_dict)
    # Test_DF =                        Preprocess.fill_dummies_cols(Test_DF,Dummies_list)

    return Test_DF



X_train, X_test, y_train, y_test = train_test_split(
    X_trn, y_trn, test_size=0.33, random_state=42)

X_train_origial = X_train.copy()

X_train , tuple1 = DataImport_and_preProdceesing(X_train , PD_Data_tranformation)
X_test  =      Test_Data_Train(X_test,tuple1,X_train_origial)

Print_DF_to_file(X_train)
Print_DF_to_file(X_test)

# # print (df_trn.dtypes)
# x = df_trn.columns.to_series().groupby(df_trn.dtypes).groups
# print(x)

# writer = pd.ExcelWriter('output.xlsx')
# df1 = pd.DataFrame(df_trn.describe().T)
# df1.to_excel(writer,'Sheet1')
# # df2.to_excel(writer,'Sheet2')
# writer.save()

  # basic stats per variable
# df_trn = Preprocess.fill_nans(df_trn)
# print_unique(df_trn)

