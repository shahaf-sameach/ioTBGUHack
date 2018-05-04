import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler

def Remvoe_Low_STD(df):
    threshold = 0.2
    df.loc[:, df.std() > threshold]
    # df.drop(df.std()[df.std() < threshold].index.values, axis=1)
    return df


def custom_features_NA_conversions(dataset):

    # # NA in all garage features means no basement exist
    # # 'GarageType', 'GarageFinish', 'GarageQual'
    for col in (['GarageType', 'GarageQual']):
        dataset[col] = dataset[col].fillna('NoGarage')# TODO - do only for one to not create nultipile variable
    dataset['GarageCars'] = dataset['GarageCars'].fillna(0)
    dataset['GarageArea'] = dataset['GarageArea'].fillna(0)

    dataset['BsmtQual'] = dataset['BsmtQual'].fillna('NoBsmt')
    dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(0)  # TODO - doesn't make change
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)
    dataset['BsmtFullBath'] = dataset['BsmtFullBath'].fillna(0)


    # MasVnrType None means no MasVnrType exist
    dataset["MasVnrType"] = dataset["MasVnrType"].replace(to_replace="None", value='NoMasVnr') # TODO - doesn't make change

    # FireplaceQu None means no FireplaceQu exist
    dataset["FireplaceQu"] = dataset["FireplaceQu"].fillna('NoFire')  # TODO - make conteverese little diffeence

    # PooLQC None means no PooLQC exist
    dataset["PoolQC"] = dataset["PoolQC"].fillna('NoPool')  # TODO - make conteverese little diffeence

    # # Alley NA means no alley exist # TODO - making worse
    dataset['Alley'] = dataset['Alley'].fillna('NOACCESS')

    # MSZoning real NA - fill with most popular values # TODO - doesn't make change
    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])

    # # Electrical NA  - fill with most popular values # TODO - make conteverese little diffeence
    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])

    # KitchenQual NA - fill with most popular values
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode()[0])

    # SaleType NA - fill with most popular values  TODO - doesn't make change
    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode()[0])

    # dataset["Fence"] = dataset["Fence"].fillna('NoFence')

    # dataset["MiscFeature"] = dataset["MiscFeature"].replace(to_replace="Othr", value="NA")
    # dataset["SaleType"] = dataset["SaleType"].replace(to_replace="Oth", value="NA")

    # functional NA - fill with most popular values
    # dataset['Functional'] = dataset['Functional'].fillna(dataset['Functional'].mode()[0])


    return dataset


def ordinal_variables_string_to_numbers(dataset):
    #TODO - decicde about "Utilities" - yes or not, "HouseStyle" - split last from all and convert?, "SaleType" - convert?
    # TODO - check again features when NA - "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageFinish", "GarageQual", "GarageCond", "PoolQC"
    ordinal_string_list = ["LotShape", "LandSlope", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure",
                           "BsmtFinType1", "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu",
                           "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence"]

    # # LotShape -
    dataset["LotShape"] = dataset["LotShape"].replace(to_replace="Reg", value=0)
    dataset["LotShape"] = dataset["LotShape"].replace(to_replace="IR1", value=1)
    dataset["LotShape"] = dataset["LotShape"].replace(to_replace="IR2", value=2)
    dataset["LotShape"] = dataset["LotShape"].replace(to_replace="IR3", value=3)

    # # LandSlope TODO - make worse
    dataset["LandSlope"] = dataset["LandSlope"].replace(to_replace="Gtl", value=0)
    dataset["LandSlope"] = dataset["LandSlope"].replace(to_replace="Mod", value=1)
    dataset["LandSlope"] = dataset["LandSlope"].replace(to_replace="Sev", value=2)

    # HeatingQC
    dataset[["HeatingQC"]] = dataset[["HeatingQC"]].replace(to_replace="Ex", value=0)
    dataset[["HeatingQC"]] = dataset[["HeatingQC"]].replace(to_replace="Gd", value=1)
    dataset[["HeatingQC"]] = dataset[["HeatingQC"]].replace(to_replace="TA", value=2)
    dataset[["HeatingQC"]] = dataset[["HeatingQC"]].replace(to_replace="Fa", value=3)
    dataset[["HeatingQC"]] = dataset[["HeatingQC"]].replace(to_replace="Po", value=4)

    # No Garage treatment - TODO - make worse
    # newcol = dataset["GarageQual"]
    # dataset = dataset.assign(**{'NoGrg': newcol})
    # dataset['NoGrg'] = dataset['NoGrg'].fillna(0)
    # dataset['NoGrg'] = dataset['NoGrg'].replace(to_replace="Ex", value=1)
    # dataset['NoGrg'] = dataset['NoGrg'].replace(to_replace="Gd", value=1)
    # dataset['NoGrg'] = dataset['NoGrg'].replace(to_replace="TA", value=1)
    # dataset['NoGrg'] = dataset['NoGrg'].replace(to_replace="Fa", value=1)
    # dataset['NoGrg'] = dataset['NoGrg'].replace(to_replace="Po", value=1)

    # "GarageQual", "GarageCond"
    dataset[["GarageQual", "GarageCond"]] = dataset[["GarageQual", "GarageCond"]].replace(to_replace="Ex", value=0)
    dataset[["GarageQual", "GarageCond"]] = dataset[["GarageQual", "GarageCond"]].replace(to_replace="Gd", value=1)
    dataset[["GarageQual", "GarageCond"]] = dataset[["GarageQual", "GarageCond"]].replace(to_replace="TA", value=2)
    dataset[["GarageQual", "GarageCond"]] = dataset[["GarageQual", "GarageCond"]].replace(to_replace="Fa", value=3)
    dataset[["GarageQual", "GarageCond"]] = dataset[["GarageQual", "GarageCond"]].replace(to_replace="Po", value=4)
    dataset[["GarageQual", "GarageCond"]] = dataset[["GarageQual", "GarageCond"]].replace(to_replace='NoGrg', value=5)
    dataset[["GarageQual", "GarageCond"]] = dataset[["GarageQual", "GarageCond"]].replace(to_replace='NA', value=5)

    # Functional
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Typ", value=0)
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Min1", value=1)
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Min2", value=2)
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Mod", value=3)
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Maj1", value=4)
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Maj2", value=5)
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Sev", value=6)
    dataset["Functional"] = dataset["Functional"].replace(to_replace="Sal", value=7)

    # # "GarageFinish" TODO - NA is number here but works better
    dataset["GarageFinish"] = dataset["GarageFinish"].replace(to_replace="Fin", value=0)
    dataset["GarageFinish"] = dataset["GarageFinish"].replace(to_replace="RFn", value=1)
    dataset["GarageFinish"] = dataset["GarageFinish"].replace(to_replace="Unf", value=2)
    dataset["GarageFinish"] = dataset["GarageFinish"].replace(to_replace="NA", value=3)
    dataset["GarageFinish"] = dataset["GarageFinish"].replace(to_replace="NoGrg", value=3)


    # # "PavedDrive" # TODO - worse
    dataset["PavedDrive"] = dataset["PavedDrive"].replace(to_replace="Y", value=0)
    dataset["PavedDrive"] = dataset["PavedDrive"].replace(to_replace="P", value=1)
    dataset["PavedDrive"] = dataset["PavedDrive"].replace(to_replace="N", value=2)

    # # "PoolQC" - # TODO - good for lasso, bad for ridge
    dataset["PoolQC"] = dataset["PoolQC"].replace(to_replace="Ex", value=0)
    dataset["PoolQC"] = dataset["PoolQC"].replace(to_replace="Gd", value=1)
    dataset["PoolQC"] = dataset["PoolQC"].replace(to_replace="TA", value=2)
    dataset["PoolQC"] = dataset["PoolQC"].replace(to_replace="Fa", value=3)
    dataset["PoolQC"] = dataset["PoolQC"].replace(to_replace="NoPool", value=4)
    dataset["PoolQC"] = dataset["PoolQC"].replace(to_replace="NA", value=4)

    # "Fence" - TODO - NA is number here but works better
    dataset["Fence"] = dataset["Fence"].replace(to_replace="GdPrv", value=0)
    dataset["Fence"] = dataset["Fence"].replace(to_replace="MnPrv", value=1)
    dataset["Fence"] = dataset["Fence"].replace(to_replace="GdWo", value=2)
    dataset["Fence"] = dataset["Fence"].replace(to_replace="MnWw", value=3)
    dataset["Fence"] = dataset["Fence"].replace(to_replace="NA", value=4)
    dataset["Fence"] = dataset["Fence"].replace(to_replace="NoFence", value=4)

    return dataset


def categorical_variables_numeric_to_cat(dataset):
    # newcol = dataset["MSSubClass"].apply(str)
    # dataset = dataset.assign(MSSubClassStr=newcol)

    # newcol = dataset["MoSold"].apply(str)
    # dataset = dataset.assign(MoSoldStr=newcol)

    # dataset["MSSubClass"] = dataset["MSSubClass"].apply(str)
    # dataset['MoSold'] = dataset['MoSold'].astype(str)
    return dataset

def categorical_variable_str_to_binary(dataset):
    categorical_binary_list = ["CentralAir"]
    dataset["CentralAir"] = dataset["CentralAir"].replace(to_replace="N", value=0)
    dataset["CentralAir"] = dataset["CentralAir"].replace(to_replace="Y", value=1)
    return dataset

def custom_features_manipulations(dataset):
    # TODO - check with graphs if really behave not lineary


    # add all inner area square feet
    dataset['TotalSfIn'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF'] +\
                           dataset['EnclosedPorch'] + dataset['3SsnPorch']
    dataset.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'EnclosedPorch', '3SsnPorch'], axis=1, inplace=True)

    # add all outer area square feet
    # dataset['TotalSfOut'] = dataset['WoodDeckSF'] + dataset['PoolArea']
    # dataset.drop(['WoodDeckSF', 'PoolArea'], axis=1, inplace=True)

    # add polynomial features for most corelated ones
    best_arr = ["OverallQual", "GrLivArea", "GarageCars", "GarageArea", "TotalSfIn", "FullBath", "TotRmsAbvGrd", "YearBuilt", "YearRemodAdd"]
    # ordinal_non_linear_behaviour_list = best_arr
    # for feature in ordinal_non_linear_behaviour_list:
    #     newcol = np.power(dataset[feature], 2)
    #     dataset = dataset.assign(**{'Pow2' + feature: newcol})
    #     newcol = np.power(dataset[feature], 3)
    #     dataset = dataset.assign(**{'Pow3'+feature: newcol})
    #     newcol = np.power(dataset[feature], 4)
    #     dataset = dataset.assign(**{'Pow4' + feature: newcol})

    return dataset

def bin_year(dataset):
    # bin years
    # dataset['YearRemodAdd'] = pd.cut(dataset['YearRemodAdd'], 7, labels=np.linspace(0, 6, 7)) # TODO - good for lasso, bad for ridge. monor difference
    dataset['YearBuilt'] = pd.cut(dataset['YearBuilt'], 7, labels=np.linspace(0, 6, 7))
    # dataset['GarageYrBlt'] = pd.cut(dataset['GarageYrBlt'], 5, labels=np.linspace(0, 4, 5))
    # dataset['MoSold'] = pd.cut(dataset['MoSold'], 4, labels=np.linspace(0, 3, 4))
    # 'MoSold'

    return dataset

def drop_features(dataset, Cols_to_remvoe):
    dataset.drop(Cols_to_remvoe, axis=1, inplace=True)
    return dataset



def data_preprocess_skewed(dataset,Ignore_cols,fill_na=False,skew_thershold=0.75,kurt_thershold=0.3):

    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
    skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    # skewed_feats = skewed_feats[np.abs(skewed_feats) > 0.75]
    skewed_feats = skewed_feats[skewed_feats > skew_thershold]
    skewed_feats = skewed_feats.index
    # skewed_feats = skewed_feats.symmetric_difference(Ignore_cols)
    skewed_feats = [x for x in skewed_feats if x not in Ignore_cols]
    skewed_feats = [x for x in skewed_feats if len(dataset[x].unique())>3]
    dataset[skewed_feats] = np.log1p(dataset[skewed_feats])  # TODO - change to only train data

    kurt_feat = dataset[numeric_feats].apply(lambda x: kurtosis(x.dropna()))  # compute skewness
    # kurt_feat = kurt_feat[np.abs(kurt_feat) > 0.3]
    kurt_feat = kurt_feat[kurt_feat > kurt_thershold]
    kurt_feat = kurt_feat.index
    kurt_feat = [x for x in kurt_feat if x not in Ignore_cols]
    kurt_feat = [x for x in kurt_feat if len(dataset[x].unique())>3]
    dataset[kurt_feat] = np.log1p(dataset[kurt_feat])  # TODO - change to only train data

    both = set(skewed_feats+kurt_feat)
    return dataset, both  #, kurt_feat




def fill_nans(dataset):
    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
    for feature in numeric_feats:
        if feature != "Id":
            # if len(dataset[feature].unique())==2:
            #     dataset[feature] = dataset[feature].fillna(0)
            # else:
                dataset[feature] = dataset[feature].fillna(dataset[feature].median())
    return dataset

def stadardize_features(dataset):
    numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
    for feature in numeric_feats:
        if feature != "Id":
            scaled_feature = StandardScaler().fit_transform(dataset[feature].reshape(-1, 1))
            dataset[feature] = scaled_feature
    return dataset


def Df_col_binnining(DF,cols_to_bin,Bin_sizes):

    cols_to_bin_naming=[]

    for col in cols_to_bin:
        bin_size=Bin_sizes[col]
        bins = range(bin_size)
        bins=list(bins)

        bins=np.linspace(1, bin_size, bin_size)
        name=col+"_binning"
        cols_to_bin_naming.append(name)
        DF[name] = pd.qcut(DF[col],q=bin_size,labels=list(range(1,bin_size+1)))
    DF=drop_features(DF,cols_to_bin)

    return DF, cols_to_bin_naming

def DF_Drop_X_values(DF,Number_Of_values=1):
    droped_cols=[]
    for col in DF.columns:
        if len(DF[col].unique()) == Number_Of_values:
            DF.drop(col, inplace=True, axis=1)
            droped_cols.append(col)
    # print(droped_cols)
    return DF, droped_cols

def DF_Get_Dummies(DF,cols_to_dummies):
    Dummies_dict={}
    Original_cols = DF.columns
    DF = pd.get_dummies(DF, columns=cols_to_dummies,dummy_na=False)
    new_cols = DF.columns
    dummies_cols = set(new_cols).symmetric_difference(set(Original_cols))
    dummies_cols = dummies_cols.symmetric_difference(set(cols_to_dummies))

    return DF ,dummies_cols

def DF_Get_Dummies(DF,cols_to_dummies):
    Dummies_dict={}
    Original_cols = DF.columns


    for col in cols_to_dummies:
        category_list=list(DF[col].unique())
        DF[col].astype('category',categories=category_list)
        Dummies_dict[col]=category_list
    DF = pd.get_dummies(DF, columns=cols_to_dummies,dummy_na=False)

    new_cols = DF.columns
    dummies_cols = set(new_cols).symmetric_difference(set(Original_cols))
    dummies_cols = dummies_cols.symmetric_difference(set(cols_to_dummies))

    return DF ,dummies_cols,Dummies_dict



def DF_Get_Dummies_Train(DF_Test,cols_to_dummies,DF_Train):
    train = DF_Train.copy()
    test = DF_Test

    train_objs_num = len(train)
    dataset = pd.concat(objs=[train, test], axis=0)
    dataset_preprocessed = pd.get_dummies(dataset, columns=cols_to_dummies,dummy_na=False)
    train_preprocessed = dataset_preprocessed[:train_objs_num]
    test_preprocessed = dataset_preprocessed[train_objs_num:]

    return test_preprocessed

def DF_Get_Dummies_Train_B(DF_Test,cols_to_dummies,DF_Train,Dummies_dict):
    for key in Dummies_dict:
        DF_Test[key].astype('category',categories=Dummies_dict[key])
    dataset_preprocessed = pd.get_dummies(DF_Test, columns=cols_to_dummies,dummy_na=False)
    return dataset_preprocessed


def fill_dummies_cols(DF,dummies_cols):
    cols_to_add = [col for col in dummies_cols if col  not in DF.columns]

    # new_cols = [ col for col in DF.columns if col not in dummies_cols]
    assert(len(cols_to_add)==0)
    print(cols_to_add)
    return DF