
# coding: utf-8

# In[4]:


import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import VotingClassifier

from datetime import datetime
from datetime import timedelta


pd.set_option('display.max_columns', None)  # display all columns
# data = pd.read_csv("data.csv")
data = pd.read_csv("C:\\Users\\Rom\\Desktop\\data.csv")

data.sort_values('game_date', inplace=True)

################added###########
data["shot_id_sub"] = range(1, len(data) + 1)
################################


data.set_index('shot_id_sub', inplace=True)

data["action_type"] = data["action_type"].astype('object')
data["combined_shot_type"] = data["combined_shot_type"].astype('category')
data["game_event_id"] = data["game_event_id"].astype('category')
data["game_id"] = data["game_id"].astype('category')
data["period"] = data["period"].astype('object')
#data["playoffs"] = data["playoffs"].astype('category')
data["season"] = data["season"].astype('category')
data["shot_made_flag"] = data["shot_made_flag"].astype('category')
data["shot_type"] = data["shot_type"].astype('category')
data["team_id"] = data["team_id"].astype('category')

data.describe(include=['object', 'category'])

##############################dealing with the data###########################################
unknown_mask = data['shot_made_flag'].isnull()
data_cl = data.copy()  # create a copy of data frame
target = data_cl['shot_made_flag'].copy()

data_cl.drop('team_id', axis=1, inplace=True)  # Always one number
data_cl.drop('lat', axis=1, inplace=True)  # Correlated with loc_x
data_cl.drop('lon', axis=1, inplace=True)  # Correlated with loc_y
data_cl.drop('game_id', axis=1, inplace=True)  # Independent
data_cl.drop('game_event_id', axis=1, inplace=True)  # Independent
data_cl.drop('team_name', axis=1, inplace=True)  # Always LA Lakers
data_cl.drop('shot_made_flag', axis=1, inplace=True)
data_cl.drop('playoffs', axis=1, inplace=True)

############
## Remaining time
data_cl['seconds_from_period_end'] = 60 * data_cl['minutes_remaining'] + data_cl['seconds_remaining']
data_cl['last_5_sec_in_period'] = data_cl['seconds_from_period_end'] < 5
data_cl['last_5_sec_in_period']=data_cl['last_5_sec_in_period'].astype('int')

data_cl['secondsFromPeriodStart'] = 60*(11-data_cl['minutes_remaining'])+(60-data_cl['seconds_remaining'])
data_cl['secondsFromGameStart']   = (data['period'] <= 4).astype(int)*(data_cl['period']-1)*12*60 + (data_cl['period'] > 4).astype(int)*((data_cl['period']-4)*5*60 + 3*12*60) + data_cl['secondsFromPeriodStart']

data_cl.drop('minutes_remaining', axis=1, inplace=True)
data_cl.drop('seconds_remaining', axis=1, inplace=True)
data_cl.drop('seconds_from_period_end', axis=1, inplace=True)

## Matchup - (away/home)
data_cl['home_play'] = data_cl['matchup'].str.contains('vs').astype('int')
data_cl.drop('matchup', axis=1, inplace=True)
# Game date
x =  data_cl.game_date.str.replace('-','')
data_cl['yesterday'] = (pd.to_datetime(x, format='%Y%m%d')- timedelta(days=1)) 
data_cl['back2back'] = data_cl.yesterday.isin(data_cl.game_date).astype('int')
data_cl.drop('yesterday', axis=1, inplace=True) 

data_cl['game_date'] = pd.to_datetime(data_cl['game_date'])
data_cl['game_year'] = data_cl['game_date'].dt.year
data_cl['game_month'] = data_cl['game_date'].dt.month
#data_cl['dayOfWeek']=data_cl['game_date'].dt.dayofweek

data_cl.drop('game_date', axis=1, inplace=True)

data_cl['3pt'] = data_cl['shot_type'].str.contains('3PT').astype('int')
data_cl.drop('shot_type', axis=1, inplace=True)

# Replace 20 least common action types with value 'Other'
rare_action_types = data_cl['action_type'].value_counts().sort_values().index.values[:20]
data_cl.loc[data_cl['action_type'].isin(rare_action_types), 'action_type'] = 'Other'

data_cl.drop('season', axis=1, inplace=True)
data_cl.drop('shot_distance', axis=1, inplace=True)
#data_cl.drop('loc_y', axis=1, inplace=True)
data_cl.drop('opponent', axis=1, inplace=True)

# ENCODE CATEGORIAL VARIABLES
categorial_cols = [
    'action_type', 'combined_shot_type', 'period',
    'shot_zone_area', 'shot_zone_basic', 'shot_zone_range',
    'game_month','game_year']
# 'loc_x', 'loc_y' ,'opponent' 

for cc in categorial_cols:
    dummies = pd.get_dummies(data_cl[cc])
    dummies = dummies.add_prefix("{}#".format(cc))
    data_cl.drop(cc, axis=1, inplace=True)
    data_cl = data_cl.join(dummies)

# taking down features
data_submit = data_cl[unknown_mask]
##################creating the model##################
X = data_cl[~unknown_mask]
Y = target[~unknown_mask]
seed = 7
processors=1
num_folds=3
num_instances=len(X)
scoring='log_loss'

kfold = KFold(n=num_instances, n_folds=num_folds, random_state=seed)

             
lr_grid = GridSearchCV(estimator = LogisticRegression(random_state=seed),param_grid = {'penalty': [ 'l2'],'solver':['newton-cg', 'lbfgs', 'sag', 'saga']
    }, cv = kfold,scoring = scoring, n_jobs = processors)
lr_grid.fit(X, Y)
#print(lr_grid.best_score_)
#print(lr_grid.best_params_)
 #Create sub models
estimators = []

estimators.append(('lr', LogisticRegression(solver='newton-cg',penalty='l2', C=1)))
estimators.append(('dtc', DecisionTreeClassifier( max_depth=4, random_state=seed)))
estimators.append(('lda', LinearDiscriminantAnalysis( solver='svd', n_components=None)))

# create the ensemble model
ensemble = VotingClassifier(estimators, voting='soft', weights=[6,3,1])

results = cross_val_score(ensemble, X, Y, cv=kfold, scoring=scoring,n_jobs=processors)
#print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

############for submissiom##############
submissionFinal = pd.DataFrame()

# the for should run from here and on
counter = 0;
for index, row in data_submit.iterrows():
    counter += 1
    if (len(range(index)) > 100):
      #  if (len(range(index)) > 5000):
       #     X = data_cl[(index-2000):index]
        #    Y = target[(index-2000):index]
        #else:
         #   X = data_cl[:index]
          #  Y = target[:index]
        X = data_cl[:index]
        Y = target[:index]
        # take the known data
        X = X[~unknown_mask]
        Y = Y[~unknown_mask]
        
        ensemble.fit(X, Y)
        preds =  ensemble.predict_proba(data_cl[(index-1):index])
        ##tunning
        #lr=LogisticRegression()
        #lr.fit(X, Y)
        #preds =  lr.predict_proba(data_cl[(index-1):index])
        
        #lda=LinearDiscriminantAnalysis()
        #lda.fit(X, Y)
        #preds1 =  lda.predict_proba(data_cl[(index-1):index])
        
        #dt=DecisionTreeClassifier(max_depth=4,random_state=0)
        #dt.fit(X, Y)
        #preds2 =dt.predict_proba(data_cl[(index-1):index])
        
        #tmp=0.7*preds+0.2*preds1+0.1*preds;
        
        submission = pd.DataFrame()
        submission["shot_id"] = data_submit[(counter - 1):counter]["shot_id"]

        submission["shot_made_flag"] = preds[:, 1]
        frames = [submissionFinal, submission]
        result = pd.concat(frames)
        submissionFinal = result
    else:
        submission = pd.DataFrame()
        submission["shot_id"] = data_submit[(counter - 1):counter]["shot_id"]

        submission["shot_made_flag"] = 0.446161#kobe pg%
        frames = [submissionFinal, submission]
        result = pd.concat(frames)
        submissionFinal = result

    print("the counter is : {} ".format(counter))
#  print("the shot id is {}".format( data_submit[(counter-1):counter]["shot_id"]))

result.to_csv("res_21.csv", index=False)


# In[5]:


result.to_csv("last_run.csv", index=False)


# # code with the methodes

# # keras

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense

test_AF=splitData(test_clean,'A','SA','test');

trainX_A, trainY_A = train_AF[col], train_AF.y
testX_A, testY_A = test_AF[col], test_AF.y

model_A = Sequential()
model_A.add(Dense(100, input_dim=12, activation='relu',))
model_A.add(Dense(30))
model_A.add(Dense(80, input_dim=30, activation='relu'))
model_A.add(Dense(15))
model_A.add(Dense(50, input_dim=15, activation='relu'))
model_A.add(Dense(10))
model_A.add(Dense(30, input_dim=10, activation='relu'))
model_A.add(Dense(5))
model_A.add(Dense(10, input_dim=5, activation='relu'))
model_A.add(Dense(1))

model_A.compile(loss='mean_absolute_error', optimizer='adam')
model_A.fit(trainX_A, trainY_A, epochs=300, batch_size=2, verbose=2)

testPredict_A = model_A.predict(testX_A)

