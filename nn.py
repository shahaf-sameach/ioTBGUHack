from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

import pandas as pd
import pdb

class Network(object):

  def __init__(self):
    model = Sequential()
    model.add(Dense(100, input_dim=297, activation='relu',))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(11, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    self.model = model

  def train(self, x, y, epochs=60, batch_size=100, weight_file="weights.hdf5", nb_sentences=None):
    # Callback for model saving:
    checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1)

    # Training
    self.model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2)
 
  def load_weights(self, weight_file):
    self.model.load_weights(weight_file)

  def predict(self, x):
    return self.model.predict(x, verbose=0)


if __name__ == '__main__':
  net = Network()
  df_trn = pd.read_csv('hackathon_IoT_training_set_based_on_01mar2017.csv', low_memory=False, na_values='?')
  x_trn = df_trn.iloc[:,0:(df_trn.shape[1]-1)].copy() # all but the last column
  x_trn.fillna(value=0, inplace=True)

  y_trn = df_trn.iloc[:,(df_trn.shape[1]-1)].copy() # last column
  y_trn = pd.get_dummies(y_trn)
  net.train(x_trn,y_trn)


