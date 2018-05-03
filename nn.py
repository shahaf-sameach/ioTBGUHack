from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint

class Network(object):

  def __init__(self):
    model = Sequential()
    model.add(Dense(100, input_dim=297, activation='relu',))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

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
  a = Network()

