import numpy as np
from time_wrapper import timewrapper
import pickle
import pandas as pd
import pdb

@timewrapper
def load_data(file_name):
  df = pd.read_csv(file_name, sep=',')
  return df


file_name = 'hackathon_IoT_training_set_based_on_01mar2017.csv'
df = load_data(file_name)
pdb.set_trace()



