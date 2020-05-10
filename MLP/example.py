import numpy as np
import pandas as pd

dataset = pd.read_csv('circle_data.csv')
del dataset['Unnamed: 0']
from sklearn.utils import shuffle
dataset = shuffle(dataset)
dataset = dataset.reset_index()
del dataset['index']
train = dataset[0:320]
test = dataset[320:]

