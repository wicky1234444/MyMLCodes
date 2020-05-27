import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from Decision_Trees.build_tree import *
from MLP.MLP import *
from Regression_models. regression import *
from Regression_models.lasso_regression import *

def Booststrap(X, subsets = 10):
    ind = np.asarray([x for x in range(X.shape[0])])
    n_subsets = []
    for i in range(subsets):
        n_subsets.append(np.random.choice(ind, X.shape[0]))
    return n_subsets

def Bagging(algorithm = ['dt'], X=[], Y=[], subsets=10):
    bagged_alg = {}
    if len(algorithm)==1:
        datasets = Booststrap(X, subsets)
        bagged_alg['type'] = 'homogenious'
        bagged_alg['algorithm'] = 'decision tree'
        bagged_alg['predictors'] = []
        for i in range(subsets):
            if algorithm[0]=='dt':
                dt = Decision_tree()
                dt.fit(X.loc[datasets[i]], Y.loc[datasets[i]])
                bagged_alg['predictors'].append(dt)
    return bagged_alg

def predict(Ensembling = 'avg', test=[], bagging_dict={}):
    bagged_predictions = []
    all_predictions = np.zeros((test.shape[0]))
    if bagging_dict['type'] == 'homogenious':
        if bagging_dict['algorithm'] == 'decision tree':
            for i in range(len(bagging_dict['predictors'])):
                dt = bagging_dict['predictors'][i]
                predictions = []
                for j in test.index:
                    predictions.append(dt.predict(test.loc[j], dt.tree))
                bagged_predictions.append(np.asarray(predictions))
    
    if Ensembling=='avg':
        for i in range(len(bagged_predictions)):
            all_predictions = np.add(all_predictions, bagged_predictions[i])
        all_predictions/= len(bagged_predictions)

    return all_predictions

