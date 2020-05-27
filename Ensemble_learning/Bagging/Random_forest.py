import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import random
from scipy import stats
from Decision_Trees.build_tree import *
from MLP.MLP import *
from Regression_models.regression import *
from Regression_models.lasso_regression import *

class Random_forest:
    def __init__(self, max_feature ='sqrt', n_estimators = 20, criterion='Entropy', max_depth=5, random_seed = 0, node_eval='mean'):
        self.max_f = max_feature
        self.depth = max_depth
        self.n_predictors = n_estimators
        self.predictors = {}
        self.crit = criterion
        self.eval = node_eval

    def fit(self, X, Y):
        self.predictors['forest'] = []
        cols = list(X.columns)
        ind = np.asarray([x for x in range(X.shape[0])])
        n=0
        if self.max_f == 'sqrt':
            n = int(np.ceil(np.power(len(cols), 0.5)))
        for i in range(self.n_predictors):
            dt = Decision_tree(max_depth=self.depth, split_criterion=self.crit, node_eval=self.eval)
            x_ind = np.random.choice(ind, X.shape[0])
            features = random.sample(cols, n)
            dt.fit(X[features].loc[x_ind], Y.loc[x_ind])
            self.predictors['forest'].append(dt)

    def predict(self, test, forest):
        predictions = []
        for i in range(len(forest)):
            dt = forest[i]
            pred = dt.predict(test, dt.tree)
            predictions.append(pred)
        final_predictions = []
        predictions = np.asarray(predictions)
        for i in range(len(predictions[0])):
            if self.eval == 'mode':
                final_predictions.append(stats.mode(predictions[:,i])[0][0])
            elif self.eval == 'mean':
                final_predictions.append(np.mean(predictions[:,i]))
        return np.asarray(final_predictions)


