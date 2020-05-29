import numpy as np
import sys
sys.path.append('../../')
from Decision_Trees.build_tree import *
from Regression_models.regression import *
from Regression_models.lasso_regression import *

class AdaBoost:
    def __init__(self, base_learner='dt', n_estimators=10):
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.estimators= {}

    def fit(self, X, Y):
        self.w = np.zeros((X.shape[0]))
        self.error = np.zeros((self.n_estimators))
        for i in range(X.shape[0]):
            self.w[i] = (1/X.shape[0])
        for i in range(self.n_estimators):
            if self.base_learner == 'dt':
                dt = Decision_tree(max_depth=1, split_criterion='Entropy', node_eval='mode')        ## decision stumps
                dt.fit(X, Y, sample_weight=self.w)
                y = dt.predict(X, dt.tree['root'])
                e = 0
                for j in range(X.shape[0]):
                    if y[j]!=Y.loc[i]:
                        e+=self.w[i]
                self.error[i] = 0.5*np.log((1-e)/e)
                for j in range(X.shape[0]):
                    if y[j]!=Y.loc[i]:
                        self.w[i] = self.w[i]*np.exp(-self.error[i]*y[i]*Y.loc[i])
                self.w/=np.sum(self.w)      ## renormalizing the weights
