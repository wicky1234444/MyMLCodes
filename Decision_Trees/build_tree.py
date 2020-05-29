import sys
sys.path.append('../')
from Decision_Trees.tree_splitting_criterion import *
import numpy as np
import pandas as pd
from scipy import stats

class Decision_tree:
    def __init__(self, max_depth=10, split_criterion = 'Entropy', node_eval='mean'):
        self.max_depth = max_depth
        self.tree = {}
        self.node_eval = node_eval
        self.criterion = split_criterion

    def find_best_split(self, X, col, Y, sample_weight=[]):
        if self.criterion == 'Entropy' or self.criterion == 'Gini':
            criterion_val = 10
        elif self.criterion == 'IG' or self.criterion=='Chi':
            criterion_val = -1
        split_val = -1
        for val in set(X[col]):
            #print(val)
            y_pred = X[col]<val
            if self.criterion == 'Entropy' or self.criterion == 'Gini':
                entropy = Entropy(Y.to_numpy(), y_pred.to_numpy(), sample_weight=sample_weight)
                if(entropy<=criterion_val):
                    criterion_val=entropy
                    split_val = val
            elif self.criterion == 'IG' or self.criterion=='Chi':
                ig = Information_Gain(Y.to_numpy(), y_pred.to_numpy(), sample_weight=sample_weight)
                if(ig>=criterion_val):
                    criterion_val = ig
                    split_val = val
        return [criterion_val, split_val]

    def best_column_to_split(self, X, Y, sample_weight=[]):
        if self.criterion == 'Entropy' or self.criterion == 'Gini':
            criterion_val = 10
        elif self.criterion == 'IG' or self.criterion=='Chi':
            criterion_val = -1
        split_val = -1
        split_col = ""
        for col in list(X.columns)[:-1]:
            if self.criterion == 'Entropy' or self.criterion == 'Gini':
                entropy, val = self.find_best_split(X, col, Y, sample_weight=sample_weight)
                if entropy==0:
                    return [entropy, val, col]
                elif(entropy<=criterion_val):
                    criterion_val = entropy
                    split_val = val
                    split_col = col
            elif self.criterion == 'IG' or self.criterion=='Chi':
                ig, val = self.find_best_split(X, col, Y, sample_weight=sample_weight)
                if ig==1:
                    return [ig, val, col]
                elif(ig>=criterion_val):
                    criterion_val = ig
                    split_val = val
                    split_col = col
        return [criterion_val, split_val, split_col]

    def build_tree(self, X, Y, depth, node = {}, sample_weight = []):
        if node==None:
            return None
        elif len(Y)==0:
            return None
        elif len(np.unique(Y))==1:
            return {'val':Y.to_numpy()[0]}
        elif depth>=self.max_depth:
            return None
        else:
            entropy, cutoff, col = self.best_column_to_split(X, Y, sample_weight=sample_weight)
            y_left = Y[X[col]<cutoff]
            y_right = Y[X[col]>=cutoff]
            if self.node_eval== 'mean':
                node = {'col': col, 'cutoff':cutoff, 'val':np.mean(Y)}
            elif self.node_eval == 'mode':
                node = {'col': col, 'cutoff':cutoff, 'val':stats.mode(Y)[0][0]}
            node['left'] = self.build_tree(X[X[col]<cutoff], y_left, depth+1, {})
            node['right'] = self.build_tree(X[X[col]>=cutoff], y_right, depth+1, {})
            return node

    def fit(self, X, Y, sample_weight=[]):
        self.tree['features'] = list(X.columns)
        self.tree['root'] = self.build_tree(X,Y, 0, {}, sample_weight=sample_weight)

    def single_predict(self, x, tree):
        if(len(tree.keys())==1):
            return tree['val']
        elif(x[tree['col']]<tree['cutoff'] and tree['left']!=None):
            return self.single_predict(x, tree['left'])
        elif(x[tree['col']]<tree['cutoff'] and tree['left']==None):
            return tree['val']
        elif(x[tree['col']]>=tree['cutoff'] and tree['right']!=None):
            return self.single_predict(x, tree['right'])
        elif(x[tree['col']]>=tree['cutoff'] and tree['right']==None):
            return tree['val']

    def predict(self, test, tree):
        predictions = []
        for i in test.index.to_numpy():
            predictions.append(self.single_predict(test[tree['features']].loc[i], tree['root']))
        return np.asarray(predictions)
