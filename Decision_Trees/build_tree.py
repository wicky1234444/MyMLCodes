from Decision_Trees.tree_splitting_criterion import Information_Gain, Entropy
import numpy as np
import pandas as pd

class Decision_tree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.tree = {}

    def find_best_split(self, X, col, Y):
        min_entropy = 10
        split_val = -1
        for val in set(X[col]):
            #print(val)
            y_pred = X[col]<val
            entropy = Entropy(Y.to_numpy(), y_pred.to_numpy())
            if(entropy<=min_entropy):
                min_entropy=entropy
                split_val = val
        return [min_entropy, split_val]

    def best_column_to_split(self, X, Y):
        min_entropy = 10
        split_val = -1
        split_col = ""
        for col in list(X.columns)[:-1]:
            entropy, val = find_best_split(X, col, Y)
            if entropy==0:
                return [entropy, val, col]
            elif(entropy<=min_entropy):
                min_entropy = entropy
                split_val = val
                split_col = col
        return [min_entropy, split_val, split_col]

    def build_tree(self, X, Y, depth, node = {}):
        if node==None:
            return None
        elif len(Y)==0:
            return None
        elif len(np.unique(Y))==1:
            return {'val':Y.to_numpy()[0]}
        elif depth>=self.max_depth:
            return None
        else:
            entropy, cutoff, col = self.best_column_to_split(X, Y)
            y_left = Y[X[col]<cutoff]
            y_right = Y[X[col]>=cutoff]
            node = {'col': col, 'cutoff':cutoff, 'val':np.mean(Y)}
            node['left'] = self.build_tree(X[X[col]<cutoff], y_left, depth+1, {})
            node['right'] = self.build_tree(X[X[col]>=cutoff], y_right, depth+1, {})
            return node

    def fit(self, X, Y):
        self.tree = self.build_tree(X,Y, 0, {})

    def predict(self, x, tree):
        if(len(tree.keys())==1):
            return tree['val']
        elif(x[tree['col']]<tree['cutoff']):
            return self.predict(x, tree['left'])
        else:
            return self.predict(x, tree['right'])
