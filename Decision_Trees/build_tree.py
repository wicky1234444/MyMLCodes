from tree_splitting_criterion import Information_Gain, Entropy, Gini_index
import numpy as np
import pandas as pd

class Decision_tree:
    def __init__(self, max_depth=10, split_criterion = 'Entropy'):
        self.max_depth = max_depth
        self.tree = {}
        self.criterion = split_criterion

    def find_best_split(self, X, col, Y):
        if self.criterion == 'Entropy' or self.criterion == 'Gini':
            criterion_val = 10
        elif self.criterion == 'IG':
            criterion_val = -1
        split_val = -1
        for val in set(X[col]):
            #print(val)
            y_pred = X[col]<val
            if self.criterion == 'Entropy' or self.criterion == 'Gini':
                entropy = Entropy(Y.to_numpy(), y_pred.to_numpy())
                if(entropy<=criterion_val):
                    criterion_val=entropy
                    split_val = val
            elif self.criterion == 'IG':
                ig = Information_Gain(Y.to_numpy(), y_pred.to_numpy())
                if(ig>=criterion_val):
                    criterion_val = ig
                    split_val = val
        return [criterion_val, split_val]

    def best_column_to_split(self, X, Y):
        if self.criterion == 'Entropy' or self.criterion == 'Gini':
            criterion_val = 10
        elif self.criterion == 'IG':
            criterion_val = -1
        split_val = -1
        split_col = ""
        for col in list(X.columns)[:-1]:
            if self.criterion == 'Entropy' or self.criterion == 'Gini':
                entropy, val = self.find_best_split(X, col, Y)
                if entropy==0:
                    return [entropy, val, col]
                elif(entropy<=criterion_val):
                    criterion_val = entropy
                    split_val = val
                    split_col = col
            elif self.criterion == 'IG':
                ig, val = self.find_best_split(X, col, Y)
                if ig==1:
                    return [ig, val, col]
                elif(ig>=criterion_val):
                    criterion_val = ig
                    split_val = val
                    split_col = col
        return [criterion_val, split_val, split_col]

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
        elif(x[tree['col']]<tree['cutoff'] and tree['left']!=None):
            return self.predict(x, tree['left'])
        elif(x[tree['col']]<tree['cutoff'] and tree['left']==None):
            return tree['val']
        elif(x[tree['col']]>=tree['cutoff'] and tree['right']!=None):
            return self.predict(x, tree['right'])
        elif(x[tree['col']]>=tree['cutoff'] and tree['right']==None):
            return tree['val']
