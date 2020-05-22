from Decision_Trees.tree_splitting_criterion import Information_Gain, Entropy
import numpy as np
import pandas as pd

class Decision_Tree:
    def __init__(self, criterion='Entropy'):
        self.crit = criterion

    def build_tree(self, X, Y):
        ## X and Y are pandas dataframes
        attributes = list(X.columns)
        attrib = self.find_best_attribute(attributes)
        attributes = attributes-set([attrib])


    def find_best_attribute(self, attributes, X, Y):
        min_score = 0
        split_attribute=None
        for attrib in attributes:
            en = Entropy(Y, X[attrib])
            if(en>min_score):
                min_score = en
                split_attribute = attrib
        return attrib