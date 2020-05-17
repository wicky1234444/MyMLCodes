import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import copy

class perceptron:
    def __init__(self, W_dimensions, lr):
        self.W = np.zeros(W_dimensions+1) ## including bias
        self.lr = lr
        
    def update(self, X):                        ##update the weights for all missclassified points 
        misclassification=0
        Y = X.values[-1]
        X = np.append(1, X.values)[0:-1] 
        if Y==-1 and np.dot(self.W,X)>=0:
            self.W-=self.lr*X
            misclassification=1
        elif Y==1 and np.dot(self.W,X)<0:
            self.W+=self.lr*X
            misclassification=1
        return misclassification
    
    def predict(self, x):                         ##predict the class of the input 
        y = x.values[-1]
        x = np.append(1, x.values)[0:-1]
        if np.dot(self.W, x)>=0 and y==1:
            return [1,1]
        elif np.dot(self.W,x)>=0 and y==-1:
            return [1,-1]
        elif np.dot(self.W,x)<0 and y==-1:
            return [-1,-1]
        else:
            return [-1,1]