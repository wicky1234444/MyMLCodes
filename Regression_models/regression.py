import numpy as np
from itertools import combinations_with_replacement as cr

class regression:
    def __init__(self, alg = 'least_squares', alg_params = 1, features = 'linear', degree = 1):
        self.method = alg
        self.k = alg_params
        self.features = features
        self.degree = degree
        self.W = []

    def features_calci(self, X1):
        m, n = X1.shape
        if self.features == 'poly':
            X2 = []
            c = cr([j for j in range(n)], self.degree)
            print([j for j in range(n)])
            f = list(c)
            print(f)
            for i in range(X1.shape[0]):
                w = np.ones((1,len(f)))[0]
                q = X1[i][np.array(f)]
                for j in range(q.shape[1]):
                    w = w*q[:,j]
                X2.append(w)
            X1 = np.asarray(X2)
        return X1
        
    def fit(self, X, Y):
        m,n = X.shape
        X1 = np.hstack((np.ones((m,1)), X))
        n = X1.shape[1]
        X1 = self.features_calci(X1)
        #print(X1)
        if self.method == 'least_squares':
            a = np.dot(X1.T, X1)
            b = np.linalg.inv(a)
            c = np.dot(b,X1.T)
            self.W = np.dot(c, Y)
        elif self.method == 'ridge':
            a = np.dot(X1.T, X1)+ self.k*np.eye(X.shape[0])
            b = np.linalg.inv(a)
            c = np.dot(b,X1.T)
            self.W = np.dot(c, Y)

    def predict(self, X):
        X1 = np.hstack((np.ones((X.shape[0],1)),X))
        X1 = self.features_calci(X1)
        return np.dot(X1, self.W)