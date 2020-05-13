import numpy as np
from itertools import combinations_with_replacement as cr

class linear_regression:
    def __init__(self, loss = 'least_squares', features = 'linear', degree = 1):
        self.loss = loss
        self.features = features
        self.degree = 1
        self.W = []

    def fit(self, X, Y):
        if self.features == 'poly':
            X2 = []
            c = cr([j for j in range(X.shape[1])], self.degree)
            f = list(c)
            for i in range(X.shape[0]):
                w = np.ones((1,len(f)))
                q = X[i][np.array(f)]
                for j in range(q.shape[1]):
                    w = w*q[:,j]
                X2.append(w)
            X = np.asarray(X2)
        if self.loss == 'least_squares':
            m,n = X.shape
            X1 = np.hstack((np.ones(m,1), X))
            a = np.dot(X1.T, X1)
            b = np.linalg.inv(a)
            c = np.dot(b,X1.T)
            self.W = np.dot(c, Y)

    def predict(self, X):
        X1 = np.hstack((np.ones(X.shape[0],1),X))
        return np.dot(X1, self.W)