import numpy as np

class lasso_regression:
    def __init__(self, lr, lb):
        self.W = []
        self.b = 0
        self.lr = lr
        self.lb = lb

    def fit(self, X, Y, iterations):
        m,n = X.shape
        self.W = np.random.randn(n,1)
        
        for i in range(iterations):
            y_pred = np.dot(X, self.W)
            error = Y-y_pred