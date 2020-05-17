import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class Logistic:
    def __init__(self, W_dimensions, lr):
        self.W = np.zeros(W_dimensions+1)
        self.lr = lr
        
    def update(self, X):
        Y = X.values[-1]
        if Y==-1:
            Y=0
        X = np.append(1, X.values)[0:-1]
        z = sigmoid(np.dot(self.W, X))              ## error function used is log loss since it gives a convex error surface
        self.W-=self.lr*X*(z-Y)                     ##classification threshold is set at 0.5
        if (z>=0.5 and Y==1) or (z<0.5 and Y==0):
            return 0
        else:
            return 1
        
    def predict(self, X):
        Y = X.values[-1]
        X = np.append(1, X.values)[0:-1]
        z = sigmoid(np.dot(self.W, X))
        if z>=0.5:
            return [1, Y]
        else:
            return [-1, Y]