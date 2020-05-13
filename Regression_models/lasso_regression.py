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
            for j in range(n):   #for every feature that exists
                 
                yi = np.dot(X,self.W)   #calculate the sum of weights multiplied by features
                residuals = Y-yi      #calculating the error term
                x_j = X[:,j].reshape(-1,1)     #taking only the jth feature values
                
                #cost = np.sum(residuals**2) + self.l1*np.sum(np.abs(self.w))   #computing cost
                #self.cost_.append(cost)   
                rho = np.dot(x_j.T,(residuals+self.w[j,:]*x_j))   #calculating rho, for the jth feature alone 
                
                if(rho<(-self.l1/2)):   #soft thresh holding
                    self.w[j,:]=rho+(self.l1/2)
                elif(rho>(self.l1/2)):
                    self.w[j,:]=rho-(self.l1/2)
                else:
                    self.w[j,:]=0

    def predict(self, X):
        return np.dot(X, self.W)