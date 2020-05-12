import numpy as np
import numpy
import copy
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
##solving SVM as a convex optimization problem

class SVM:
    def __init__(self, kernel, kernel_parameters=1):
        self.lagrangian_multipliers = []
        self.support_vectors = []
        self.sv_classes = []
        self.kernel = kernel
        self.param = kernel_parameters

    def convex_optimization(self, X, Y, C=1, alpha_threshold = 1e-7):
        m,n = X.shape
        X1= copy.deepcopy(X)
        for i in range(Y.shape[0]):
            X[i] = X[i]*Y[i]
        H = self.kernel_trick(X,X)
        Y = Y.reshape(1,-1).astype(float)
        
        P = cvxopt_matrix(H)
        q = cvxopt_matrix(-np.ones((m, 1)))
        if C!=0:
            G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
            h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
        else:
            G = cvxopt_matrix(-np.eye(m))
            h = cvxopt_matrix(np.zeros(m))
        A = cvxopt_matrix(Y)
        b = cvxopt_matrix(np.zeros(1))
        
        cvxopt_solvers.options['show_progress'] = False

        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x'])

        alphas = alphas.reshape(1,-1)[0]
        alphas = np.where(alphas<alpha_threshold, 0, alphas)
        alpha_ind = np.nonzero(alphas)[0]
        self.lagrangian_multipliers = alphas[np.array(alpha_ind)]
        self.support_vectors = X1[np.array(alpha_ind)]
        self.sv_classes = Y[0][np.array(alpha_ind)]
        return alphas
    
    def kernel_trick(self,x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1,x2.T)
        elif self.kernel == 'poly':
            return np.power(np.dot(x1,x2.T)+1,self.param)
        elif self.kernel == 'gaussian':
            return (1/(2*self.param))*np.exp(-np.power((x1-x2),2)/(2*self.param))

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            p = self.lagrangian_multipliers*self.sv_classes*self.kernel_trick(self.support_vectors, X[i])
            p = np.sum(p)
            if p<0:
                predictions.append(-1)
            else:
                predictions.append(1)
        return np.asarray(predictions)
