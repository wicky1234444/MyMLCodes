import numpy as np
import numpy
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
##solving SVM as a convex optimization problem

class SVM:
    def __init__(self):
        self.lagrangian_multipliers = []

    def convex_optimization(self, X, Y, C=1, kernel = 'None', p):
        m,n = X.shape
        for i in range(Y.shape[0]):
            X[i] = X[i]*Y[i]
        H = np.dot(X,X.T)
        if kernel=='polynomial':
            H = H+1
            H = np.power(H,p)
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
        return alphas
    
    def kernel_trick(kernel, x1, x2, p):
        if kernel == 'poly':
            return np.power(np.dot(x1,x2.T)+1,p)
        elif kernel == 'gaussian':
            return (1/(2*p))*np.exp(-np.power((x1-x2),2)/(2*p))

