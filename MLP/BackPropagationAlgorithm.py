import numpy as np

class BackPropagation:
    def __init__(self, Weights, input, target, activation_function, learning_rate):
        self.W = Weights
        self.input = input
        self.target = target
        self.act = activation_function
        self.lr = learning_rate
        self.grad=[]

    def calculate_grad(self):
        output_error = (self.target-self.input[-1])[0]
        local_grads = []
        L = len(self.input)
        for i in range(1,L):      #number of layers
            grads = np.zeros(self.W[L-i-1].shape)
            local = []
            for j in range(self.W[L-i-1].shape[0]):
                if i==1:                                        ## for last layer
                    local_grad = -output_error[j]*self.input[-1][0][j]*(1-self.input[-1][0][j])
                    local.append(local_grad)
                else:                                           ## for all other hidden layers
                    local_grad = self.input[L-i][0][j]*(1-self.input[L-i][0][j])
                    p=0
                    for h in range(self.input[L-i+1].shape[1]):
                        p+=local_grads[-1][h]*self.W[L-i][h][j]
                    local_grad*=p
                    local.append(local_grad)
                for k in range(self.W[L-i-1].shape[1]):
                    grads[j][k] = self.lr*local_grad*self.input[L-i-1][0][k]
            local_grads.append(local)
            self.grad.append(grads)
            return grad

    def update_weights(self):
        L=len(self.input)
        for i in range(1,L):
            grads = self.grad[i-1]
            for j in range(self.W[L-i-1].shape[0]):
                for k in range(self.W[L-i-1].shape[1]):
                    self.W[L-i-1][j][k]+=grads[j][k]