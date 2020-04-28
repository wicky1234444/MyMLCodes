import numpy as np

class MLP:
    def __init__(self, layer_dimensions, lr, activation_function):
        self.W = []
        for i in range(len(layer_dimensions)-1):
            Weights = np.zeros((layer_dimensions[i+1], layer_dimensions[i]))
            self.W.append(Weights)
        self.lr = lr
        self.act = activation_function

    def FeedForward(self, input):
        L = len(self.W)
        layer_wise_outputs = []
        layer_wise_outputs.append(input)
        for i in range(L):
            output = np.dot(layer_wise_outputs[-1],np.transpose(self.W[i]))
            layer_wise_outputs.append(output)
        return layer_wise_outputs

    def calculate_grad(self, target, layer_wise_outputs):
        output_error = (target-layer_wise_outputs[-1])[0]
        local_grads = []
        grad = []
        L = len(layer_wise_outputs)
        for i in range(1,L):      #number of layers
            grads = np.zeros(self.W[L-i-1].shape)
            local = []
            for j in range(self.W[L-i-1].shape[0]):
                if i==1:                                        ## for last layer
                    local_grad = -output_error[j]*layer_wise_outputs[-1][0][j]*(1-layer_wise_outputs[-1][0][j])
                    local.append(local_grad)
                else:                                           ## for all other hidden layers
                    local_grad = layer_wise_outputs[L-i][0][j]*(1-layer_wise_outputs[L-i][0][j])
                    p=0
                    for h in range(layer_wise_outputs[L-i+1].shape[1]):
                        p+=local_grads[-1][h]*self.W[L-i][h][j]
                    local_grad*=p
                    local.append(local_grad)
                for k in range(self.W[L-i-1].shape[1]):
                    grads[j][k] = self.lr*local_grad*layer_wise_outputs[L-i-1][0][k]
            local_grads.append(local)
            grad.append(grads)
        return grad

    def backprop(self, grad):
        L=len(self.W)
        for i in range(L):
            grads = grad[i]
            for j in range(self.W[L-i-1].shape[0]):
                for k in range(self.W[L-i-1].shape[1]):
                    self.W[L-i-1][j][k]+=grads[j][k]
