import numpy as np

class MLP:      ##multi layer perceptron
    def __init__(self, layer_dimensions, lr, activation_function='sigmoid'):
        self.W = []
        for i in range(len(layer_dimensions)-1):    ##xavier initialization
            Weights = np.random.randn(layer_dimensions[i+1], layer_dimensions[i])*np.sqrt(2/layer_dimensions[i])
            self.W.append(Weights)
        self.lr = lr
        self.act = activation_function      ##sigmoid is used for now

    def Forward(self, input):           ##feed the input forward into the network and calculate layer wise outputs
        L = len(self.W)
        layer_wise_outputs = []
        layer_wise_outputs.append(input)
        for i in range(L):
            output = np.dot(layer_wise_outputs[-1],np.transpose(self.W[i]))
            if self.act == 'sigmoid':
                output = 1/(1+np.exp(-output))
            layer_wise_outputs.append(output)
        return layer_wise_outputs

    def calculate_grad(self, target, layer_wise_outputs):   ##returns the gradients based on the target vector and output from FeedForward() 
        ## gradient(j) = learning rate * local gradient(i) * input(j)
        ## local gradient(i) = -(target(i)-output(i))*sigmoid'(i)     if i is the last layer
        ## local gradient(i) = sigmoid'(i)*sum(W(k)*local gradient(k)) if i is the hidden layer and k is the next layer in forward direction
        last_layer_error = (target-layer_wise_outputs[-1])
        local_grads = []
        grad = []       
        L = len(layer_wise_outputs)         ##number of layers
        for i in range(1,L):
            grads = np.zeros(self.W[L-i-1].shape)       ##layer wise gradients
            local = []
            for j in range(self.W[L-i-1].shape[0]):
                if i==1:                     ## for the last layer
                    local_grad = -last_layer_error[j]*layer_wise_outputs[-1][j]*(1-layer_wise_outputs[-1][j])
                    local.append(local_grad)
                else:                        ## for all other hidden layers
                    local_grad = layer_wise_outputs[L-i][j]*(1-layer_wise_outputs[L-i][j])
                    p=0
                    for h in range(layer_wise_outputs[L-i+1].shape[1]):
                        p+=local_grads[-1][h]*self.W[L-i][h][j]
                    local_grad*=p
                    local.append(local_grad)
                for k in range(self.W[L-i-1].shape[1]):
                    grads[j][k] = self.lr*local_grad*layer_wise_outputs[L-i-1][k]
            local_grads.append(local)
            grad.append(grads)
        return grad

    def backprop(self, grad):               ##update the weights on the networks based on gradents from calculate-gradients
        L=len(self.W)
        for i in range(L):
            grads = grad[i]
            for j in range(self.W[L-i-1].shape[0]):
                for k in range(self.W[L-i-1].shape[1]):
                    self.W[L-i-1][j][k]+=grads[j][k]
