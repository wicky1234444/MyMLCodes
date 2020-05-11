import numpy as np

class MLP:      ##multi layer perceptron
    def __init__(self, layer_dimensions, lr, activation_function='sigmoid'):
        self.W = []        ## weights
        self.B = []        ## biases                         
        for i in range(len(layer_dimensions)-1):    ## weight and bias initialization ~ uniform(0,1)
            Weights = np.random.uniform(0,1, (layer_dimensions[i+1], layer_dimensions[i]))
            bias = np.random.uniform(0,1, layer_dimensions[i+1])
            self.W.append(Weights)
            self.B.append(bias)
        self.lr = lr
        self.act = activation_function      ##sigmoid is used for now

    def Forward(self, input):           ## feed the input forward into the network and calculate layer wise outputs
        L = len(self.W)
        layer_wise_outputs = []
        layer_wise_outputs.append(input)
        for i in range(L):
            output = np.dot(layer_wise_outputs[-1],np.transpose(self.W[i]))
            output+=self.B[i]
            if self.act == 'sigmoid':       ## non-linear activation
                output = 1/(1+np.exp(-output))
            elif self.act == 'tanh':
                output = np.tanh(output)
            elif self.act == 'relu':
                output = max(output, 0)
            layer_wise_outputs.append(output)
        return layer_wise_outputs

    def gradient_calci(self, outputs, target):
        error = (target-outputs[-1])
        W_grad = []     ## gradients for weights
        B_grad = []     ## local gradients (also gradients for biases)
        B_grad.append(np.multiply(error, np.multiply(1-outputs[-1],outputs[-1])))
        for i in reversed(range(len(self.W))):
            W_grad.append(np.dot(B_grad[-1].reshape(-1,1), outputs[i].reshape(1,-1)))
            gr = np.dot(B_grad[-1], self.W[i])
            gr = np.multiply(np.multiply(1-outputs[i], outputs[i]), gr)
            B_grad.append(gr)
        return [W_grad, B_grad]

    def backprop(self, grad):
        k=0               ##update the weights and biases of the network based on gradents from gradient_calci
        W_grad = grad[0]
        B_grad = grad[1]
        for i in reversed(range(len(self.W))):   
            self.W[i]+=self.lr*W_grad[k]
            self.B[i]+=self.lr*B_grad[k]
            k+=1
