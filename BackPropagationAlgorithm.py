import numpy as np

class BackPropagation:
    def __init__(self, Weights, activation_function, inputs, target, learning_rate):
        self.layers = len(Weights)
        self.Weights = Weights
        self.target = target
        self.input = inputs
        self.lr = learning_rate
        self.act = activation_function
    
    def calculate_gradients(self):
        L = self.layers
        error = self.target-self.input[L]
        grad = []
        for i in range(L):      #for each layer
            if i == 0:
                local_grad = self.lr*error*
            else:
                local_gradients = np.zeros(self.Weights.shape[0])
                
