import numpy as np

class convolution_layer:
    def __init__(self, kernel_dim, pooling='Max', stride=1):
        self.n_kernels = kernel_dim[0]
        self.kernel_shape = kernel_dim[1:]
        self.kernel = []
        for i in range(self.n_kernels):
            if len(kernel_dim[1:])==2:
                self.kernel.append(np.random.randn(kernel_dim[1], kernel_dim[2]))
            elif len(kernel_dim[1:])==3:
                self.kernel.append(np.random.randn(kernel_dim[1], kernel_dim[2], kernel_dim[3]))
        self.pooling_layer = pooling
        self.s = stride

    def convolve(self, input, activation_function='relu'):
        convolution_output = []
        if type(input)==list:
            steps1 = int(np.floor((input[0].shape[0]-self.kernel_shape[0])/self.s))
            steps2 = int(np.floor((input[0].shape[1]-self.kernel_shape[1])/self.s))
            for h in range(len(input)):
                for i in range(self.n_kernels):
                    img_out = np.zeros((steps1,steps2))
                    j = 0
                    while j < steps2:
                        k = 0
                        while k < steps1:
                            img_out[k,j] = np.sum(np.multiply(self.kernel[i], input[h][k:k+self.kernel_shape[0], j:j+self.kernel_shape[1]]))
                            if activation_function == 'relu':                   
                                img_out[k,j] = np.max(img_out[k,j],0)
                            k+=self.s
                        j+=self.s
                    convolution_output.append(img_out)
        else:
            steps1 = int(np.floor((input.shape[0]-self.kernel_shape[0])/self.s))
            steps2 = int(np.floor((input.shape[1]-self.kernel_shape[1])/self.s))
            for i in range(self.n_kernels):
                img_out = np.zeros((steps1,steps2))
                j = 0
                while j < steps2:
                    k = 0
                    while k < steps1:
                        img_out[k,j] = np.sum(np.multiply(self.kernel[i], input[k:k+self.kernel_shape[0], j:j+self.kernel_shape[1]]))
                        if activation_function == 'relu':                   
                            img_out[k,j] = np.max(img_out[k,j],0)
                        k+=self.s
                    j+=self.s
                convolution_output.append(img_out)
        return convolution_output

    def subsampling(self, input, pool_dim):          
        steps1 = int(np.floor((input[0].shape[0]-pool_dim[0])/self.s))
        steps2 = int(np.floor((input[0].shape[1]-pool_dim[1])/self.s))
        pool_output = []
        for i in range(len(input)):
            img = np.zeros((steps1,steps2))
            j,k = 0,0
            while j < steps2:
                while k < steps1:
                    if self.pooling_layer=='Max':
                        img[k,j] = np.max(input[i][k:k+pool_dim[0], j:j+pool_dim[0]])
                    elif self.pooling_layer=='Average':
                        img[k,j] = np.mean(input[i][k:k+pool_dim[0], j:j+pool_dim[0]])
                    k+=self.s
                j+=self.s
            pool_output.append(img)
        return pool_output

class dense_layer:
    def __init__(self, layer_dim, activation_function='softmax'):
        self.W = np.random.randn(layer_dim[0], layer_dim[1])
        self.activation = activation_function      

    def forward(self, input):
        out = np.dot(self.W, input)
        out = np.exp(out)/sum(np.exp(out))
        return out

