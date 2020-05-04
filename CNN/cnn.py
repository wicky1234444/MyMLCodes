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
        steps = int(np.floor((input.shape[0]-self.kernel_shape[0])/self.s))
        for i in range(self.n_kernels):
            img_out = np.zeros((steps,steps))
            j,k = 0,0
            while j < input[0].shape[0]:
                while k < input[0].shape[1]:
                    img_out[j,k] = np.sum(np.multiply(kernel[i], input[k:k+self.kernel_shape[0], j:j+self.kernel_shape[0]]))
                    if activation_function == 'relu':                   
                        img_out[j,k] = np.max(img_out[j,k],0)
                k+=self.s
            j+=self.s
            convolution_output.append(img_out)
        return convolution_output

    def subsampling(self, input, pool_dim):          
        steps = int(np.floor((input[0].shape[0]-pool_dim[0])/self.s))
        pool_output = []
        for i in range(len(input)):
            img = np.zeros((steps,steps))
            j,k = 0,0
            while j < input[0].shape[0]:
                while k < input[0].shape[1]:
                    if self.pooling_layer=='Max':
                        img[j,k] = np.max(input[i][k:k+pool_dim[0], j:j+pool_dim[0]])
                    elif self.pooling_layer=='Average':
                        img[j,k] = np.mean(input[i][k:k+pool_dim[0], j:j+pool_dim[0]])
                    k+=self.s
                j+=self.s
            pool_output.append(img)
        return pool_output

        


