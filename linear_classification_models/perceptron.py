import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import copy

class perceptron:
    def __init__(self, W_dimensions, lr):
        self.W = np.zeros(W_dimensions+1) ## including bias
        self.lr = lr
        
    def update(self, X):                        ##update the weights for all missclassified points 
        misclassification=0
        Y = X.values[-1]
        X = np.append(1, X.values)[0:-1] 
        if Y==-1 and np.dot(self.W,X)>=0:
            self.W-=self.lr*X
            misclassification=1
        elif Y==1 and np.dot(self.W,X)<0:
            self.W+=self.lr*X
            misclassification=1
        return misclassification
    
    def predict(self, x):                         ##predict the class of the input 
        y = x.values[-1]
        x = np.append(1, x.values)[0:-1]
        if np.dot(self.W, x)>=0 and y==1:
            return [1,1]
        elif np.dot(self.W,x)>=0 and y==-1:
            return [1,-1]
        elif np.dot(self.W,x)<0 and y==-1:
            return [-1,-1]
        else:
            return [-1,1]

def train(epochs, threshold, classifier, train_data):           ##on-line training of data
    best_W = np.zeros(train_data.shape[1])
    best_so_far=train_data.shape[0]
    best_accuracy=0
    for epoch in range(epochs):
        train_data = shuffle(train_data)
        missed=0
        for i in range(train_data.shape[0]):
            miss = classifier.update(train_data.loc[i])
            missed+=miss
        train_accuracy = (train_data.shape[0]-missed)/train_data.shape[0]
        if missed< best_so_far:                             ##save the model with best accuracies
            best_accuracy = train_accuracy
            best_so_far = missed
            best_W = copy.deepcopy(classifier.W)
    print('train_accuracy: ', best_accuracy)
    return best_W
        
def test(classifier, data):                                 ##calculate accuracy on test data
    accuracy = 0
    for i in range(data.shape[0]):
        [p,l] = classifier.predict(data.loc[i])
        if p==l:
            accuracy+=1
    print('test accuracy: ', accuracy/data.shape[0])
    return  accuracy/data.shape[0]



data = pd.read_csv('./2d_gaussian_2_class_data.csv')    ##run classification_data.py to get train and test data
train_data = data[0:2000]
test_data = data[2000:]
test_data = test_data.reset_index()
del test_data['index']

##visualize data

import matplotlib.pyplot as plt

c1 = test_data[test_data['class_label']==1]
c2 = test_data[test_data['class_label']==-1]
#print(c1['x1'].values, c1['x2'].values)
plt.plot(c1['x1'].values, c1['x2'].values, 'rx', label='class 1')
plt.plot(c2['x1'].values, c2['x2'].values, 'bx', label='class 0')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

##perceptron training and testing

classifier_p = perceptron(2,1)
classifier_p.W = copy.deepcopy(train(20, 0, classifier_p, train_data))## train for 20 iterations
print(classifier_p.W)
test(classifier_p, train_data)
test(classifier_p, test_data)

##visualize decision boundary
plt.figure(figsize=[10,10])
x = np.linspace(min(test_data['x1']), max(test_data['x2']), 50)
y_p = (-classifier_p.W[1]/classifier_p.W[2])*x + (-classifier_p.W[0]/classifier_p.W[2])
plt.plot(c1['x1'].values, c1['x2'].values, 'rx', label='class 1')
plt.plot(c2['x1'].values, c2['x2'].values, 'bx', label='class 0')
plt.plot(x,y_p,'g-', label='perceptron decision boundary')
plt.ylim(min(test_data['x2']), max(test_data['x2']))
plt.xlim(min(test_data['x1']), max(test_data['x2']))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()