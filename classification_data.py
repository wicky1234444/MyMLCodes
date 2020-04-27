import numpy as np
import pandas as pd
import random

train_samples, test_samples = 4000, 2000
prior = [0.7,0.3]
mean1, mean2 = [3,6], [3,-2]
cov1, cov2 = [[0.5,0],[0,2]], [[2,0],[0,2]]

train_data, class_label = [], []
data = pd.DataFrame(columns = ['x1', 'x2', 'class_label'], index=[x for x in range(train_samples+test_samples)])
for i in range(train_samples+test_samples):
    if(random.uniform(0,1)<=prior[0]):
        train_data.append(np.random.multivariate_normal(mean1, cov1, 1))
        class_label.append(1)
    else:
        train_data.append(np.random.multivariate_normal(mean2, cov2, 1))
        class_label.append(-1)
    data.loc[i] = np.append(train_data[i][0],class_label[i])
    
data.to_csv('./2d_gaussian_2_class_data.csv', index=False)
