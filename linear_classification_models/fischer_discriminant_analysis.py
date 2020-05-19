import numpy as np

def fischer_da(class1_data, class2_data):
    M1 = np.zeros(class1_data.shape[1]-1)
    M2 = np.zeros(class2_data.shape[1]-1)
    i=0
    for column in class1_data.columns.to_list()[:-1]:
        M1[i] = class1_data[column].mean()
        M2[i] = class2_data[column].mean()
        i+=1
    Sw = np.zeros((class1_data.shape[1]-1, class1_data.shape[1]-1))
    for i in range(class1_data.shape[0]):
        Sw+=np.outer((class1_data.loc[i][0:-1].values-M1),np.transpose(class1_data.loc[i][0:-1].values-M1))
        
    for i in range(class2_data.shape[0]):
        Sw+=np.outer((class2_data.loc[i][0:-1].values-M2),np.transpose(class2_data.loc[i][0:-1].values-M2))
        
    w = np.dot(np.linalg.inv(Sw),(M2-M1)) ## W = Sw^(-1)*(M1-M0)

    print('W:', w)

    misses=0
    for i in range(class1_data.shape[0]):
        pred = np.dot(w, class1_data.loc[i][0:-1].values)
        if pred<0:
            misses+=1
    
    for i in range(class2_data.shape[0]):
        pred = np.dot(w, class2_data.loc[i][0:-1].values)
        if pred>=0:
            misses+=1

    print('train accuracy: ',(class1_data.shape[0]+class2_data.shape[0]-misses)/(class1_data.shape[0]+class2_data.shape[0]))

