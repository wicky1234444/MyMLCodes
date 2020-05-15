import numpy as np
from numpy import trapz

def confusion_matrix(true_labels, predictions):
    # output is dictionary containing classes, confusion matrix and measures like predictive values, recall
    classes = np.unique(true_labels)
    N = classes.shape[0]
    cm = np.zeros((N,N))
    confusion_matrix_dict = {}
    confusion_matrix_dict['classes'] = classes
    confusion_matrix_dict['measures'] = {}
    for i in range(N):
        ind = np.where(predictions == classes[i])
        true_counts = np.unique(true_labels[ind], return_counts=True)
        for j in range(true_counts[0].shape[0]):
            cm[i][np.where(classes==true_counts[0][j])] = true_counts[1][j]
        confusion_matrix_dict['measures']['class '+str(classes[i])+' precision'] = cm[i][i]/sum(cm[i])
    a = 0
    for i in range(N):
        confusion_matrix_dict['measures']['class '+str(classes[i])+' recall'] = cm[i][i]/(sum(cm[:, i]))
        a+=cm[i][i]
    confusion_matrix_dict['measures']['accuracy'] = a/np.sum(cm)
    
    confusion_matrix_dict['matrix'] = cm
    return confusion_matrix_dict

def F1_score(true_labels, predictions):
    ## F1 score is used when we are trying to improve both precision and recall, it takes care of extreme values-
    ##-since it is a harmonic mean of values
    f1_score = {}
    cm_dict  = confusion_matrix(true_labels, predictions)
    for i in range(cm_dict['classes'].shape[0]):
        f1_score['class '+str(cm_dict['classes'][i])+' f1_score'] = (2*cm_dict['measures']['class '+str(cm_dict['classes'][i])+' precision']*cm_dict['measures']['class '+str(cm_dict['classes'][i])+' recall'])/(cm_dict['measures']['class '+str(cm_dict['classes'][i])+' precision']+cm_dict['measures']['class '+str(cm_dict['classes'][i])+' recall'])
    return f1_score

def AUROC(true_labels, prediction_probabilities): ##assuming class labels are 1 and -1
    ## area > 0.9 is considered very good predictor, area<0.5 is a very bad model and area=0 predicts +ve as -ve and vice versa
    threshold = np.linspace(0,1, 20)
    x1,y1 = [],[]
    true_labels = np.where(true_labels==0, -1, true_labels)
    for t in threshold:
        y = prediction_probabilities[:,1]
        y = np.where(y<t, -1, y)
        y = np.where(y>=t, 1, y)
        d1 = confusion_matrix(true_labels, y)
        #print(d1['measures'])
        y1.append(d1['measures']['class 1.0 recall'])
        x1.append(1-d1['measures']['class -1.0 recall'])
    area = trapz(y1, x= x1)
    print(y1,x1)
    return -area

def Gini_coefficient(true_labels, prediction_probabilities):
    ## gini coefficient > 0.6 is a good model
    return 2*AUROC(true_labels, prediction_probabilities)-1

def concordant_ratio(true_labels, prediction_probabilities):
    ## divide the true classes into +ve and -ve and calculate concordant and discordant pairs
    pos_prob = prediction_probabilities[:,1]
    pos = np.where(true_labels==1)
    neg = np.where(true_labels==-1)     ##or true_labels==0
    total_pairs = pos.shape[0]*neg.shape[0]
    concordant_pairs = 0
    discordant_pairs = 0
    for i in range(pos.shape[0]):
        for j in range(neg.shape[0]):
            if pos[i] > neg[j]:
                concordant_pairs+=1
            elif pos[i] < neg[j]:
                discordant_pairs+=1
    cd_dict = {}
    cd_dict['concordant ratio'] = concordant_pairs/total_pairs
    cd_dict['discordant_ratio'] = discordant_pairs/total_pairs
    return cd_dict

def R_square(true_labels, predictions, adjusted=False, k=1):
    ## gives a measure of how good our prediction model is when compared to the model that always predicts the-
    #-mean of true labels.
    y_mean = np.mean(true_labels)
    base_line = np.mean(np.sum(np.power((y_mean - predictions[:,1]),2)))
    model_mse = np.mean(np.sum(np.power((true_labels - predictions[:,1]),2)))
    r2 = 1-(model_mse/base_line)
    coeff=1
    if adjusted==True:      ## code for calculating adjusted R-square value(k= number of features)
        ## adjusted R square is to penalize the addition of features which add no value to the model. 
        n = true_labels.shape[0]
        coeff = (n-1)/(n-k-1)
        r2 = 1-(1-r2)*coeff
    return r2
