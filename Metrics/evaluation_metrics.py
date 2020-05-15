import numpy as np

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
    f1_score = {}
    cm_dict  = confusion_matrix(true_labels, predictions)
    for i in range(cm_dict['classes'].shape[0]):
        f1_score['class '+str(cm_dict['classes'][i])+' f1_score'] = (2*cm_dict['measures']['class '+str(cm_dict['classes'][i])+' precision']*cm_dict['measures']['class '+str(cm_dict['classes'][i])+' recall'])/(cm_dict['measures']['class '+str(cm_dict['classes'][i])+' precision']+cm_dict['measures']['class '+str(cm_dict['classes'][i])+' recall'])
    return f1_score