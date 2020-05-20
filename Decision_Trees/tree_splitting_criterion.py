import numpy as np

def Entropy(class1, class2=[]):
    ## entropy is measure of randomness, attribute with highest entropy is splitted at each node. attribute-
    ## with 0 entropy will be a leaf node, entropy>0 needs further splitting.
    ## class1 is usually the target class
    if(len(class2)==0):
        unique_counts = np.unique(class1, return_counts=True)
        size = len(class1)
        prob = unique_counts[1]/size
        return -np.sum(prob*np.log2(prob))
    else:   ##groupby class2 and find entropy of each group
        unique_counts = np.unique(class2, return_counts=True)
        size = len(class2)
        prob = unique_counts[1]/size
        entropy=0
        classes = np.asarray(unique_counts[0])
        for i in range(len(classes)):  ##for each class
            ind = np.where(class2==classes[i])[0]
            p = Entropy(class1[ind])
            entropy+= prob[i]*p
        return entropy

def Information_Gain(class1, class2):       ## change in entropy before and after split
    return (Entropy(class1) - Entropy(class1, class2))

def Gini_index(class1, class2):
    ## higher the value of gini index higher the homogeneity.
    unique_counts = np.unique(class2, return_counts=True)
    size = len(class2)
    prob = unique_counts[1]/size
    gini=0
    classes = np.asarray(unique_counts[0])
    for i in range(len(classes)):  ##for each class
        ind = np.where(class2==classes[i])[0]
        p1 = np.unique(class1[ind], return_counts=True)
        p2 = p1[1]/np.sum(p1[1])
        p3 = np.sum(np.power(p2,2))
        gini+= p3*prob[i]
    return gini



