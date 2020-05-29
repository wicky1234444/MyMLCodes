import numpy as np

def Entropy(class1, class2=[], sample_weight=[]):
    ## entropy is measure of randomness, attribute with highest entropy is splitted at each node. attribute-
    ## with 0 entropy will be a leaf node, entropy>0 needs further splitting.
    ## class1 is usually the target class
    if(len(class2)==0):
        unique_counts1 = np.unique(class1, return_counts=True)
        if(len(sample_weight)==0):
            size = len(class1)
            prob = unique_counts1[1]/size
            return -np.sum(prob*np.log2(prob))
        else:
            entropy = 0
            for j in range(len(unique_counts1[0])):
                ind1 = np.where(class1==unique_counts1[0][j])[0]
                p = np.sum(sample_weight[ind1])
                entropy+=(-p*np.log2(p))
            return entropy
    else:   ##groupby class2 and find entropy of each group
        unique_counts = np.unique(class2, return_counts=True)
        size = len(class2)
        prob = unique_counts[1]/size
        entropy=0
        classes = np.asarray(unique_counts[0])
        for i in range(len(classes)):  ##for each class
            ind = np.where(class2==classes[i])[0]
            if len(sample_weight)!=0:
                p = Entropy(class1[ind], sample_weight[ind])
            else:
                p = Entropy(class1[ind])
            entropy+= prob[i]*p
        return entropy

def Information_Gain(class1, class2, sample_weight=[]):       ## change in entropy before and after split
    return (Entropy(class1, sample_weight=sample_weight) - Entropy(class1, class2, sample_weight=sample_weight))

def Gini_index(class1, class2, sample_weight=[]):
    ## higher the value of gini index higher the homogeneity.
    unique_counts = np.unique(class2, return_counts=True)
    size = len(class2)
    prob = unique_counts[1]/size
    gini=0
    classes = np.asarray(unique_counts[0])
    for i in range(len(classes)):  ##for each class
        ind = np.where(class2==classes[i])[0]
        p1 = np.unique(class1[ind], return_counts=True)
        if len(sample_weight)==0:
            p2 = p1[1]/np.sum(p1[1])
            p3 = np.sum(np.power(p2,2))
            gini+= p3*prob[i]
        else:
            p2 = 0
            for j in range(len(p1[0])):
                ind1 = np.where(class1[ind]==p1[0][j])[0]
                p2+=np.power(np.sum(sample_weight[ind1]),2)
            gini+=p2*prob[i]
    return gini

def Chi_square(class1, class2):
    class_values = np.unique(class2, return_counts=True)
    classes = np.unique(class1, return_counts=True)
    expected = classes[1]/np.sum(classes[1])
    chaid = 0
    for i in range(len(class_values[0])):  ##for each class
        ind = np.where(class2==class_values[0][i])[0]
        p1 = np.unique(class1[ind], return_counts=True)
        p2 = np.power(np.divide(np.power((p1[1]-expected*np.sum(p1[1])),2),expected*np.sum(p1[1])), 0.5)
        #print(p1, p2)
        chaid +=np.sum(p2)
    return chaid

#def Variance_reduction():



