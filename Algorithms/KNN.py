import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from collections import Counter


def dist (p,q):
    return  np.sqrt(np.sum((p-q)**2))

#X,y -- Training data
#xt -- test point 
# X -(a,b)--where a is number of points with b number of features
#Y - (a,)--where a is label for corresponding point of x
def knn(X,y,xt,k=5):
    m = X.shape[0]
    dList = []
    for i in range(m) :
        test_dist = dist(X[i],xt)
        dList.append((test_dist,y[i]))
    sorted_dList = sorted(dList)
    final_classes = []
    for i in sorted_dList:
        final_classes.append(i[1])
    final_classes_count = Counter(final_classes)
    result_class =  final_classes_count.most_common(1)[0][0]
    print(result_class)
    
    return result_class


    


# Preparation of Sample Data for KNN
X,y = make_blobs(n_samples=2000,n_features=3,cluster_std=3,centers=3,random_state=42)
n_features = 3
n_points = X[0]
xt = np.array([-10,5,4])
finalList = knn(X=X,y=y,xt=xt,k=5)