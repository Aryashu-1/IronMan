import numpy as np


def dist (p,q):
    return  np.sqrt(np.sum(p-q)**2)

def knn(X,y,xt,k=5):
    m = X.shape[0]
    dlist =[]
    
