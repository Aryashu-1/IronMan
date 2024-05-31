import os
import cv2
import numpy as np


#Data
datasetPath = 'C:/Users/DELL/IronMan/Face Detection and Recognition/data/'
faceData =[]
labels = []
labelMap = {}
classId=0

for f in os.listdir(datasetPath):
    if f.endswith(".npy"):

        #X-values
        dataItem = np.load(datasetPath + f)
        print(dataItem.shape)

        faceData.append(dataItem)

        #Y-values
        m = dataItem.shape[0]
        target = classId * np.ones((m,))
        labels.append(target)
        labelMap[f[:-4]] = classId
        classId+=1
        


print(faceData)
print(labels)
print(labelMap)


X = np.concatenate(faceData,axis=0)
Y = np.concatenate(labels,axis=0).reshape((-1,1))

print(X.shape)
print(Y.shape)

#Algorithm
