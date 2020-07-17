#import functions as fs
#import time

#train, test = fs.init_data()
#trainSet, testSet = fs.data_ready2(train, test)

#result = fs.knn(trainSet, testSet, 10)
#acc, pre, rec, f1 = fs.calcMeasure(result)



import sklearn
import functions as fs
import numpy as np

train, test = fs.init_data()
trainSet, testSet = fs.data_ready2(train, test)
label = np.tile(np.arange(0,10),(300,1))

knn = KNeighborsClassifier(n_neighbors=10,
                           weights="distance",metric="euclidean")
knn.fit(trainSet, label.T.flatten())
result = knn.predict(testSet)
result = result.reshape(10,100).T
acc, pre, rec, f1 = fs.calcMeasure(result)
