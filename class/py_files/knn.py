from sklearn.neighbors import KNeighborsClassifier
import functions as fs
import numpy as np

train, test = fs.init_data()
trainSet, testSet = fs.data_ready2(train, test)
result = fs.knn(trainSet, testSet, 3000)

#label = np.tile(np.arange(0,10),(300,1))

#knn = KNeighborsClassifier(n_neighbors=10)
#knn.fit(trainSet, label.T.flatten())
#print(testSet)
#result = knn.predict(testSet)
#result = result.reshape(10,100).T
acc, pre, rec, f1 = fs.calcMeasure(result)
#print(acc, pre, rec, f1)
print(acc, end="\n\n")
print(f1)
