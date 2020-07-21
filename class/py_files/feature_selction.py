import functions as fs
import numpy as np
import matplotlib.pyplot as plt
import time
import pdb

t1 = time.time()
train, test = fs.init_data()
trainSet, testSet = fs.data_ready1(train, test)
trainSetf1, testSetf1 = fs.feat1(trainSet, testSet)
#trainSetf1, testSetf1 = fs.feat2(trainSet, testSet, dX=3)
result = fs.knn(trainSetf1, testSetf1, k=1)
acc, pre, rec, f1 = fs.calcMeasure(result)
t2 = time.time()
print(t2-t1)
print(acc.mean())
print(f1.mean())
