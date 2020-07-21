import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

def init_data():
    with open('train.bin', 'rb') as f1:
        train =pickle.load(f1)

    with open('test.bin', 'rb') as f2:
        test =pickle.load(f2)
    return train, test

##def showTmpl(imsi):
##    for i in range(10):
##        plt.subplot(2,5,i+1),plt.imshow(imsi[i,::], 'gray')
##        plt.axis('off')
##    plt.show()

def data_ready1(train, test):
    trainSet = []
    testSet = []
    for i in range(10):
        trainSet.append(train[i][0:300])
        testSet.append(test[i][0:500])
    return trainSet, testSet

def createTmpl(trainSet):
    tmpl = np.zeros((28,28*10))
    for i in range(10):
        imsi = np.array(trainSet[i])
        tmpl[:,i*28:(i+1)*28] = np.mean(imsi, axis = 0)
    return tmpl

def tmplMatch(tmpl, testSet):
    result = np.zeros((100,10))

    for i in range(len(testSet)):
        for j in range(len(testSet[0])):
            imsiTest = np.tile(testSet[i][j], (1,10))
            error = np.abs(tmpl-imsiTest)
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(), error[:,56:84].sum(), error[:,84:112].sum(), error[:,112:140].sum(),
                 error[:,140:168].sum(), error[:,168:196].sum(), error[:,196:224].sum(), error[:,224:252].sum(), error[:,252:280].sum()]
            result[j,i] = np.argmin(errorSum)
    return result

def data_ready2(train, test):
    trainSetf = np.zeros((300*10, 28*28))
    testSetf = np.zeros((100*10, 28*28))

    for i in range(len(train)):
        for j in range(300):
            trainSetf[i*300+j,:] = train[i][j].flatten()

    for i in range(len(test)):
        for j in range(100):
            testSetf[i*100+j,:] = test[i][j].flatten()
    return trainSetf, testSetf


def knn(trainSet, testSet, k):

    trS1,trS2 = trainSet.shape
    teS1,teS2 = testSet.shape
    trS3 = int(trS1/10)
    teS3 = int(teS1/10)

    label = np.tile(np.arange(0,10), (teS3,1))
    result = np.zeros((teS3,10))

    for i in range(teS1):
        imsi = np.sum((trainSet - testSet[i,:])**2,axis=1)
        no = np.argsort(imsi)[0:k]
        hist, bins = np.histogram(no//trS3, np.arange(-0.5,10.5,1))
        result[i%teS3, i//teS3] = np.argmax(hist)
        
    return result



def feat1(trainSet, testSet):
    
    trS1 = len(trainSet)
    trS2 = len(trainSet[0])
    teS1 = len(testSet)
    teS2 = len(testSet[0])
    
    trainSetf = np.zeros((trS1 * trS2, 5))
    testSetf = np.zeros((teS1 * teS2, 5))

    for i in range(trS1):
        for j in range(trS2):
            
            imsi = trainSet[i][j]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi,1)
            
            imsi3 = np.cov(imsi)
            trainSetf[i*trS2+j,:] = np.array([imsi2[0], imsi2[1], imsi3[0,0],imsi3[0,1], imsi3[1,1]])

    for i in range(len(testSet)):
        for j in range(len(testSet[0])):
            imsi = testSet[i][j]
            imsi = np.where(imsi != 0)
            imsi2 = np.mean(imsi,1)
            imsi3 = np.cov(imsi)            
            testSetf[i*teS2+j,:] = np.array([imsi2[0], imsi2[1], imsi3[0,0],imsi3[0,1], imsi3[1,1]])
    return trainSetf, testSetf


def feat2(trainSet, testSet, dX):
    size = trainSet[0][0].shape[0]
    s = size-dX+1
    

    trS1 = len(trainSet)
    trS2 = len(trainSet[0])
    teS1 = len(testSet)
    teS2 = len(testSet[0])

    trainImsi = np.zeros((trS1 * trS2, s, s))
    testImsi = np.zeros((teS1 * teS2, s, s))
    trainSetf = np.zeros((trS1 * trS2, s*s))
    testSetf = np.zeros((teS1 * teS2, s*s))
              

    for i in range(len(trainSet)):
        for j in range(len(trainSet[0])):
            imsi = trainSet[i][j]
            for ii in range(s):
                for jj in range(s):
                    trainImsi[i*trS2+j, ii,jj] = imsi[ii:dX+ii,jj:dX+jj ].sum()
            trainSetf[i*trS2+j, :] = trainImsi[i*trS2+j, ::].flatten()

    for i in range(len(testSet)):
        for j in range(len(testSet[0])):
            imsi = testSet[i][j]
            for ii in range(s):
                for jj in range(s):
                    testImsi[i*teS2+j, ii,jj] = imsi[ii:dX+ii,jj:dX+jj ].sum()
            testSetf[i*teS2+j, :] = testImsi[i*teS2+j, ::].flatten()
            
    return trainSetf, testSetf  


def calcMeasure(result):

    # acc = (tp+tn)/ (tp+fn+fp+tn)
    # pre = tp/ (tp+fp)
    # rec = tp/ (tp+fn)
    # f1 = 2*pre*rec/(pre+rec)
    s1, s2 = result.shape
    label = np.tile(np.arange(0,s2), (s1,1))

    TP = []; TN = []; FN = []; FP = []
    for i in range(10):
##        TP.append(((result == label) & (label == i)).sum())
##        TN.append(((result == label) & (label != i)).sum())
####        FP.append(((result != label) & (result == i)).sum())
##        FN.append(((result != label) & (label == i)).sum())
        TP.append(((result == label) & (label == i)).sum())
        TN.append(((result != i) & (label != i)).sum())
        FP.append(((result != label) & (label == i)).sum())
        FN.append(((result == i) & (label != i)).sum())


    TP = np.array(TP); TN = np.array(TN); FN = np.array(FN); FP = np.array(FP)
    acc = (TP+TN)/(TP+TN+FP+FN)
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = 2*pre*rec/(pre+rec)

    return acc, pre, rec, f1




