import numpy as np
import matplotlib.pyplot as plt
import pickle


def init_data():
    with open('train.bin','rb') as f1:
        train=pickle.load(f1)
    with open('test.bin','rb') as f2:
        test=pickle.load(f2)
    return train, test

def createTmpl(trainSet):
    tmpl=np.zeros((28,28*10))
    for i in range(10):
        imsi=np.array(trainSet[i])
        tmpl[:,i*28:(i+1)*28]=np.mean(imsi, axis=0)
    return tmpl

def tmplMatch(tmpl, testSet):
    result=np.zeros((100,10))
    
    for i in range(len(testSet)):
        for j in range(len(testSet[0])):
            imsiTest=np.tile(testSet[i][j],(1,10))
            error=np.abs(tmpl-imsiTest)
            errorSum = [error[:,0:28].sum(), error[:,28:56].sum(), error[:,56:84].sum(), error[:,84:112].sum(), error[:,112:140].sum(),
                        error[:,140:168].sum(), error[:,168:196].sum(), error[:,196:224].sum(), error[:,224:252].sum(), error[:,252:280].sum()]
            result[j,i] = np.argmin(errorSum)
    return result


def data_ready2(train,test):
    trainSetf=np.zeros((300*10,28*28))
    testSetf=np.zeros((100*10,28*28))
    
    for i in range(len(train)):
        for j in range(300):
            trainSetf[i*300+j,:]=train[i][j].flatten()
    for i in range(len(test)):
        for j in range(100):
            testSetf[i*100+j,:]=test[i][j].flatten()
        return trainSetf,testSetf

def knn(trainSet, testSet,k):
    label=np.tile(np.arange(0,10),(100,1))
    result=np.zeros((100,10))
    
    
    for i in range(testSet.shape[0]):
        imsi=np.sum((trainSet-testSet[i,:])**2,axis=1)
        no=np.argsort(imsi)[0:k]
        hist,bins=np.histogram(no//300,np.arange(-0.5,10.5,1))
        result[i%100,i//100]=np.argmax(hist)
    return result
        
def calcMeasure(result):
    label=np.tile(np.arange(0,10),(100,1))
    TP=[]; TN=[]; FN=[]; FP=[]
    for i in range(10):
        TP.append(((result==label)&(label==i)).sum())
        TN.append(((result==label)&(label !=i)).sum())
        FP.append(((result!=label)&(label==i)).sum())
        FN.append(((result!=label)&(label !=i)).sum())
    TP=np.array(TP);TN=np.array(TN);FN=np.array(FN);FP=np.array(FP)
    acc=(TP+TN)/(TP+TN+FP+FN)
    pre=TP/(TP+FP)
    rec=TP/(TP+FN)
    f1=2*pre*rec/(pre+rec)

    return acc,pre,rec,f1
