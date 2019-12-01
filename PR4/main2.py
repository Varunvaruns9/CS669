import numpy as np

from perceptron_multi import perceptron_multi
from plotter import plotter_multi


def split(data):
    np.random.shuffle(data)
    split = int(len(data)*0.75)
    train = data[:split]
    test = data[split:]
    return train, test

data1 = np.loadtxt('Data/Class1.txt')
data2 = np.loadtxt('Data/Class2.txt')
data3 = np.loadtxt('Data/Class3.txt')
train1, test1 = split(data1)
train2, test2 = split(data2)
train3, test3 = split(data3)

xtrain = np.append(train1, np.append(train2, train3, axis=0), axis=0)
ytrain = np.append(np.zeros(train1.shape[0], dtype=int), np.append(np.ones(train2.shape[0], dtype=int),
                    np.full(train3.shape[0], 2, dtype=int)))
xtest = np.append(test1, np.append(test2, test3, axis=0), axis=0)
ytest = np.append(np.zeros(test1.shape[0], dtype=int), np.append(np.ones(test2.shape[0], dtype=int),
                    np.full(test3.shape[0], 2, dtype=int)))
w1, w2, w3 = perceptron_multi(xtrain, ytrain, 3)
plotter_multi(xtest, ytest, w1, w2, w3)
