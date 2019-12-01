import numpy as np

from perceptron_single import perceptron_single
from plotter import plotter_single


def split(data):
    np.random.shuffle(data)
    split = int(len(data)*0.75)
    train = data[:split]
    test = data[split:]
    return train, test

data1 = np.loadtxt('Data/Class1.txt')
data2 = np.loadtxt('Data/Class2.txt')
train1, test1 = split(data1)
train2, test2 = split(data2)


xtrain = np.append(train1, train2, axis=0)
ytrain = np.append(np.zeros(train1.shape[0], dtype=int), np.ones(train2.shape[0], dtype=int))
xtest = np.append(test1, test2, axis=0)
ytest = np.append(np.zeros(test1.shape[0], dtype=int), np.ones(test2.shape[0], dtype=int))
w = perceptron_single(xtrain, ytrain)
plotter_single(xtest, ytest, w)
