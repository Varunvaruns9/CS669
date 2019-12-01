import numpy as np

from perceptron_single import perceptron_single


def perceptron_multi(xarr_, yarr_, classes):
    xarr, yarr = xarr_, yarr_
    xx = xarr[yarr == 0]
    xx1 = xarr[yarr != 0]
    xtrain = np.append(xx, xx1, axis=0)
    ytrain = np.append(np.ones(xx.shape[0], dtype=int), np.zeros(xx1.shape[0], dtype=int))
    w1 = perceptron_single(xtrain, ytrain)
    xx = xarr[yarr == 1]
    xx1 = xarr[yarr != 1]
    xtrain = np.append(xx, xx1, axis=0)
    ytrain = np.append(np.ones(xx.shape[0], dtype=int), np.zeros(xx1.shape[0], dtype=int))
    w2 = perceptron_single(xtrain, ytrain)
    xx = xarr[yarr == 2]
    xx1 = xarr[yarr != 2]
    xtrain = np.append(xx, xx1, axis=0)
    ytrain = np.append(np.ones(xx.shape[0], dtype=int), np.zeros(xx1.shape[0], dtype=int))
    w3 = perceptron_single(xtrain, ytrain)
    return w1, w2, w3


def predictm(x, w1, w2, w3):
    X = np.insert(x, 0, 1, axis=0)
    a1 = (w1.T @ X)
    a2 = (w2.T @ X) 
    a3 = (w3.T @ X) 
    if a1 > a2 and a1 > a3:
        return 0
    elif a2 > a1 and a2 > a3:
        return 1
    return 2
