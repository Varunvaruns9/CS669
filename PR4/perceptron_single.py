import matplotlib.pyplot as plt
import numpy as np


def perceptron_single(xarr_, yarr_):
    xarr, yarr = xarr_, yarr_
    xarr = np.concatenate((np.ones(xarr.shape[0])[:, np.newaxis], xarr), axis=1)
    w = np.random.rand(xarr.shape[1])
    correct = 0
    ind = 0
    itr = 0
    result = []
    while correct <= xarr.shape[0] and itr < 1000:
        if w.T @ xarr[ind] >= 0 and yarr[ind] == 0:
            w = w - 0.00001 * xarr[ind]
            correct = -1
        elif w.T @ xarr[ind] < 0 and yarr[ind] == 1:
            w = w + 0.00001 * xarr[ind]
            correct = -1
        correct += 1
        if ind == 0:
            bad = 0
            for j in range(xarr.shape[0]):
                if w.T @ xarr[j] >= 0 and yarr[j] == 0:
                    bad += 1
                elif w.T @ xarr[j] < 0 and yarr[j] == 1:
                    bad += 1
            result.append(bad/xarr.shape[0])
            itr += 1
        ind = (ind + 1) % xarr.shape[0]
    plt.plot(result)
    plt.show()
    return w


def predict(x, w):
    X = np.insert(x, 0, 1, axis=0)
    return 0 if w.T @ X >= 0 else 1
