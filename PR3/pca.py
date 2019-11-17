import matplotlib.pyplot as plt
import numpy as np


def PCA(k, x, use_perc=False):
    mean = np.sum(x, axis=0) / x.shape[0]
    mean_corr_x = x - mean
    cov = mean_corr_x.transpose() @ mean_corr_x
    evalues, evectors = np.linalg.eigh(cov)
    idx = evalues.argsort()[::-1]
    evalues = evalues[idx]
    evectors = evectors[:,idx]
    if use_perc:
        tot = 0
        ind = 0
        while tot*100 < k and ind < 784:
            tot += evalues[ind] / (sum(evalues))
            ind += 1
        return mean_corr_x @ evectors[:,:ind], mean, evectors[:,:ind]
    return mean_corr_x @ evectors[:,:k], mean, evectors[:,:k]

def reconstruct(x, mean, evectors):
    return (x @ evectors.transpose()) + mean if x.shape[1] else np.zeros((2, 784))

def plot(img, label):
    plt.figure()
    plt.imshow(img.reshape([28, 28]))
    plt.title(label)
