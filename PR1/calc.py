import numpy as np

def calc_mean_cov(train):
    mean = np.zeros(2)
    size = len(train)
    for row in train:
        mean = mean + (row / size)

    cov = np.zeros((2, 2))
    for row in train:
        cov = cov + (np.matmul([[x] for x in (row - mean)], [row - mean]) / size)

    return mean, cov
