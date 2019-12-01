import matplotlib.pyplot as plt
import numpy as np

from perceptron_multi import predictm
from perceptron_single import predict


def plotter_single(xarr, yarr, w):
    xmax = -1e9
    xmin = 1e9
    ymax = -1e9
    ymin = 1e9
    xmax = max(xmax, xarr[:,0].max())
    xmin = min(xmin, xarr[:,0].min())
    ymax = max(ymax, xarr[:,1].max())
    ymin = min(ymin, xarr[:,1].min())
    val = xarr[yarr == 0]
    plt.scatter(val[:,0], val[:,1], label="Class {}".format(1))
    val = xarr[yarr == 1]
    plt.scatter(val[:,0], val[:,1], label="Class {}".format(2))
    hx = (xmax - xmin) / 100
    hy = (ymax - ymin) / 100
    x_min, x_max = xmin - 20*hx, xmax + 20*hx
    y_min, y_max = ymin - 20*hy, ymax + 20*hy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
    Z = np.array([predict(x, w) for x in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    plt.legend()
    plt.show()


def plotter_multi(xarr, yarr, w1, w2, w3):
    xmax = -1e9
    xmin = 1e9
    ymax = -1e9
    ymin = 1e9
    xmax = max(xmax, xarr[:,0].max())
    xmin = min(xmin, xarr[:,0].min())
    ymax = max(ymax, xarr[:,1].max())
    ymin = min(ymin, xarr[:,1].min())
    val = xarr[yarr == 0]
    plt.scatter(val[:,0], val[:,1], label="Class {}".format(1))
    val2 = xarr[yarr == 1]
    plt.scatter(val2[:,0], val2[:,1], label="Class {}".format(2))
    val3 = xarr[yarr == 2]
    plt.scatter(val3[:,0], val3[:,1], label="Class {}".format(3))
    hx = (xmax - xmin) / 100
    hy = (ymax - ymin) / 100
    x_min, x_max = xmin - 20*hx, xmax + 20*hx
    y_min, y_max = ymin - 20*hy, ymax + 20*hy
    xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))
    Z = np.array([predictm(x, w1, w2, w3) for x in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    plt.legend()
    plt.show()
