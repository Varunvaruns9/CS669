import matplotlib.pyplot as plt
import numpy as np
from mlxtend.data import loadlocal_mnist
from pca import PCA, plot, reconstruct

k = 235
images, labels = loadlocal_mnist(images_path='./Data/train-images-idx3-ubyte',
                                    labels_path='./Data/train-labels-idx1-ubyte')
mat = images[labels == 0].astype('float64')
noise = np.random.normal(0, 128, mat.shape) * 0.2
mat += noise
pca_mat, mean, k_evectors = PCA(k, mat, use_perc=False)
recons_mat = reconstruct(pca_mat, mean, k_evectors)
plot(mat[0], 'Original')
plot(recons_mat[0], 'Reconstructed')
plt.show()
