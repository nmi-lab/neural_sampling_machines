import numpy as np
from torchvision.datasets import CIFAR10


class ZCA(object):
    def __init__(self, regularization=1e-5, x=None):
        self.regularization = regularization
        if x is not None:
            self.fit(x)

    def fit(self, x):
        s = x.shape
        x = x.copy().reshape((s[0], np.prod(s[1:])))
        m = np.mean(x, axis=0)
        x -= m
        sigma = (x.T @ x) / x.shape[0]
        U, S, V = np.linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1./np.sqrt(S+self.regularization)))
        self.ZCA_mat = tmp @ U.T
        self.mean = m


def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)

    mean = X.mean(axis=0)
    X -= mean
    print("Data have been centered")

    # sigma = np.dot(X.T, X)
    sigma = np.cov(X.T)
    print("Computed covariance matrix")
    # d, E = np.linalg.eigh(sigma)
    U, S, V = np.linalg.svd(sigma)
    print("Computed eigenvalues")
    # D = np.diag(1. / np.sqrt(d + EPS))
    W = U.dot(np.diag(1.0 / np.sqrt(S + 1e-5))).dot(U.T)
    print("Computed diagonal matrix D")
    # W = np.dot(np.dot(E, D), E.T)
    print("Computed whitening matrix W")

    return W, mean


if __name__ == '__main__':
    # train data set
    m, k = 50000, 32 * 32 * 3
    raw = CIFAR10("/share/data", train=True)
    data = raw.data.astype('f')
    # imgs = -127.5 + data.reshape(m, k) / 128.0
    imgs = data.reshape(m, k) / 255.0

    w, mu = zca_whiten(imgs)
    np.save("zca_mat_train_1", w)
    X = np.dot(imgs - mu, w)
    X = ((X - X.min()) * (1/(X.max() - X.min()) * 255)).astype('uint8')
    np.save("white_cifar10_train_1", X.reshape(50000, 32, 32, 3))

    # test data set
    m, k = 10000, 32 * 32 * 3
    raw = CIFAR10("/share/data", train=False)
    data = raw.data.astype('f')
    # imgs = -127.5 + data.reshape(m, k) / 128.0
    imgs = data.reshape(m, k) / 255.0

    w, mu = zca_whiten(imgs)
    np.save("zca_mat_test_1", w)
    X = np.dot(imgs - mu, w)
    X = ((X - X.min()) * (1/(X.max() - X.min()) * 255)).astype('uint8')
    np.save("white_cifar10_test_1", X.reshape(10000, 32, 32, 3))
