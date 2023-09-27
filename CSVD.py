import hlsvdpropy
import numpy as np
from matplotlib import pyplot as plt
from optht import optht
from scipy.signal import butter, lfilter

from watrem import watrem


class CSVD:
    def __init__(self, data, dt):
        self.data = data
        self.U, self.s, self.V = np.linalg.svd(self.data, full_matrices=False)
        self.dt = dt


    def remove(self, rank, frequency_band, n_comp , sigma = None):
        """
        The function takes in the data, the rank of the data matrix, the sampling rate, the number of components to remove, and the
        frequency band to remove. It then calculates the rank of the data, and then removes the specified number of
        components from the specified frequency band

        :param rank: the rank of the SVD decomposition. If you set it to 'auto', it will use the optimal rank as determined
        by the optimal hard thresholding algorithm
        :param frequency_band: a tuple of the form (fmin, fmax)
        :param n_comp: number of components to remove
        :param sigma: the threshold for the singular values. If None, the default is the median of the singular values
        :return: The signal after the removal of the noise.
        """
        if rank == 'auto':
            limit = svht(sigma, self.data.shape,self.s)
            rank = np.where(self.s < limit)[0][0]
            if rank == 0:
                rank = 1
        U_ = self.U.copy()

        for i in range(0, rank):
            # SNR = np.max(np.abs(U_[:, i]))/np.std(U_[-32:, i])
            # if SNR > 3:
            U_[:, i] = watrem(self.U[:, i], self.dt, n_comp, frequency_band)

            # plt.plot(np.fft.fftshift(np.fft.fft(self.U[900:1500, i]))[:])
            # plt.plot(np.fft.fftshift(np.fft.fft(U_[900:1500, i]))[:])
            # # plt.ylim(-2, 2)
            # plt.show()
            # n_comp = int(1.05*n_comp)
        sigs = np.dot(U_[:, :len(self.s)] * self.s, self.V)
        return sigs


def svht(var, data_shape, S):
    """
    > The function takes in the singular values of a matrix and returns the threshold value for the singular values
    Eq. derived from: The Optimal Hard Threshold for Singular Values is 4/√3
    :param var: the variance of the noise in the data. If you don't know it, you can set it to None
    :param data_shape: the shape of the data matrix
    :param S: The singular values of the matrix
    :return: The limit of the singular values.
    """
    if data_shape[1] > data_shape[0]:
        n = data_shape[1]
        m = data_shape[0]
    else:
        n = data_shape[0]
        m = data_shape[1]
    beta = m / n
    if var is None:
        omega = 0.56 * np.power(beta, 3) - 0.95 * (beta ** 2) + (1.82 * beta) + 1.43
        y_med = np.median(S)
        limit = omega * y_med
    else:
        lanbda = np.sqrt(2 * (beta + 1) + ((8 * beta) / ((beta + 1) + np.sqrt((beta ** 2) + 14 * beta + 1))))
        limit = lanbda * np.sqrt(n) * var
    return limit

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y