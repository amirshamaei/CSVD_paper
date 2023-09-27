import numpy as np
from numpy.fft import ifft
import matplotlib.pyplot as plt

def l2_sup(csi,water,beta = 0.0001):
    N = csi.shape


    water_inv = np.linalg.inv(np.eye(N[0]) + beta*(water @ water.T))

    # csi = np.reshape(csi, (N[0], N[1]*N[2]))
    csi_f = fftshift(fft(csi, axis=0), axes=0)
    csiws = water_inv @ csi_f
    csiws_t = (ifft(fftshift(csiws,axes=0), axis=0))
    # csi = np.reshape(csi, (N[0], N[1], N[2]))

    # csi = np.transpose(csi, (2, 1, 0))

    # csiws = csi

    return csiws_t


import numpy as np
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt


def generator_water(SW, Datapoints, Zerofill, wat_frq_range, damp, damp_range, Watnum, echoposition):
    # Parameters
    NF2 = Datapoints  # data points
    NNW = Watnum  # number of water components
    WR = wat_frq_range  # water frequency dispersion range
    t = np.arange(0, NF2 / SW, 1/SW) # time points

    # generate fid
    fid = np.zeros((NF2, NNW), dtype=complex)

    for ii in range(NNW):
        # frequence, damping facter, phase, and amplitude
        feq_wat = 2 * WR * (np.random.rand(1) - 0.5)
        dam_wat = (np.random.rand(1)) * (damp_range - damp) + damp
        pha_wat = (np.random.rand(1) - 0.5) * 2 * np.pi
        amp_wat = 100  # *np.exp(-feq_wat/WR)

        # signal model
        fid[:, ii] = amp_wat * np.exp(-dam_wat * np.abs(t-(echoposition*NF2/SW))) * np.exp(
            -1j * (feq_wat * 2 * np.pi * t - pha_wat))

    # wat = (fftshift(ifft(fid, Zerofill)))
    wat = fftshift(fft(fid, Zerofill,axis=0), axes=0)
    # plot figures
    # plt.subplot(2,1,1)
    # plt.plot(np.real(fid))

    # plt.subplot(2,1,2)
    # plt.plot(np.real(wat))

    return wat