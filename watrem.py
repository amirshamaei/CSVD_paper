import hlsvdpropy as hlsvdpro
import numpy as np
import matplotlib.pyplot as plt


def watrem( data, dt, n, f):
    npts = len(data)
    dwell = dt
    nsv_sought = n
    result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    nsv_found, singvals, freq, damp, ampl, phas = result
    # while np.isclose(singvals,0).any():
    #      nsv_sought -= np.isclose(singvals,0).sum()
    #      # if nsv_sought<0:
    #      #     nsv_sought -= 1
    #      result = hlsvdpro.hlsvd(data, nsv_sought, dwell)
    #      nsv_found, singvals, freq, damp, ampl, phas = result
    #     # fid = hlsvdpro.create_hlsvd_fids(result, npts, dwell, sum_results=True, convert=False)
    #     # plt.plot(np.fft.fftshift(np.fft.fft(data))[900:1500])
    #     # plt.plot(np.fft.fftshift(np.fft.fft(fid))[900:1500])
    #     # plt.show()
    # idx = np.where((np.abs(result[2]) < f) & (np.abs(result[3])>0.005))
    min_band = f[0]
    max_band = f[1]
    idx = []
    for min_, max_ in zip(min_band,max_band):
        idx.append(np.where(((result[2]) > min_) & ((result[2]) < max_)))
    idx = np.unique(np.concatenate(idx, 1))
    result_ = (len(idx), result[1][idx], result[2][idx], result[3][idx], result[4][idx], result[5][idx])
    fid = hlsvdpro.create_hlsvd_fids(result_, npts, dwell, sum_results=True, convert=False)
    # plt.plot(np.linspace(-(1/(2*dt)), (1/(2*dt)), len(data)), np.fft.fftshift(np.fft.fft(data)))
    # plt.plot(np.linspace(-(1/(2*dt)), (1/(2*dt)), len(data)),np.fft.fftshift(np.fft.fft(fid)))
    # plt.show()
    return data - fid

def watrem_batch(dataset, dt, n, f):
    dataset_ = np.zeros_like(dataset)
    for idx in range(len(dataset[0])):
        dataset_[:,idx] = watrem(dataset[:,idx],dt, n, f)
        if idx % 100 == 0:
            print(str(idx))
    return dataset_