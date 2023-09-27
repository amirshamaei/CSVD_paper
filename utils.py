import json
from datetime import time

# import mrs_denoising
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.fft import fft,fftshift
from scipy import linalg
import nibabel as nib

def savefig(saving_dir, save=True, plt_tight=True):
    if plt_tight:
        plt.tight_layout()
    if save:
        plt.savefig(saving_dir  + ".svg", format="svg")
        plt.savefig(saving_dir  + " .png", format="png", dpi=1200)
    plt.show()

    # %%
def tic():
    global start_time
    start_time = time.time()

def toc(saving_dir,name):
    elapsed_time = (time.time() - start_time)
    print("--- %s seconds ---" % elapsed_time)
    timingtxt = open(saving_dir + ".txt", 'w')
    timingtxt.write(name)
    timingtxt.write("--- %s ----" % elapsed_time)
    timingtxt.close()

def RMSE(x,x_hat):
    return np.sqrt(np.mean(np.power(np.abs(x-x_hat),2)))

def cal_snr(data, endpoints=128):
    return np.abs(data[0, :]) / np.std(data.real[-(68 + endpoints):-68, :], axis=0)

def cal_snrf(data_f,endpoints=128):
    return np.max(np.abs(data_f), 0) / (np.std(data_f.real[0:endpoints, :],axis=0))

def ppm2p(r, len,trnfreq=123.32,t_step=0.25e-3):
    r = 4.7 - r
    return int(((trnfreq * r) / (1 / (t_step * len))) + len / 2)

def ppm2f(r,trnfreq):
    return r * trnfreq

def MSE(d1,d2):
    return np.mean(np.square((d2-d1)),axis=0)

def load_data(path):
    data_ = np.load(path)
    return [data_[x] for x in data_]

def plotppm(sig, ppm1, ppm2, rev, trnfreq, t_step, linewidth=0.3, linestyle='-',FFT=True, mode='abs'):
    if FFT:
        sig = fftshift(fft(sig))
    else:
        pass
    p1 = int(ppm2p(ppm1, len(sig),trnfreq,t_step))
    p2 = int(ppm2p(ppm2, len(sig),trnfreq,t_step))
    n = p2 - p1
    x = np.linspace(int(ppm1), int(ppm2), abs(n))
    sig = np.squeeze(sig)
    if mode=='abs':
        df = pd.DataFrame({'Real Signal (a.u.)': np.abs(sig[p2:p1])})
    else:
        df = pd.DataFrame({'Real Signal (a.u.)': sig[p2:p1].real})
    df['Frequency(ppm)'] = np.flip(x)
    g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df, linewidth=linewidth, linestyle=linestyle)
    plt.tick_params(axis='both', labelsize=18)
    if rev:
        plt.gca().invert_xaxis()
    return g

def plotsppm(sig, ppm1, ppm2, rev, trnfreq, t_step, linewidth=0.3, linestyle='-'):
    sig = fftshift(fft(sig,axis=1),axes=1)
    p1 = int(ppm2p(ppm1, sig.shape[1],trnfreq,t_step))
    p2 = int(ppm2p(ppm2, sig.shape[1],trnfreq,t_step))
    n = p2 - p1
    x = np.linspace(int(ppm1), int(ppm2), abs(n))
    sig = np.squeeze(sig)
    df = pd.DataFrame(sig[:,p2:p1].T.real)
    df['Frequency(ppm)'] = np.flip(x)
    df_m = df.melt(id_vars='Frequency(ppm)',value_name='Real Signal (a.u.)')
    g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df_m, linewidth=linewidth, linestyle=linestyle)
    sns.despine()
    if rev:
        plt.gca().invert_xaxis()
    return g

def fillppm(y1, y2, ppm1, ppm2, rev, trnfreq, t_step, alpha=.1, color='red'):
    p1 = int(ppm2p(ppm1, len(y1),trnfreq,t_step))
    p2 = int(ppm2p(ppm2, len(y1),trnfreq,t_step))
    n = p2 - p1
    x = np.linspace(int(ppm1), int(ppm2), abs(n))
    plt.fill_between(np.flip(x), y1[p2:p1].real,
                     y2[p2:p1].real, alpha=alpha, color=color)
    if rev:
        plt.gca().invert_xaxis()

def add_noise(data,noise_level):
    noise = np.random.normal(0, noise_level, (data.shape)) + (
                1j * np.random.normal(0, noise_level, (data.shape)))
    return data + noise

def sub_mean(data,mean=None):
    if mean is None:
        mean = np.mean(data,1)
    return data - np.matmul(np.expand_dims(mean,1),np.expand_dims(np.ones(data.shape[1]),0)),mean

def add_mean(data,mean):
    return data + np.matmul(np.expand_dims(mean,1),np.expand_dims(np.ones(data.shape[1]),0))


def PSNR(X,X_hat):
    MSE = np.mean(np.power(X-X_hat,2),0)
    MAX = np.max(np.abs(X),0)
    PSNR = 20*np.log10((MAX)/(np.sqrt(MSE)))
    PSNR[PSNR == -np.inf] = 0
    PSNR[PSNR == np.inf] = 0
    return PSNR

def SSIM(X,X_hat):
    X = np.abs(X)
    X_hat = np.abs(X_hat)
    mio_x = np.mean(X)
    mio_x_hat = np.mean(X_hat)
    var_x = np.var(X)
    var_x_hat = np.var(X_hat)
    cov_ = np.cov(X,X_hat)
    c1 = 10
    c2 = 5
    cov_ = np.cov(X.flatten(), X_hat.flatten())[0,1]
    return ((2*mio_x*mio_x_hat+1)*(2*cov_+3))/((mio_x**2+mio_x_hat**2+1)*(var_x+var_x_hat+3))

def total_PSNR(X,X_hat):
    psnr_real = PSNR(X.real, X_hat.real)
    psnr_imag = PSNR(X.imag, X_hat.imag)
    # plt.imshow(psnr_real.reshape(shape))
    psnr_mean = np.mean(psnr_real + psnr_imag)
    psnr_std = np.std(psnr_real + psnr_imag)
    return psnr_mean, psnr_std

def skewness(data):
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    return (3*(mean-median))/std
def accuracy_analysis(gt_data, denoised_data,df,i):
    gt_data_f = fftshift(fft(gt_data,axis=0),axes=0)[ppm2p(4.5,len(gt_data)):ppm2p(0,len(gt_data)),:]
    denoised_data_f = fftshift(fft(denoised_data,axis=0),axes=0)[ppm2p(4.5,len(gt_data)):ppm2p(0,len(gt_data)),:]
    psnr_mean, psnr_sd = total_PSNR(gt_data_f, denoised_data_f)
    df.loc[i, "psnr_mean"] = psnr_mean
    df.loc[i, "psnr_std"] = psnr_sd
    df.loc[i, "var"] = np.var((gt_data_f)- denoised_data_f) / 2
    df.loc[i, "cov"] = np.cov(np.abs(gt_data_f).flatten() - np.abs(denoised_data_f).flatten())
    df.loc[i, "cor"] = np.real(np.vdot((gt_data_f).flatten(), (denoised_data_f).flatten())) / (
                np.linalg.norm((gt_data_f).flatten()) * np.linalg.norm((denoised_data_f).flatten()))
    df.loc[i, "mean"] = np.real(np.mean(gt_data_f - denoised_data_f))
    df.loc[i, "skew"] = skewness(np.real(gt_data_f - denoised_data_f))
    df.loc[i, "max"] = np.max(np.abs(gt_data_f - denoised_data_f))
    df.loc[i, "SSIM"] = SSIM((gt_data_f), denoised_data_f)
    df.loc[i, "RMSE"] = RMSE((gt_data_f), denoised_data_f)
    df.loc[i,"spectral_ent"] = spectral_entropy(gt_data_f,denoised_data_f)
    snrs = cal_snrf(
        fftshift(fft((gt_data - denoised_data), axis=0), axes=0))
    df.loc[i, "snr_res_mean"] = np.mean(snrs)
    df.loc[i, "snr_res_sd"] = np.std(snrs)
    return df



def load_data(path):
    data_ = np.load(path)
    return [data_[x] for x in data_]

def reshape_data(data):
    data_ = data.reshape(-1, data.shape[2]).T
    return data_

def reshape_data_back(data,shape):
    data_ = data.T.reshape(shape)
    return data_

def svd_data(data):
    U, S, V = linalg.svd(data, full_matrices=False)
    return U, S, V

def find_rank(S_,data,method='mp'):
    S = S_.copy()
    if method == 'mp':
        limit = max_sv_from_mp(np.var((data[-256:,:])),data.shape)
        r = np.where(S<limit)[0][0]
        if r == 0:
            r = 1
    if method == 'sure':
        r = sure_(200, S, np.std((data[-256:,:].real)), data.shape)
    if method == 'svht':
        limit = svht(np.std((data[-256:,:])),data.shape,S)
        r = np.where(S<limit)[0][0]
        if r == 0:
            r = 1
    if method == 'svht_unk':
        limit = svht(None,data.shape,S)
        r = np.where(S<limit)[0][0]
        if r == 0:
            r = 1
    return r

# from lo-rank-denoising-tools by Will Clarke
def max_sv_from_mp(data_var, data_shape):
    """Calculate the upper Marchenko–Pastur limit for a pure noise
    Matrix of defined shape and data variance.

    :param data_var: Noise variance
    :type data_var: float
    :param data_shape: 2-tuple of data dimensions.
    :type data_shape: tuple
    :return: Upper MP singular value limit
    :rtype: float
    """
    # Make sure dimensions agree with utils.lsvd
    # utils.lsvd will always move largest dimension to first dim.
    if data_shape[1] > data_shape[0]:
        data_shape = (data_shape[1], data_shape[0])

    c = data_shape[1] / data_shape[0]
    sv_lim = data_var * (1 + np.sqrt(c))**2
    return (sv_lim * data_shape[0])**0.5

# from lo-rank-denoising-tools by Will Clarke -> modified
def sure_(rank, S, var, data_shape):
    def div_HARD(r, S, Nx, Nt):
        z = (Nx + Nt) * r - r**2
        for idx in range(r):
            for jdx in range(r, S.size):
                z += 2*S[jdx]**2 / (S[idx]**2 - S[jdx]**2)
        return z

    def SURE(r, S, v, Nx, Nt):
        return -2 * Nx * Nt * v \
               + np.sum(S[r:]**2) \
               + 4 * v * div_HARD(r, S, Nx, Nt)

    sure = np.zeros((data_shape[0],))
    rank = np.arange(1, data_shape[0] + 1)
    for r in rank:
        sure[r - 1] = SURE(r,
                           S,
                           var,
                           data_shape[0],
                           data_shape[-1])

    thresh_SURE = rank[np.argmin(sure[0:200])]
    return thresh_SURE

# Eq. derived from:
# The Optimal Hard Threshold for Singular Values is 4/√3
def svht(var,data_shape,S):
    if data_shape[1] > data_shape[0]:
        n = data_shape[1]
        m = data_shape[0]
    else:
        n = data_shape[0]
        m = data_shape[1]
    beta = m / n
    if var is None:
        omega = 0.56 * np.power(beta,3) - 0.95 * (beta**2) + (1.82*beta) + 1.43
        y_med = np.median(S)
        limit = omega * y_med
    else:
        lanbda = np.sqrt(2*(beta+1) + ((8*beta)/((beta+1)+np.sqrt((beta**2) + 14 * beta + 1))))
        limit = lanbda * np.sqrt(n) * var
    return limit

def inv_left(U,r,beta):
    Ut = U[:,0:r]
    l = U.shape[0]
    D = np.matmul(Ut,Ut.conj().T)-np.eye(l,dtype=np.complex128)
    Z = linalg.pinv(np.eye(l,dtype=np.complex128) + beta*np.matmul((D).conj().T,(D)))
    return Z

def inv_left_masked(U, r, beta,mask):
    Ut = U[:, 0:r]
    l = U.shape[0]
    D = np.matmul(Ut, Ut.conj().T) - np.eye(l, dtype=np.complex128)
    Z = linalg.pinv(mask + beta * np.matmul((D).conj().T, (D)))
    return Z

def spectral_power(x):
    return np.abs(x)**2

def shanon_ent(x):
    x = norm_power(x)
    return -1*np.sum(x*np.log(x))

def norm_power(x):
    return x/np.max(x)

def spectral_entropy(gt_data, denoised_data):
    gt_data_f = spectral_power(gt_data-denoised_data)
    info = shanon_ent(gt_data_f)
    return info

def listToDict(lstA, lstB):
    zippedLst = zip(lstA, lstB)
    op = dict(zippedLst)
    return op

def denoise(data,Z):
    d = np.matmul(Z,data)
    # if plot == True:
    #     fig, axs = plt.subplots(2,2)
    #     axs[0, 0].imshow(d.T.reshape(data_won.shape)[:,:,0].__abs__(),cmap='viridis',aspect='auto')
    #     axs[0, 1].imshow(data.T.reshape(data_won.shape)[:, :, 0].__abs__(),cmap='viridis',aspect='auto')
    #     axs[1, 0].imshow(d.T.reshape(data_won.shape)[:, :, 0].__abs__()-data_won[:, :, 0].__abs__(),cmap='viridis',aspect='auto')
    #     axs[1, 1].imshow(data_won[:, :, 0].__abs__(),cmap='viridis',aspect='auto')
    return d

def super_res(data,Z):
    d = np.matmul(Z,data)
    return d

def plot_map(data_wn,data_won,data_denoised):
    fig, axs = plt.subplots(2,2)
    a = axs[0, 0].imshow(data_wn[:,:,0].__abs__(),cmap='viridis',aspect='auto')
    axs[0, 0].set_title("noisy")
    fig.colorbar(a,ax = axs[0, 0])
    a = axs[0, 1].imshow(data_won[:, :, 0].__abs__(),cmap='viridis',aspect='auto')
    axs[0, 1].set_title("clean")
    fig.colorbar(a,ax = axs[0, 1])
    a = axs[1, 0].imshow(data_denoised[:, :, 0].__abs__()-data_won[:, :, 0].__abs__(),cmap='viridis',aspect='auto')
    mse = np.mean(np.power(data_denoised[:, :, 0].__abs__()-data_won[:, :, 0].__abs__(),2))
    axs[1, 0].set_title(str(mse))
    fig.colorbar(a,ax = axs[1, 0])
    a = axs[1, 1].imshow(data_denoised[:, :, 0].__abs__(),cmap='viridis',aspect='auto')
    axs[1, 1].set_title("denoised")
    fig.colorbar(a,ax = axs[1, 1])

def write_nifti(data, dwelltime, transmitter_frequency_mhz, nucleus_str, save_path):
    metadata_dict = {'SpectrometerFrequency': [transmitter_frequency_mhz, ],
                     'ResonantNucleus': [nucleus_str, ]}

    json_meta = json.dumps(metadata_dict)
    newobj = nib.nifti2.Nifti2Image(data, np.eye(4))

    # Write new header with the dwell time
    pixDim = newobj.header['pixdim']
    pixDim[4] = dwelltime
    newobj.header['pixdim'] = pixDim

    # newobj.header['dim']
    # Set version information
    newobj.header['intent_name'] = b'mrs_v0_3'

    # Write extension with ecode 44
    extension = nib.nifti1.Nifti1Extension(44, json_meta.encode('UTF-8'))
    newobj.header.extensions.append(extension)

    # Form nifti obj and write
    nib.save(newobj,  save_path)

def similarityMat(signal1, signal2, p1, p2):
    signal1 = np.fft.fftshift(np.fft.fft((signal1), axis=0),axes=0)
    signal2 = np.fft.fftshift(np.fft.fft((signal2), axis=0), axes=0)
    sMat = np.zeros((np.shape(signal1)[1]))
    for i in range(0, np.shape(signal1)[1]):
            sMat[i] = np.real(np.vdot(signal1[p1:p2, i], signal2[p1:p2, i])) / (
                        np.linalg.norm(signal1[p1:p2, i]) * np.linalg.norm(signal2[p1:p2, i]))
    return sMat

def integ_fit(signal, p1, p2):
    signal = np.fft.fftshift(np.fft.fft((signal), axis=0), axes=0)
    integ = np.mean(np.abs(signal[p1:p2,:]),axis=0)
    return integ

def mask_it(data,mask):
    return np.ma.masked_where(np.squeeze(mask) != True, data)

def bland_altman_plot(data1, data2, fontsize=18,*args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='-',linewidth=3)
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    plt.tick_params(axis='both', labelsize=fontsize)

def modified_bland_altman_plot(data1, data2, gt=None,c_map=None, fontsize=18, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference
    if gt is not None:
        ax = plt.scatter(gt, diff,cmap='Spectral', *args, **kwargs)
    else:
        ax = plt.scatter(range(0, data1.shape[0]), diff ,cmap='Spectral', *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='-',linewidth=3)
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--',linewidth=2)
    sns.despine()
    if c_map != None:
        plt.set_cmap(c_map)
        cb = plt.colorbar()
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=fontsize)
    plt.tick_params(axis='both', labelsize=fontsize)

    return ax
