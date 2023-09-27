#%%
import json
import math
from functools import reduce
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import numpy.fft as fft
import seaborn as sns
from skimage.measure import block_reduce
#%%
from magnetic_field import generate_fieldmap, B2hz
from utils import cal_snr


def ppm2p(r, len):
    r = 4.7 - r
    return int(((trnfreq * r) / (1 / (t_step * len))) + len / 2)

def plotppm( sig, ppm1, ppm2, rev, linewidth=0.3, linestyle='-'):
    """
    > This function takes in a signal, a starting ppm, an ending ppm, and a boolean value to determine whether or not to
    reverse the x-axis. It then plots the signal in the specified ppm range

    :param sig: the signal you want to plot
    :param ppm1: The lower bound of the ppm range you want to plot
    :param ppm2: The ppm value of the leftmost point of the plot
    :param rev: True if you want to reverse the x-axis (i.e. from left to right)
    :param linewidth: The width of the line
    :param linestyle: '-' for solid line, '--' for dashed line, '-.' for dash-dot line, ':' for dotted line, defaults to -
    (optional)
    :return: A plot of the real signal in the range of ppm1 to ppm2.
    """
    p1 = int(ppm2p(ppm1, len(sig)))
    p2 = int(ppm2p(ppm2, len(sig)))
    n = p2 - p1
    x = np.linspace(int(ppm1), int(ppm2), abs(n))
    sig = np.squeeze(sig)
    df = pd.DataFrame({'Real Signal (a.u.)': np.real(sig[p2:p1])})
    df['Frequency(ppm)'] = np.flip(x)
    g = sns.lineplot(x='Frequency(ppm)', y='Real Signal (a.u.)', data=df, linewidth=linewidth, linestyle=linestyle)
    if rev:
        plt.gca().invert_xaxis()
    return g




# reading block
# Loading the json file that contains the parameters for the simulation.
json_file_path = 'runs/exp1.json'
with open(json_file_path, 'r') as j:
    contents = json.loads(j.read())
run = contents["config"]
basisset = sio.loadmat(run["basis_dir"]).get('data')
mm = sio.loadmat(run["mm_dir"]).get('data')
# mm[0] = mm[0] - 1*fft.fftshift(fft.fft(mm, axis=0))[0]
basisset = np.concatenate((basisset, mm), axis=1)
for kk, num in enumerate(range(1,2)):
    nn = str(num).zfill(2)
    path2sub = run['saving_dir'] + "/" + run['subject_name'] + nn + "/"
    Path(path2sub).mkdir(parents=True,exist_ok=True)
    img = nib.load(run["image_dir"] + "/" + "IBSR_" + nn + "_segTRI_fill_ana.nii.gz")
    data = img.get_fdata()
    trnfreq = run['trnfreq']
    k = 128
    data=block_reduce(data[:,:,k,:], block_size=(8, 4, 1), func=np.max)
    FM = generate_fieldmap(data.shape[0],data.shape[1])


    # 0:void 1:csf  2:gray  3:white
    csf = data==1
    gray = data==2
    white = data==3
    out_mask = data==0
    # plt.imshow(data[:,70,:,0]==2)
    #parameters
    numofsig = basisset.shape[1]


    numOfSample = data.shape[0] * data.shape[1]
    numofp = run["siglen"]
    t_step = run["t_step"]
    t = np.arange(0, numofp) * t_step
    t = np.expand_dims(t, 1)
    met_name = run["met_name"]
    BW = 1/t_step
    f = np.linspace(-BW / 2, BW / 2, numofp)
    noise_level = run["noise_level"]
    noise = np.random.normal(0,noise_level,(numofp,numOfSample)) + (1j*np.random.normal(0,noise_level,(numofp,numOfSample)))

    # % NAA Glu Cr Tau mIns  [NAA Glu Cr Tau mIns Cho naag]# % 1.94 1.57 0.26 0.23
    # GMCon_avg = [9.8 9.3 9.65 3.01 4.7]/10;
    # WMCon_avg = [7.8 6.6 5.24 2.64 3.9]/10;
    # GMCon_sd = [1.4 2.1 9.65 0.23 0.8]/10;
    # WMCon_sd = [0.7 1.2 9.65 0.14 0.9]/10;
    # T2_avg = 90.25;
    # % avg of std 23.8 23 10.8 0.7 18
    # T2_sd = 15.12;

    # T2randGM = (T2_avg + T2_sd*randn(128,128,5)).*cast(GM(:,:),'double');
    # T2randWM = 1.15*(T2_avg + T2_sd*randn(128,128,5)).*cast(WM(:,:),'double');
    cmap="rocket"
    f_b = B2hz(FM,-10,10)
    sns.heatmap(f_b, cmap=cmap, square=False, yticklabels=False,xticklabels=False)
    plt.savefig(path2sub +  "field_map.svg")
    plt.show()

    sns.heatmap(f_b*(1-out_mask[:,:,0]), cmap=cmap, square=False, yticklabels=False,xticklabels=False)
    plt.savefig(path2sub + "masked_field_map.svg")
    plt.show()

    shift = np.random.normal(0, run['sim_param'][0], numOfSample) + f_b.flatten()
    sns.heatmap(shift.reshape((data.shape[0],data.shape[1])), cmap=cmap, square=False, yticklabels=False,xticklabels=False)
    plt.savefig(path2sub +  "frequency_map.svg")
    plt.show()
    freq = (-2 * math.pi * (shift) * t )
    ph = np.random.normal(0, run['sim_param'][1], numOfSample)
    dw = np.random.normal(run['sim_param'][2]*np.pi, run['sim_param'][4]*np.pi, numOfSample)
    dg = np.random.normal(run['sim_param'][3]*np.pi, run['sim_param'][5]*np.pi, numOfSample)
    # dc = np.random.normal(run['sim_param'][4], run['sim_param'][5], numOfSample)

    # ave_c = np.array([0.8, 0, 1.5, 9.65, 1.5, 1.25, 2.25, 1.1, 2.7, 8,
    #                   0.7, 5.6, 0.6, 9.8, 0.5, 0.6, 4.25, 1.5, 4, 0.44, 10])
    # sd_c = np.array([0.1, 0, 0.1, 1.16, 0.1, 0.1, 0.1, 0.4, 1.7, 1.8,
    #                   0.1, 1.1, 0.1, 1.4, 0.6, 0.1, 0.1, 0.1, 0.1, 0.07, 0.1])
    # ave_g = np.ones(len(run["met_name"]))
    # ave_w = np.ones(len(run["met_name"]))
    # for idx, met in enumerate(run["met_name"]):
    #     ave_g[idx] = run["min_c"][idx] + (np.random.rand()*(run["max_c"][idx]-run["min_c"][idx]))
    #     ave_w[idx] = run["ratio"][idx] * ave_g[idx]
    # sd_g = ave_g * 0.1
    # sd_w = ave_w * 0.05
    ave_g = np.asarray(run["mean_g"])
    ave_w = np.asarray(run["mean_w"])
    sd_g = np.asarray(run["sd_g"])
    sd_w = np.asarray(run["sd_w"])

    min_c = np.asarray(run['min_c'])
    max_c = np.asarray(run['max_c'])
    ratio = np.asarray(run["ratio"])

    # ave_w = (max_c)
    # ave_g = ave_w*ratio
    #
    # sd_g = 0.1*ave_w
    # sd_w = 0.1*ave_g

    data_mrsi= np.zeros((data.shape[0],data.shape[1],1, numofp),dtype='complex')

    # for k in range(0,data.shape[2]):
    # ampl_c = ave_c + np.multiply(np.random.random(size=(int(numOfSample), numofsig))-0.5, (sd_c))
    # signal_c = np.matmul(ampl_c[:, :], basisset[:, :].T)
    # signal_c = np.multiply(signal_c * np.expand_dims(np.exp(ph*1j),1), np.exp(freq*1j).T)
    # signal_c = np.multiply(signal_c, np.exp(-(1000/dc)*t).T)
    # signal_c = signal_c.reshape((data.shape[0],data.shape[1],numofp))
    # signal_c = signal_c * csf[:, :]

    ampl_g =  np.random.normal(ave_g, sd_g, size=(int(numOfSample), numofsig))
    ampl_g[ampl_g < 0] = 0
    signal_g = np.matmul(ampl_g[:, 0:21], basisset[0:2048, :].T)
    signal_g = signal_g * np.expand_dims(np.exp(ph * 1j), 1)
    # signal_g = np.multiply(signal_g, np.exp(freq*1j).T)
    signal_g = np.multiply(signal_g, np.exp(-(2*np.pi/dg)*t).T)
    signal_g = signal_g.reshape((data.shape[0],data.shape[1],numofp))

    signal_g = signal_g * gray[:, :]

    ampl_w = np.random.normal(ave_w, sd_w, size=(int(numOfSample), numofsig))
    ampl_w[ampl_w<0] = 0
    signal_w = np.matmul(ampl_w[:, 0:21], basisset[0:2048, :].T)
    signal_w = signal_w * np.expand_dims(np.exp(ph * 1j), 1)
    # signal_w = np.multiply(signal_w,np.exp(freq*1j).T)
    signal_w = np.multiply(signal_w,(np.exp(-(2*np.pi/dw)*t).T))
    signal_w = signal_w.reshape((data.shape[0],data.shape[1],numofp))
    signal_w = signal_w * white[:, :]
    # water
    numofwat = 5
    shifts_water = [-trnfreq*0.2, -trnfreq*0.1, 0, trnfreq*0.1, trnfreq*0.2]
    ph_water = [-45.0* (2*np.pi/180), 30.0* (1/180), 20.0* (2*np.pi/180), -60.0* (2*np.pi/180), 120.0* (2*np.pi/180)]
    d_water = [30, 25, 40, 20, 35]
    ampl_water = [30, 80, 180, 70, 40]
    signal_water_t = 0
    for i in range(0, numofwat):
        # shift = np.random.normal(shifts_water[i], 1, numOfSample) + f_b.flatten()
        shift = shifts_water[i] + f_b.flatten()
        freq_water = (2 * math.pi * (shift) * t)
        ampl_water = np.random.normal(ampl_water[i], 1, size=(int(numOfSample), 1))
        ampl_water[ampl_water < 0] = 0
        if run["gauss"] == False:
            signal_water_l = np.multiply(ampl_water * np.exp(ph_water[i]*1j),np.exp(freq_water*1j).T)
            signal_water_l = np.multiply(signal_water_l,np.exp(-(d_water[i])*t).T)* np.expand_dims(np.exp(ph * 1j), 1)
            signal_water_l = signal_water_l.reshape((data.shape[0],data.shape[1],numofp))
            # plotppm(fft.fftshift(fft.fft(signal_water_l[8, 8, :])), 4, 6, False)
            signal_water_t += signal_water_l
        else:
            signal_water_l = np.multiply(ampl_water * np.exp(ph_water[i] * 1j), np.exp(freq_water * 1j).T)
            signal_water_l = np.multiply(signal_water_l, np.exp(-(d_water[i]) * t).T)* np.expand_dims(np.exp(ph * 1j), 1)
            signal_water_l = signal_water_l.reshape((data.shape[0], data.shape[1], numofp))
            signal_water_g = np.multiply(ampl_water * np.exp(ph_water[i] * 1j), np.exp(freq_water * 1j).T)
            signal_water_g = np.multiply(signal_water_g, np.exp(-(d_water[i]) *2* (t**2)).T)* np.expand_dims(np.exp(ph * 1j), 1)
            signal_water_g = signal_water_g.reshape((data.shape[0], data.shape[1], numofp))
            signal_water_t += (signal_water_l / 2)
            signal_water_t += (signal_water_g / 2)
        plotppm(fft.fftshift(fft.fft(signal_water_l[8, 8, :])), 4, 6, True)
        plt.show()
    signal_water = signal_water_t * (white + gray)
    ampls = ampl_w.reshape((data.shape[0],data.shape[1],numofsig)) * white[:, :] \
            + ampl_g.reshape((data.shape[0],data.shape[1],numofsig)) * gray[:, :]
            # +ampl_c.reshape((data.shape[0],data.shape[1],numofsig)) * csf[:, :]

    data_mrsi_without_water = signal_w+signal_g + noise.reshape((data.shape[0],data.shape[1],numofp))
    snrs = cal_snr(data_mrsi_without_water[(white+gray)[:,:,0],:].T)
    print("snr mean:{} and std{}".format(snrs.mean,snrs.std))
    wss = [125]
    np.save(path2sub + "data_w",signal_water_t.reshape(-1,1024))
    for ws in wss:
        data_mrsi = data_mrsi_without_water + ws*signal_water
        data_mrsi_noisy  = data_mrsi
        plt.figure()
        i = 8
        j = 8
        plotppm(fft.fftshift(fft.fft(data_mrsi[i,j,:])),0,5,True)
        plt.figure()
        plotppm(fft.fftshift(fft.fft(data_mrsi_noisy[i,j,:])),0,5,True)
        plt.show()
        plotppm(fft.fftshift(fft.fft(data_mrsi_without_water[i,j,:])),0,5,True)
        plt.show()

        sns.heatmap(dw.reshape((data.shape[0],data.shape[1]))*(white[:,:,0])+(dg.reshape((data.shape[0],data.shape[1]))*(gray[:,:,0])), cmap=cmap, square=False, yticklabels=False,xticklabels=False)
        plt.savefig(path2sub + "_T2"  ".svg")
        plt.show()
        np.savez(path2sub + "data_ws_test"+str(ws)+"_"+ str(run["gauss"]),data_mrsi_noisy, data_mrsi_without_water, ampls,data,white,gray,csf,f_b,shift,dw,dg,ph,ave_g,ave_w)