# This is a sample Python script.

# Press Alt+Shift+X to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
from pathlib import Path

from l2 import l2_sup, generator_water

from utils import bland_altman_plot
import mat73 as mat73
import pandas as pd
import scipy.io as sio
from sklearn.linear_model import LinearRegression
from scipy import stats
from CSVD import CSVD
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, ifft
import numpy as np

from utils import plotppm, savefig, plotsppm, write_nifti, similarityMat, ppm2p, integ_fit, load_data, MSE, mask_it
from watrem import watrem, watrem_batch
from optht import optht
import seaborn as sns
# %%
def tic():
    global start_time
    start_time = time.time()

def toc(name):
    elapsed_time = (time.time() - start_time)
    print("--- %s seconds ---" % elapsed_time)
    timingtxt = open(name + ".txt", 'w')
    timingtxt.write(name)
    timingtxt.write("--- %s ----" % elapsed_time)
    timingtxt.close()
    return elapsed_time

def testBiggaba():
    sns.set_style("white")
    dir = "testBigGABA/"
    Path(dir).mkdir(parents=True,exist_ok=True)
    dataset = mat73.loadmat("data/SfidsCoiledPRESS.mat").get("fids4096")
    dt = 0.25e-3
    tic()
    sigs = CSVD(np.conj(dataset), dt).remove('auto',50,10)
    sigma = np.std(dataset[-68:, :])
    hlsvd_r = False
    # x = watrem(dataset[:, 11 * 32 + 14], dt, 30, 50)
    if hlsvd_r:
        tic()
        dataset_hsvd = watrem_batch(dataset, dt, 10, 50)
        toc(dir + "testBG_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = np.conjugate(np.load(dir+"hsvd.npy"))

    for soi in [100,400,700,1000,2000,2500,39]:
        plotppm(dataset[:, soi], 1, 6, False, 127.798258, dt)
        plotppm(dataset_hsvd[:, soi], 1, 6, False, 127.798258, dt)
        plotppm(sigs[:, soi], 1, 6, True, 127.798258, dt)
        plt.ylim(-0.002,0.007)
        plt.legend(["orginal","hlsvd","csvd"])
        savefig(dir+"_soi={}".format(soi))


    # # x_hsvd = watrem(dataset[:, soi], 0.25e-3, 10, 50)
    # plotppm(dataset[:, soi], 1, 5, False, 127.798258, dt)
    # plotppm(dataset_hsvd[:, soi], 1, 5, False, 127.798258, dt)
    # plotppm(sigs[:, soi], 1, 5, True, 127.798258, dt)
    # plt.legend(["orginal","hlsvd","csvd"])
    # plt.show()

def testsimulatedCSI_rank_analyze(ws=25,hlsvd_r = False):
    sns.set_style("white")
    dir = "testsimulatedCSI/" + str(ws) + "_test/"
    Path(dir).mkdir(parents=True,exist_ok=True)

    data_mrsi, data_mrsi_wow,_, _, white, gray, _, _, _, _, _, _, _, _ = load_data(
        "data/golden brain/ismrm challenge/HomoDeus_01/" + "data_ws_test"+str(ws)+"_False"+".npz")
    dataset_ = data_mrsi.reshape((-1,2048)).T
    # dataset_ = sio.loadmat("data/fid1.mat").get("fid1")
    mask = white + gray
    # L2 = np.conj(sio.loadmat(dir + "l2rslt.mat").get("mrs1")) * np.expand_dims(mask, 2)
    sigma = np.std(dataset_[-68:,mask.reshape(-1)])
    zf = 2048
    dt = 0.25e-3
    tsf = 123.32
    dataset = np.zeros((zf,dataset_.shape[1]),dtype=np.complex128)
    dataset[0:dataset_.shape[0],:] = dataset_
    tic()
    csvd = CSVD(dataset[:,mask.reshape(-1)], dt)


    ranks = [5, 10, 15, 20]
    ks = [10, 20, 30,40,50,60,70,80]

    mse = np.zeros((len(ranks), len(ks)))
    std = np.zeros((len(ranks), len(ks)))
    time = np.zeros((len(ranks), len(ks)))

    for i, rank in enumerate(ranks):
        for j, k in enumerate(ks):
            sigs = np.zeros_like(dataset)
            tic()
            sigs[:,mask.reshape(-1)] = csvd.remove(rank,([-tsf*0.5],[tsf*0.5]),k,sigma=sigma)
            el_time = toc(dir + f"testCSI_csvd_{rank}_{k}")
            sMat = MSE(np.abs(sigs), np.abs(data_mrsi_wow.reshape((-1, 2048)).T))
            mse[i, j] = np.mean(sMat[mask.reshape(-1)])  # Compute MSE
            std[i, j] = np.std(sMat[mask.reshape(-1)])  # Compute STD
            time[i, j] = el_time  # Get time

    print("MSE:")
    print(mse)
    print("STD:")
    print(std)
    print("TIME:")
    print(time)
    np.savez(dir + 'testsimulatedCSI_rank_analyze.npz', [mse, std, time])
    fig, axs = plt.subplots(1, 3, figsize=(12, 2))
    y = [5, 10, 15, 20]
    x = [10, 20, 30, 40, 50, 60, 70, 80]
    for i, matrix in enumerate([mse, std, time]):
        ax = axs[i]
        # im = sns.heatmap(matrix, norm=LogNorm(),ax=axs[i])
        im = ax.imshow(matrix,norm=LogNorm())

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(len(x)))
        ax.set_yticks(np.arange(len(y)))
        ax.set_xticklabels(x)
        ax.set_yticklabels(y)

        if i == 0:
            ax.set_title('MSE')
        elif i == 1:
            ax.set_title('STD')
        else:
            ax.set_title('Duration')

    fig.tight_layout()
    savefig(dir + "analyse")
    plt.show()

def testsimulatedCSI(ws=25,hlsvd_r = False):
    sns.set_style("white")
    dir = "testsimulatedCSI/" + str(ws) + "_test/"
    Path(dir).mkdir(parents=True,exist_ok=True)

    data_mrsi, data_mrsi_wow,_, _, white, gray, _, _, _, _, _, _, _, _ = load_data(
        "data/golden brain/ismrm challenge/HomoDeus_01/" + "data_ws_test"+str(ws)+"_False"+".npz")
    dataset_ = data_mrsi.reshape((-1,2048)).T
    # dataset_ = sio.loadmat("data/fid1.mat").get("fid1")
    mask = white + gray
    # L2 = np.conj(sio.loadmat(dir + "l2rslt.mat").get("mrs1")) * np.expand_dims(mask, 2)
    sigma = np.std(dataset_[-68:,mask.reshape(-1)])
    zf = 2048
    dt = 0.25e-3
    tsf = 123.32
    dataset = np.zeros((zf,dataset_.shape[1]),dtype=np.complex128)
    dataset[0:dataset_.shape[0],:] = dataset_
    tic()
    csvd = CSVD(dataset[:,mask.reshape(-1)], dt)
    sigs = np.zeros_like(dataset)
    sigs[:,mask.reshape(-1)] = csvd.remove('auto',([-tsf*0.5],[tsf*0.5]),60,sigma=sigma)
    toc(dir + "testCSI_csvd")
    tic()
    # water = np.load("data/golden brain/ismrm challenge/HomoDeus_01/" + "data_w" + ".npy")
    csi_ws_l2 = np.zeros_like(dataset)
    water = generator_water(1/dt, 2048, 2048, 50, 0, 100, 2048, 0.5)
    if ws == 5:
        beta = 0.0001
    if ws == 25:
        beta = 0.001
    if ws == 125:
        beta = 0.01

    csi_ws_l2[:,mask.reshape(-1)] = l2_sup(dataset[:,mask.reshape(-1)], water.real,beta)
    toc(dir + "testCSI_l2")

    dataset_hsvd = np.zeros_like(dataset)

    # x = watrem(dataset[:, 11 * 32 + 14], dt, 30, 50)
    if hlsvd_r:
        tic()
        dataset_hsvd[:,mask.reshape(-1)] = watrem_batch(dataset[:,mask.reshape(-1)], dt, 30, tsf*0.5)
        toc(dir + "testCSI_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = np.load(dir+"hsvd.npy")

    savefig(dir + "sigma")
    plt.plot(csvd.s[0:20])
    savefig(dir + "S")
    for jj in range(0, 5):
        plotppm(csvd.U[:, jj], 1, 6, False, tsf, dt)

    plotppm(csvd.U[:, 6], 1, 6, True, tsf, dt)
    sns.despine()
    plt.legend(range(0, 6))
    savefig(dir + "U")
    width = 32
    height = width
    sns.set_style("white")
    dataset_ = np.reshape(dataset.T, (width, height, zf))
    sigs_ = np.reshape(sigs.T, (width, height, zf))
    csi_ws_l2_ = np.reshape(csi_ws_l2.T, (width, height, zf))
    dataset_hsvd_ = np.reshape(dataset_hsvd.T, (width, height, zf))


    p1 = ppm2p(0, len(dataset_hsvd), tsf, dt)
    p2 = ppm2p(9, len(dataset_hsvd), tsf, dt)

    # sMat1 = similarityMat(dataset_hsvd,data_mrsi_wow.reshape((-1,2048)).T,p2,p1)
    # sMat2 = similarityMat(sigs,data_mrsi_wow.reshape((-1,2048)).T,p2,p1)
    cmap = 'autumn'
    sMat1 = MSE(np.abs(dataset_hsvd),np.abs(data_mrsi_wow.reshape((-1,2048)).T))
    sMat2 = MSE(np.abs(sigs),np.abs(data_mrsi_wow.reshape((-1,2048)).T))
    sMat3 = MSE(np.abs(ifft(csi_ws_l2)), np.abs(data_mrsi_wow.reshape((-1, 2048)).T))
    max = np.max((sMat1[mask.reshape(-1)], sMat2[mask.reshape(-1)],sMat3[mask.reshape(-1)]))
    min = np.min((sMat1[mask.reshape(-1)], sMat2[mask.reshape(-1)],sMat3[mask.reshape(-1)]))
    fontsize = 16
    plt.imshow(mask_it(sMat1.reshape((width, height)),np.squeeze(mask)),  cmap=cmap)

    cb = plt.colorbar()
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=fontsize)
    savefig(dir + "sMat_hsvd")
    plt.imshow(mask_it(sMat2.reshape((width, height)),np.squeeze(mask)), cmap=cmap)

    cb = plt.colorbar()
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=fontsize)
    savefig(dir + "sMat_csvd")
    plt.imshow(mask_it(sMat3.reshape((width, height)),np.squeeze(mask)),  cmap=cmap)

    cb = plt.colorbar()
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=fontsize)
    savefig(dir + "sMat_l2")

    write_nifti(np.expand_dims((dataset_), 2), dt, tsf, "1H", dir + "orginal")
    write_nifti(np.expand_dims((dataset_hsvd_), 2), dt, tsf, "1H", dir + "hsvd")
    write_nifti(np.expand_dims((sigs_), 2), dt, tsf, "1H", dir + "csvd")
    write_nifti(np.expand_dims((csi_ws_l2_), 2), dt, tsf, "1H", dir + "l2")

    with open(dir + "MSE.txt",'w') as f:
        f.write("MSE of hsvd {} with std:{}\n".format(np.mean(sMat1[mask.reshape(-1)]),np.std(sMat1[mask.reshape(-1)])))
        f.write("MSE of csvd {} with std:{}\n".format(np.mean(sMat2[mask.reshape(-1)]),np.std(sMat2[mask.reshape(-1)])))
        f.write("MSE of l2 {} with std:{}\n".format(np.mean(sMat3[mask.reshape(-1)]), np.std(sMat3[mask.reshape(-1)])))
        f.write("{}\n".format(stats.ttest_ind(sMat1[mask.reshape(-1)], sMat2[mask.reshape(-1)], equal_var=False)))
        f.write("{}\n".format(stats.ttest_ind(sMat2[mask.reshape(-1)], sMat3[mask.reshape(-1)], equal_var=False)))
        f.close()


    x = range(8*2,10*2)
    y = range(3*2,5*2)
    mode = 'abs'
    for i, j in zip(x,y):
        soi = [i,j]
        plotppm(data_mrsi[soi[0], soi[1], :], 1, 6, False, tsf, dt,mode = mode)
        plotppm(data_mrsi_wow[soi[0],soi[1],:], 1, 6, False, tsf, dt,mode = mode)
        plotppm(sigs_[soi[0],soi[1],:], 1, 6, False, tsf, dt,mode = mode)
        plotppm(dataset_hsvd_[soi[0], soi[1], :], 1, 6, True, tsf, dt,mode = mode)
        plotppm(csi_ws_l2_[soi[0], soi[1], :], 1, 6, False, tsf, dt, mode=mode,FFT=True)
        plt.legend(["orginal","prginal_wow", "csvd", "hlsvd", "l2"])
        plt.title("x = {}, y = {}".format(i, j))
        plt.ylim(-5000, 40000)
        savefig(dir + "comparison" + str(soi))


        plotppm(data_mrsi_wow[soi[0],soi[1],:] - sigs_[soi[0],soi[1],:], 1, 6, False, tsf, dt,mode=mode)
        plotppm(data_mrsi_wow[soi[0], soi[1], :] - dataset_hsvd_[soi[0], soi[1], :], 1, 6, False, tsf, dt,mode=mode)
        plotppm(data_mrsi_wow[soi[0], soi[1], :] - csi_ws_l2_[soi[0], soi[1], :], 1, 6, True, tsf, dt, mode=mode,FFT=True)
        plt.legend(["csvd", "hlsvd", "l2"])
        plt.title("x = {}, y = {}".format(i, j))
        plt.ylim(-1000, 10000)
        savefig(dir + "comparison_gt" + str(soi))


        plotppm(data_mrsi_wow[soi[0], soi[1], :], 3, 6, False, tsf, dt)
        plotppm(sigs_[soi[0], soi[1], :], 3, 6, False, tsf, dt)
        plotppm(dataset_hsvd_[soi[0], soi[1], :], 3, 6, False, tsf, dt)
        plotppm(csi_ws_l2_[soi[0], soi[1], :], 3, 6, True, tsf, dt, FFT=True)
        plt.legend(["wow", "csvd", "hlsvd", "l2"])
        plt.title("x = {}, y = {}".format(i,j))
        plt.ylim(-1000, 25000)
        savefig(dir + "zoom comparison" + str(soi))

def testCSIrudy(volume):
    sns.set_style("white")
    dir = "testCSIrudy/"+volume+"/"
    Path(dir).mkdir(parents=True,exist_ok=True)
    dataset_ = np.conj(sio.loadmat("data/"+volume+".mat").get("data"))*np.exp(np.pi*1j)
    mask = np.full((16, 16), False)
    mask[5:11,5:11] = True
    sigma = np.std(dataset_[-68:,:])
    zf = 4096
    dt = 0.00025
    tsf = 123.25
    dataset = np.zeros((zf,dataset_.shape[1]),dtype=np.complex128)
    dataset[0:dataset_.shape[0],:] = dataset_
    tic()
    csvd = CSVD(dataset, dt)
    sigs = csvd.remove('auto',tsf*0.5,50,sigma=sigma)
    toc(dir + "csvd")

    hlsvd_r = False
    if hlsvd_r:
        tic()
        dataset_hsvd = watrem_batch(dataset, dt, 30, tsf*0.5)
        toc(dir + "testCSI_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = np.load(dir+"hsvd.npy")
    plt.plot(csvd.s)
    savefig(dir + "S")
    for jj in range(0, 5):
        plotppm(csvd.U[:, jj], 1, 6, False, tsf, dt)

    plotppm(csvd.U[:, 6], 1, 6, True, tsf, dt)
    sns.despine()
    plt.legend(range(0, 6))
    savefig(dir + "U")

    p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    sMat1 = MSE(np.abs(dataset_hsvd),np.abs(sigs.real))

    max = np.max(sMat1[mask.reshape(-1, order='F') == 1])

    plt.imshow(sMat1.reshape((16, 16), order='F') * mask,vmax=max)
    plt.colorbar()
    savefig(dir + "sMat_hsvd")



    p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    dataset_hsvd_fit_1_4 = integ_fit(dataset_hsvd, p2, p1)
    sigs_fit_1_4 = integ_fit(sigs, p2, p1)

    sns.set_style("white")
    sns.set_palette("dark")
    df = pd.DataFrame()
    df["sigs_fit_1_4"] = sigs_fit_1_4[mask.reshape(-1,order='F') == 1]
    df["dataset_hsvd_fit_1_4"] = dataset_hsvd_fit_1_4[mask.reshape(-1,order='F') == 1]

    rslt = pd.DataFrame(columns = ["R2", "intrcept", "coef"],index=["1.8", "2.8", "4"],dtype=object)
    model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                                     df["sigs_fit_1_4"].values.reshape((-1, 1)))
    rslt.iloc[0] = [model.score(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),df["sigs_fit_1_4"].values.reshape((-1, 1)))
                    ,model.intercept_[0]
                    ,model.coef_[0][0]]

    rslt.to_csv(dir+"regression")
    with open(dir+"t_test.txt",'w') as f:
        f.write("{}\n".format(stats.ttest_ind(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                              df["sigs_fit_1_4"].values.reshape((-1, 1)), equal_var=False)))

    max = df[["sigs_fit_1_4", "dataset_hsvd_fit_1_4"]].max().max()
    ax = sns.lmplot(x="sigs_fit_1_4", y="dataset_hsvd_fit_1_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
                    line_kws={'lw': 1}, ci=False)
    ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    plt.tick_params(axis='both', labelsize=18)
    savefig(dir + "hsvd_scater_1_4")

    bland_altman_plot(df['sigs_fit_1_4'], df['dataset_hsvd_fit_1_4'])
    savefig(dir + "hsvd_bland_altman_1_4")

    sns.set_style("white")
    diff = np.abs(sigs_fit_1_4 - dataset_hsvd_fit_1_4)[mask.reshape(-1, order='F') == 1]
    argsort = diff.argsort()
    d = np.where(mask.reshape(-1, order='F') == True)[0]
    for id,soi in enumerate([d[argsort[0]],d[argsort[0]],d[argsort[0]],d[argsort[-3]],d[argsort[-2]],d[argsort[-1]]]):
        mode = 'abs'
        plotppm(dataset[:, soi], 1, 6, False, tsf, dt, mode=mode)
        plotppm(dataset_hsvd[:, soi], 1, 6, False, tsf, dt, mode=mode)
        plotppm(sigs[:, soi], 1, 6, True, tsf, dt, mode=mode)
        plt.ylim(-0.05,0.15)
        plt.legend(["orginal","hlsvd","csvd"])
        plt.title(str(soi))
        savefig(dir+"_soi={}".format(id))

        plotppm(dataset_hsvd[:, soi], 3, 6, False, tsf, dt)
        plotppm(sigs[:, soi], 3, 6, True, tsf, dt)
        plt.legend(["hlsvd", "csvd"])
        plt.title(str(soi))
        savefig(dir+"zoomed_soi={}".format(id))

    weidth = 16
    height = 16
    dataset_ = np.reshape(dataset.T,  (weidth, height, zf), order='F')
    sigs_ = np.reshape(sigs.T, (weidth, height, zf), order='F')
    dataset_hsvd_ = np.reshape(dataset_hsvd.T, (weidth, height, zf), order='F')

    write_nifti(np.expand_dims(np.conj(dataset_),2),dt, tsf, "1H", dir + "orginal")
    write_nifti(np.expand_dims(np.conj(dataset_hsvd_), 2), dt, tsf, "1H", dir + "hsvd")
    write_nifti(np.expand_dims(np.conj(sigs_), 2), dt, tsf, "1H", dir + "csvd")

def test_GN():
    sns.set_style("white")
    dir = "testGN/"

    # id1 = "1R56OSSVCGVWmDavzhfNdvSMgQUyYuJdl"
    # url = "https://drive.google.com/uc?id=" + id1
    # output = 'data/3dcsi.mat'
    # gdown.download(url, output, quiet=False)

    Path(dir).mkdir(parents=True,exist_ok=True)
    dataset_ = np.conj(sio.loadmat("data/"+"3dcsi"+".mat").get("X"))
    mask = np.full((16, 16, 8), False)
    mask[4:12,4:12] = True
    sigma = np.std(dataset_[-68:,:])
    zf = 2048
    dt = 1/2404
    tsf = 123.262314
    dataset = np.zeros((zf,dataset_.shape[1]*dataset_.shape[2]),dtype=np.complex128)
    dataset[0:dataset_.shape[0],:] = dataset_.reshape(dataset_.shape[0],-1)
    tic()
    sigs = dataset.copy()
    csvd = CSVD(dataset[:,mask.reshape(-1)], dt)
    sigs[:,mask.reshape(-1)] = csvd.remove('auto',([-tsf*3],[tsf*0.52]),70,sigma=sigma)
    toc(dir + "csvd")

    hlsvd_r = False
    if hlsvd_r:
        tic()
        dataset_hsvd = watrem_batch(dataset, dt, 30, ([-tsf*3],[tsf*0.5]))
        toc(dir + "testCSI_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = np.load(dir+"hsvd.npy")
    plt.plot(csvd.s)
    savefig(dir + "S")
    for jj in range(0, 5):
        plotppm(csvd.U[:, jj], 1, 6, False, tsf, dt)

    plotppm(csvd.U[:, 6], 1, 6, True, tsf, dt)
    sns.despine()
    plt.legend(range(0, 6))
    savefig(dir + "U")

    p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    sMat1 = MSE(np.abs(dataset_hsvd),np.abs(sigs.real))
    sMat1 = sMat1.reshape((16,16,8))
    max = np.max(sMat1[mask == 1])

    plt.imshow((sMat1* mask)[:,:,4] ,vmax=max)
    plt.colorbar()
    savefig(dir + "sMat_hsvd")



    p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    dataset_hsvd_fit_1_4 = integ_fit(dataset_hsvd, p2, p1)
    sigs_fit_1_4 = integ_fit(sigs, p2, p1)

    sns.set_style("white")
    sns.set_palette("dark")
    df = pd.DataFrame()
    df["sigs_fit_1_4"] = sigs_fit_1_4[mask.reshape(-1) == 1]
    df["dataset_hsvd_fit_1_4"] = dataset_hsvd_fit_1_4[mask.reshape(-1) == 1]

    rslt = pd.DataFrame(columns = ["R2", "intrcept", "coef"],index=["1.8", "2.8", "4"],dtype=object)
    model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                                     df["sigs_fit_1_4"].values.reshape((-1, 1)))
    rslt.iloc[0] = [model.score(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),df["sigs_fit_1_4"].values.reshape((-1, 1)))
                    ,model.intercept_[0]
                    ,model.coef_[0][0]]

    rslt.to_csv(dir+"regression")
    with open(dir+"t_test.txt",'w') as f:
        f.write("{}\n".format(stats.ttest_ind(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                              df["sigs_fit_1_4"].values.reshape((-1, 1)), equal_var=False)))

    max = df[["sigs_fit_1_4", "dataset_hsvd_fit_1_4"]].max().max()
    ax = sns.lmplot(x="sigs_fit_1_4", y="dataset_hsvd_fit_1_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
                    line_kws={'lw': 1}, ci=False)
    ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    plt.tick_params(axis='both', labelsize=18)
    savefig(dir + "hsvd_scater_1_4")

    bland_altman_plot(df['sigs_fit_1_4'], df['dataset_hsvd_fit_1_4'])
    savefig(dir + "hsvd_bland_altman_1_4")

    sns.set_style("white")
    diff = np.mean(np.abs(dataset_hsvd - sigs),0)[mask.reshape(-1) == 1]
    print(diff.mean())
    argsort = diff.argsort()
    d = np.where(mask.reshape(-1) == True)[0]
    # for id,soi in enumerate([d[argsort[-10]],d[argsort[-100]],d[argsort[-200]],d[argsort[100]],d[argsort[200]],d[argsort[300]]]):
    soi = 725
    mode = 'real'
    plotppm(dataset[:, soi], 1, 6, False, tsf, dt, mode=mode)
    plotppm(dataset_hsvd[:, soi], 1, 6, False, tsf, dt, mode=mode)
    plotppm(sigs[:, soi], 1, 6, True, tsf, dt, mode=mode)
    # plt.ylim(-0.05,0.15)
    plt.legend(["orginal","hlsvd","csvd"])
    plt.title(str(soi))
    savefig(dir+"_soi={}".format(soi))

        # plotppm(dataset_hsvd[:, soi], 3, 6, False, tsf, dt)
        # plotppm(sigs[:, soi], 3, 6, True, tsf, dt)
        # plt.legend(["hlsvd", "csvd"])
        # plt.title(str(soi))
        # savefig(dir+"zoomed_soi={}".format(id))

    weidth = 16
    height = 16
    dataset_ = np.reshape(dataset.T,  (weidth, height, 8,zf))
    sigs_ = np.reshape(sigs.T, (weidth, height, 8,zf))
    dataset_hsvd_ = np.reshape(dataset_hsvd.T, (weidth, height, 8,zf))

    write_nifti(np.conj(dataset_),dt, tsf, "1H", dir + "orginal")
    write_nifti(np.conj(dataset_hsvd_), dt, tsf, "1H", dir + "hsvd")
    write_nifti(np.conj(sigs_), dt, tsf, "1H", dir + "csvd")

def testCSI():
    sns.set_style("white")
    dir = "testCSI/"
    cmap = 'autumn'
    Path(dir).mkdir(parents=True,exist_ok=True)
    dataset_ = sio.loadmat("data/fid1.mat").get("fid1")
    mask = sio.loadmat(dir+"/meta_mask.mat").get("meta_mask")
    # L2 = np.conj(sio.loadmat(dir + "l2rslt.mat").get("mrs1")) * np.expand_dims(mask, 2)
    sigma =np.std(dataset_[-32:,:])/np.sqrt(2)
    zf = 2048
    dt = 1/2000
    tsf = 127.798258
    dataset = np.zeros((zf,dataset_.shape[1]),dtype=np.complex128)
    dataset[0:dataset_.shape[0],:] = dataset_
    tic()
    csvd = CSVD(dataset, dt)
    sigs = csvd.remove('auto',([-tsf*0.5,tsf*2.8],[tsf*0.5,tsf*4.7]),30,sigma=sigma)
    toc(dir + "testCSI_csvd")
    tic()
    hlsvd_r = True
    if hlsvd_r:
        tic()
        dataset_hsvd = watrem_batch(dataset, dt, 30, ([-tsf*0.5,tsf*2.8],[tsf*0.5,tsf*4.7]))
        toc(dir + "testCSI_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = np.load(dir+"hsvd.npy")


    plt.plot(csvd.s[0:50])
    savefig(dir + "S")
    for jj in range(0, 5):
        plotppm(csvd.U[:, jj], 1, 6, False, tsf, dt)

    plotppm(csvd.U[:, 6], 1, 6, True, tsf, dt)
    sns.despine()
    plt.legend(range(0, 6))
    savefig(dir + "U")

    p1 = ppm2p(0, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    sMat1 = similarityMat(dataset_hsvd,sigs,p2,p1)
    # sMat2= similarityMat(L2.reshape((-1,zf),order='F').T,sigs,p2,p1)
    # sMat2[np.isnan(sMat2)] = 0
    # max = np.max(
    #     (sMat1[mask.reshape(-1, order='F') == 1], sMat2[mask.reshape(-1, order='F') == 1]))
    plt.imshow(mask_it(sMat1.reshape((32, 24), order='F') , mask),cmap = cmap)
    plt.colorbar()
    savefig(dir + "sMat_hsvd")


    # plt.imshow(sMat2.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "sMat_l2")

    p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    dataset_hsvd_fit_1_8 = integ_fit(dataset_hsvd, p2, p1)
    sigs_fit_1_8 = integ_fit(sigs, p2, p1)
    # l2_fit_1_8 = integ_fit(L2.reshape((-1,zf),order='F').T, p2, p1)

    # p1 = ppm2p(2.8, len(dataset_hsvd),tsf, dt)
    # p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    # dataset_hsvd_fit_2_8 = integ_fit(dataset_hsvd, p2, p1)
    # sigs_fit_2_8 = integ_fit(sigs, p2, p1)
    # # l2_fit_2_8 = integ_fit(L2.reshape((-1,zf),order='F').T, p2, p1)
    #
    # p1 = ppm2p(4, len(dataset_hsvd),tsf, dt)
    # p2 = ppm2p(5.2, len(dataset_hsvd), tsf, dt)
    # dataset_hsvd_fit_4 = integ_fit(dataset_hsvd, p2, p1)
    # sigs_fit_4 = integ_fit(sigs, p2, p1)
    # # l2_fit_4 = integ_fit(L2.reshape((-1,zf),order='F').T, p2, p1)

    max = np.max((dataset_hsvd_fit_1_8[mask.reshape(-1,order='F') == 1], sigs_fit_1_8[mask.reshape(-1,order='F') == 1]))
    plt.imshow(mask_it(dataset_hsvd_fit_1_8.reshape((32, 24), order='F') , mask), cmap=cmap,vmax=max)
    plt.colorbar()
    savefig(dir + "hsvd_1_8")
    plt.imshow(mask_it(sigs_fit_1_8.reshape((32, 24), order='F'),mask),cmap=cmap,vmax=max)
    plt.colorbar()
    savefig(dir + "csvd_1_8")
    # plt.imshow(l2_fit_1_8.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "l2_1_8")

    # max = np.max((dataset_hsvd_fit_2_8[mask.reshape(-1,order='F') == 1], sigs_fit_2_8[mask.reshape(-1,order='F') == 1]))
    # plt.imshow(dataset_hsvd_fit_2_8.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "hsvd_2_8")
    # plt.imshow(sigs_fit_2_8.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "csvd_2_8")
    # # plt.imshow(l2_fit_2_8.reshape((32, 24), order='F') * mask,vmax=150)
    # # plt.colorbar()
    # # savefig(dir + "l2_2_8")
    #
    # max = np.max((dataset_hsvd_fit_4[mask.reshape(-1,order='F') == 1],sigs_fit_4[mask.reshape(-1,order='F') == 1]))
    # plt.imshow(dataset_hsvd_fit_4.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "hsvd_4")
    # plt.imshow(sigs_fit_4.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "csvd_4")
    # plt.imshow(l2_fit_4.reshape((32, 24), order='F') * mask,vmax=150)
    # plt.colorbar()
    # savefig(dir + "l2_4")

    sns.set_style("whitegrid")
    sns.set_palette("dark")

    df = pd.DataFrame()
    df["sigs_fit_1_8"] = sigs_fit_1_8[mask.reshape(-1,order='F') == 1]
    # df["l2_fit_1_8"] = l2_fit_1_8[mask.reshape(-1,order='F') == 1]
    df["dataset_hsvd_fit_1_8"] = dataset_hsvd_fit_1_8[mask.reshape(-1,order='F') == 1]
    # df["sigs_fit_2_8"] = sigs_fit_2_8[mask.reshape(-1,order='F') == 1]
    # # df["l2_fit_2_8"] = l2_fit_2_8[mask.reshape(-1,order='F') == 1]
    # df["dataset_hsvd_fit_2_8"] = dataset_hsvd_fit_2_8[mask.reshape(-1,order='F') == 1]
    # df["sigs_fit_4"] = sigs_fit_4[mask.reshape(-1,order='F') == 1]
    # # df["l2_fit_4"] = l2_fit_4[mask.reshape(-1,order='F') == 1]
    # df["dataset_hsvd_fit_4"] = dataset_hsvd_fit_4[mask.reshape(-1,order='F') == 1]

    rslt = pd.DataFrame(columns = ["R2", "intrcept", "coef"],index=["1.8"],dtype=object)
    model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_1_8"].values.reshape((-1, 1)),
                                                     df["sigs_fit_1_8"].values.reshape((-1, 1)))
    rslt.iloc[0] = [model.score(df["dataset_hsvd_fit_1_8"].values.reshape((-1, 1)),df["sigs_fit_1_8"].values.reshape((-1, 1)))
                    ,model.intercept_[0]
                    ,model.coef_[0][0]]
    rslt.to_csv(dir+"regression")
    with open(dir+"t_test.txt",'w') as f:
        f.write("{}\n".format(stats.ttest_ind(df["dataset_hsvd_fit_1_8"].values.reshape((-1, 1)),
                                              df["sigs_fit_1_8"].values.reshape((-1, 1)), equal_var=False)))

    # model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_2_8"].values.reshape((-1, 1)),
    #                                                  df["sigs_fit_2_8"].values.reshape((-1, 1)))
    # rslt.iloc[1] = [model.score(df["dataset_hsvd_fit_2_8"].values.reshape((-1, 1)),df["sigs_fit_2_8"].values.reshape((-1, 1)))
    # ,model.intercept_[0]
    # ,model.coef_[0][0]]
    #
    # model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_4"].values.reshape((-1, 1)),
    #                                                  df["sigs_fit_4"].values.reshape((-1, 1)))
    # rslt.iloc[2] = [model.score(df["dataset_hsvd_fit_4"].values.reshape((-1, 1)),df["sigs_fit_4"].values.reshape((-1, 1)))
    # ,model.intercept_[0],
    # model.coef_[0][0]]

    ax = sns.lmplot(x="sigs_fit_1_8", y="dataset_hsvd_fit_1_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
                    line_kws={'lw': 1}, ci=False)
    ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    savefig(dir + "hsvd_scater_1_8")

    sns.set_style("white")
    bland_altman_plot(df['sigs_fit_1_8'], df['dataset_hsvd_fit_1_8'])
    savefig(dir + "hsvd_bland_altman_1_4")

    # ax = sns.lmplot(x="sigs_fit_1_8", y="l2_fit_1_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "l2_scater_1_8")

    # ax = sns.lmplot(x="sigs_fit_2_8", y="dataset_hsvd_fit_2_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "hsvd_scater_2_8")

    # ax = sns.lmplot(x="sigs_fit_2_8", y="l2_fit_2_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "l2_scater_2_8")

    # ax = sns.lmplot(x="sigs_fit_4", y="dataset_hsvd_fit_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "hsvd_scater_4")
    # ax = sns.lmplot(x="sigs_fit_4", y="l2_fit_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "l2_scater_4")

    sns.set_style("white")
    dataset_ = np.reshape(dataset.T, (32, 24, zf), order='F') * np.expand_dims(mask,2)
    sigs_ = np.reshape(sigs.T, (32, 24, zf), order='F')* np.expand_dims(mask,2)
    dataset_hsvd_ = np.reshape(dataset_hsvd.T, (32, 24, zf), order='F')* np.expand_dims(mask,2)
    x = [20, 20, 12, 9]
    y = [14, 7, 14, 9]

    for i, j in zip(x,y):
        soi = [i,j]
        plotppm(dataset_[soi[0],soi[1],:], 0, 6, False, tsf, dt)
        # plotppm(L2[soi[0], soi[1], :], 1, 6, False, tsf, dt)
        plotppm(dataset_hsvd_[soi[0],soi[1],:], 0, 6, False, tsf, dt)
        plotppm(sigs_[soi[0],soi[1],:], 0, 6, True, tsf, dt)
        plt.legend(["orginal","hlsvd", "csvd"])
        plt.title("x = {}, y = {}".format(i, j))
        plt.ylim(-100, 800)
        savefig(dir + "comparison" + str(soi))


        # plotppm(L2[soi[0], soi[1], :], 3, 6, False, tsf, dt)
        plotppm(dataset_hsvd_[soi[0], soi[1], :], 3, 6, False, tsf, dt)
        plotppm(sigs_[soi[0], soi[1], :], 3, 6, True, tsf, dt)
        plt.legend([ "hlsvd", "csvd"])
        plt.title("x = {}, y = {}".format(i,j))
        savefig(dir + "zoom comparison" + str(soi))
    #
    plt.imshow(mask_it(dataset_[:, :, 0].__abs__(), mask), cmap=cmap)
    plt.colorbar()
    plt.title("orginal")
    savefig(dir + "orginal")

    plt.imshow(mask_it(dataset_hsvd_[:, :, 0].__abs__(), mask), cmap=cmap)
    plt.colorbar()
    plt.title("hlsvd")
    savefig(dir + "hlsvd" )

    plt.imshow(mask_it(sigs_[:, :, 0].__abs__(), mask), cmap=cmap)
    plt.colorbar()
    plt.title("csvd")
    savefig(dir + "csvd" )

    # plt.imshow((fftshift(fft(L2, axis=2), axes=2)[:, :, 0].__abs__()))
    # plt.colorbar()
    # plt.title("l2")
    # savefig(dir + "l2")
    # L2 = fftshift(ifft(L2, axis=2), axes=2)

    write_nifti(np.expand_dims(np.conj(dataset_),2),dt, tsf, "1H", dir + "orginal")
    write_nifti(np.expand_dims(np.conj(dataset_hsvd_), 2), dt, tsf, "1H", dir + "hsvd")
    write_nifti(np.expand_dims(np.conj(sigs_), 2), dt, tsf, "1H", dir + "csvd")
    # write_nifti(np.expand_dims(L2, 2), dt, tsf, "1H", dir + "l2")

def testfmrs():
    dir = "fmrs/"
    Path(dir).mkdir(parents=True, exist_ok=True)
    dataset = np.conj(mat73.loadmat("data/fmrs.mat").get("fids"))
    dt = 1/2000
    tsf = 127.798258
    tic()
    csv = CSVD(dataset, dt)
    sigma = np.std(dataset[-68:, :])
    sigs = csv.remove('auto',tsf*0.5,15,sigma=sigma)
    toc(dir+"testfmrs_csvd")


    hlsvd_r = False
    # x = watrem(dataset[:, 11 * 32 + 14], dt, 30, 50)
    if hlsvd_r:
        tic()
        dataset_hsvd = watrem_batch(dataset, dt, 15, tsf*0.5)
        toc(dir + "test_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = (np.load(dir+"hsvd.npy"))



    # soi = 150


    # sigs = csv.remove('auto', tsf * 0.5, 10, sigma=sigma)
    p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    dataset_hsvd_fit_1_4 = integ_fit(dataset_hsvd, p2, p1)
    sigs_fit_1_4 = integ_fit(sigs, p2, p1)
    df = pd.DataFrame()
    df["sigs_fit_1_4"] = sigs_fit_1_4
    df["dataset_hsvd_fit_1_4"] = dataset_hsvd_fit_1_4
    rslt = pd.DataFrame(columns = ["R2", "intrcept", "coef"],index=["1_4"],dtype=object)
    model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                                     df["sigs_fit_1_4"].values.reshape((-1, 1)))
    rslt.iloc[0] = [model.score(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),df["sigs_fit_1_4"].values.reshape((-1, 1)))
                    ,model.intercept_[0]
                    ,model.coef_[0][0]]
    print(rslt)
    rslt.to_csv(dir+"result.csv")
    with open(dir+"t_test.txt",'w') as f:
        f.write("{}\n".format(stats.ttest_ind(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                              df["sigs_fit_1_4"].values.reshape((-1, 1)), equal_var=False)))

    sns.set_style("whitegrid")
    sns.set_palette("dark")
    ax = sns.lmplot(x="sigs_fit_1_4", y="dataset_hsvd_fit_1_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
                    line_kws={'lw': 1}, ci=False)
    max = np.max((sigs_fit_1_4,dataset_hsvd_fit_1_4))
    ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    plt.tick_params(axis='both', labelsize=18)
    savefig(dir + "hsvd_scater_1_4")

    sns.set_style("white")
    bland_altman_plot(df['sigs_fit_1_4'], df['dataset_hsvd_fit_1_4'])
    savefig(dir + "hsvd_bland_altman_1_4")

    sns.set_style("white")
    diff = np.abs(sigs_fit_1_4 - dataset_hsvd_fit_1_4)
    argsort = diff.argsort()
    for soi in [argsort[0],argsort[0],argsort[0],argsort[-3],argsort[-2],argsort[-1]]:
        mode = 'abs'
        plotppm(dataset[:, soi], 1, 6, False, tsf, dt, mode=mode)
        plotppm(dataset_hsvd[:, soi], 1, 6, False, tsf, dt, mode=mode)
        plotppm(sigs[:, soi], 1, 6, True, tsf, dt, mode=mode)
        plt.ylim(-0.02,0.07)
        plt.legend(["orginal","hlsvd","csvd"])
        plt.title(str(soi))
        savefig(dir+"_soi={}".format(soi))
    # sigs = csv.remove('auto', tsf * 0.5, 30, sigma=sigma)
    # plotppm(dataset[:, 50], 1, 6, True, tsf, dt, mode='real')
    # plotppm(sigs[:, 50], 1, 6, True, tsf, dt, mode='real')
    # plotppm(dataset_hsvd[:, 50], 1, 6, True, tsf, dt, mode='real')
    # plt.ylim(-0.05, 0.1)
    # plt.show()
    # p1 = ppm2p(2.8, len(dataset_hsvd),tsf, dt)
    # p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    # dataset_hsvd_fit_2_8 = integ_fit(dataset_hsvd, p2, p1)
    # sigs_fit_2_8 = integ_fit(sigs, p2, p1)
    #
    #
    # p1 = ppm2p(4, len(dataset_hsvd),tsf, dt)
    # p2 = ppm2p(5.2, len(dataset_hsvd), tsf, dt)
    # dataset_hsvd_fit_4 = integ_fit(dataset_hsvd, p2, p1)
    # sigs_fit_4 = integ_fit(sigs, p2, p1)


    # df["sigs_fit_2_8"] = sigs_fit_2_8
    # df["dataset_hsvd_fit_2_8"] = dataset_hsvd_fit_2_8
    #
    # df["sigs_fit_4"] = sigs_fit_4
    # df["dataset_hsvd_fit_4"] = dataset_hsvd_fit_4



    # model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_2_8"].values.reshape((-1, 1)),
    #                                                  df["sigs_fit_2_8"].values.reshape((-1, 1)))
    # rslt.iloc[1] = [model.score(df["dataset_hsvd_fit_2_8"].values.reshape((-1, 1)),df["sigs_fit_2_8"].values.reshape((-1, 1)))
    # ,model.intercept_[0]
    # ,model.coef_[0][0]]
    #
    # model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_4"].values.reshape((-1, 1)),
    #                                                  df["sigs_fit_4"].values.reshape((-1, 1)))
    # rslt.iloc[2] = [model.score(df["dataset_hsvd_fit_4"].values.reshape((-1, 1)),df["sigs_fit_4"].values.reshape((-1, 1)))
    # ,model.intercept_[0],
    # model.coef_[0][0]]

    # ax = sns.lmplot(x="sigs_fit_2_8", y="dataset_hsvd_fit_2_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # max = np.max((sigs_fit_2_8, dataset_hsvd_fit_2_8))
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "hsvd_scater_2_8")
    #
    #
    # ax = sns.lmplot(x="sigs_fit_4", y="dataset_hsvd_fit_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # max = np.max((sigs_fit_4, dataset_hsvd_fit_4))
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "hsvd_scater_4")

def test3dcsi():
    dataset = np.conj(mat73.loadmat("data/3dcsi.mat").get("c"))[0:500,:].T
    dt = 1/2778
    tic()
    csv = CSVD(dataset, dt)
    sigs = csv.remove(20,100,40)
    toc("test3csi_csvd")
    tic()
    dataset_hsvd = watrem_batch(dataset, dt, 15, 100)
    toc("test3csi_hsvd")

    soi = 150
    plotppm(dataset[:, soi], 1, 5, False, 289, dt)
    plotppm(dataset_hsvd[:, soi], 1, 5, False, 289, dt)
    plotppm(sigs[:, soi], 1, 5, True, 289, dt)
    plt.legend(["orginal","hlsvd","csvd"])
    plt.show()

def combine_rudy():
    sns.set_style("white")
    dataset_ = []
    for volume in ["vol1", "vol2", "vol3", "vol4"]:
        dir = "testCSIrudy_combined/"+volume+"/"
        Path(dir).mkdir(parents=True,exist_ok=True)
        dataset_.append(np.conj(sio.loadmat("data/"+volume+".mat").get("data"))*np.exp(np.pi*1j))
    dataset_ = np.concatenate(dataset_,1)
    mask = np.full((16, 16), False)
    mask[5:11,5:11] = True
    sigma = np.std(dataset_[-68:,:])
    zf = 4096
    dt = 0.00025
    tsf = 123.25
    dataset = np.zeros((zf,dataset_.shape[1]),dtype=np.complex128)
    dataset[0:dataset_.shape[0],:] = dataset_
    tic()
    csvd = CSVD(dataset, dt)
    sigs = csvd.remove('auto',([-tsf*0.5],[tsf*0.5]),50,sigma=sigma)
    toc(dir + "csvd")

    hlsvd_r = False
    if hlsvd_r:
        tic()
        dataset_hsvd = watrem_batch(dataset, dt, 30, tsf*0.5)
        toc(dir + "testCSI_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = np.load(dir+"hsvd.npy")
    plt.plot(csvd.s)
    savefig(dir + "S")
    for jj in range(0, 5):
        plotppm(csvd.U[:, jj], 1, 6, False, tsf, dt)

    plotppm(csvd.U[:, 6], 1, 6, True, tsf, dt)
    sns.despine()
    plt.legend(range(0, 6))
    savefig(dir + "U")

    # p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    # p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    # sMat1 = MSE(np.abs(dataset_hsvd),np.abs(sigs.real))
    #
    # max = np.max(sMat1[mask.reshape(-1, order='F') == 1])
    #
    # plt.imshow(sMat1.reshape((16, 16), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "sMat_hsvd")
    for idx,volume in enumerate(["vol1", "vol2", "vol3", "vol4"]):
        dir = "testCSIrudy_combined/"+volume+"/"
        dataset_hsvd_ = dataset_hsvd[:,idx*256:(idx+1)*256]
        sigs_ = sigs[:, idx*256:(idx+1)*256]
        dataset_ = dataset[:, idx*256:(idx+1)*256]
        p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
        p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
        dataset_hsvd_fit_1_4 = integ_fit(dataset_hsvd_, p2, p1)
        sigs_fit_1_4 = integ_fit(sigs_, p2, p1)

        sns.set_style("white")
        sns.set_palette("dark")
        df = pd.DataFrame()
        df["sigs_fit_1_4"] = sigs_fit_1_4[mask.reshape(-1,order='F') == 1]
        df["dataset_hsvd_fit_1_4"] = dataset_hsvd_fit_1_4[mask.reshape(-1,order='F') == 1]

        rslt = pd.DataFrame(columns = ["R2", "intrcept", "coef"],index=["1.8", "2.8", "4"],dtype=object)
        model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                                         df["sigs_fit_1_4"].values.reshape((-1, 1)))
        rslt.iloc[0] = [model.score(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),df["sigs_fit_1_4"].values.reshape((-1, 1)))
                        ,model.intercept_[0]
                        ,model.coef_[0][0]]

        rslt.to_csv(dir+"regression")
        with open(dir+"t_test.txt",'w') as f:
            f.write("{}\n".format(stats.ttest_ind(df["dataset_hsvd_fit_1_4"].values.reshape((-1, 1)),
                                                  df["sigs_fit_1_4"].values.reshape((-1, 1)), equal_var=False)))

        max = df[["sigs_fit_1_4", "dataset_hsvd_fit_1_4"]].max().max()
        # ax = sns.lmplot(x="sigs_fit_1_4", y="dataset_hsvd_fit_1_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
        #                 line_kws={'lw': 1}, ci=False)
        # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
        # plt.tick_params(axis='both', labelsize=18)
        # savefig(dir + "hsvd_scater_1_4")

        bland_altman_plot(df['sigs_fit_1_4'], df['dataset_hsvd_fit_1_4'])
        savefig(dir + "hsvd_bland_altman_1_4")

        sns.set_style("white")
        diff = np.abs(sigs_fit_1_4 - dataset_hsvd_fit_1_4)[mask.reshape(-1, order='F') == 1]
        argsort = diff.argsort()
        d = np.where(mask.reshape(-1, order='F') == True)[0]
        for id,soi in enumerate([d[argsort[0]],d[argsort[0]],d[argsort[0]],d[argsort[-3]],d[argsort[-2]],d[argsort[-1]]]):
            mode = 'abs'
            plotppm(dataset_[:, soi], 1, 6, False, tsf, dt, mode=mode)
            plotppm(dataset_hsvd_[:, soi], 1, 6, False, tsf, dt, mode=mode)
            plotppm(sigs_[:, soi], 1, 6, True, tsf, dt, mode=mode)
            plt.ylim(-0.05,0.15)
            plt.legend(["orginal","hlsvd","csvd"])
            plt.title(str(soi))
            savefig(dir+"_soi={}".format(id))

            plotppm(dataset_hsvd_[:, soi], 3, 6, False, tsf, dt)
            plotppm(sigs_[:, soi], 3, 6, True, tsf, dt)
            plt.legend(["hlsvd", "csvd"])
            plt.title(str(soi))
            savefig(dir+"zoomed_soi={}".format(id))

        weidth = 16
        height = 16
        dataset_ = np.reshape(dataset_.T,  (weidth, height, zf), order='F')
        sigs_ = np.reshape(sigs_.T, (weidth, height, zf), order='F')
        dataset_hsvd_ = np.reshape(dataset_hsvd_.T, (weidth, height, zf), order='F')

        write_nifti(np.expand_dims(np.conj(dataset_),2),dt, tsf, "1H", dir + "orginal")
        write_nifti(np.expand_dims(np.conj(dataset_hsvd_), 2), dt, tsf, "1H", dir + "hsvd")
        write_nifti(np.expand_dims(np.conj(sigs_), 2), dt, tsf, "1H", dir + "csvd")

def testZenon():
    sns.set_style("white")
    dir = "testZenon/"
    cmap = 'autumn'
    Path(dir).mkdir(parents=True,exist_ok=True)
    dataset_ = sio.loadmat("data/zenon.mat").get("data").T
    # mask = sio.loadmat(dir+"/meta_mask.mat").get("meta_mask")
    # L2 = np.conj(sio.loadmat(dir + "l2rslt.mat").get("mrs1")) * np.expand_dims(mask, 2)
    sigma =np.std(dataset_[-320:-128,:])/np.sqrt(2)
    zf = 4096
    dt = 1/4000
    tsf = 400.32
    dataset = np.zeros((zf,dataset_.shape[1]),dtype=np.complex128)
    dataset[0:dataset_.shape[0],:] = dataset_
    tic()
    csvd = CSVD(dataset, dt)
    sigs = csvd.remove(25,([-tsf*0.5],[tsf*0.5]),30,sigma=sigma)
    toc(dir + "testCSI_csvd")
    tic()
    hlsvd_r = True
    if hlsvd_r:
        tic()
        dataset_hsvd = watrem_batch(dataset, dt, 30, ([-tsf*0.5,tsf*2.8],[tsf*0.5,tsf*4.7]))
        toc(dir + "testCSI_hsvd")
        np.save(dir+"hsvd.npy",dataset_hsvd)
    else:
        dataset_hsvd = np.load(dir+"hsvd.npy")


    plt.plot(csvd.s[0:50])
    savefig(dir + "S")
    for jj in range(0, 5):
        plotppm(csvd.U[:, jj], 1, 6, False, tsf, dt)

    plotppm(csvd.U[:, 6], 1, 6, True, tsf, dt)
    sns.despine()
    plt.legend(range(0, 6))
    savefig(dir + "U")

    p1 = ppm2p(0, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    sMat1 = similarityMat(dataset_hsvd,sigs,p2,p1)
    # sMat2= similarityMat(L2.reshape((-1,zf),order='F').T,sigs,p2,p1)
    # sMat2[np.isnan(sMat2)] = 0
    # max = np.max(
    #     (sMat1[mask.reshape(-1, order='F') == 1], sMat2[mask.reshape(-1, order='F') == 1]))
    plt.imshow(mask_it(sMat1.reshape((32, 24), order='F') , mask),cmap = cmap)
    plt.colorbar()
    savefig(dir + "sMat_hsvd")


    # plt.imshow(sMat2.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "sMat_l2")

    p1 = ppm2p(1, len(dataset_hsvd),tsf, dt)
    p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    dataset_hsvd_fit_1_8 = integ_fit(dataset_hsvd, p2, p1)
    sigs_fit_1_8 = integ_fit(sigs, p2, p1)
    # l2_fit_1_8 = integ_fit(L2.reshape((-1,zf),order='F').T, p2, p1)

    # p1 = ppm2p(2.8, len(dataset_hsvd),tsf, dt)
    # p2 = ppm2p(4, len(dataset_hsvd), tsf, dt)
    # dataset_hsvd_fit_2_8 = integ_fit(dataset_hsvd, p2, p1)
    # sigs_fit_2_8 = integ_fit(sigs, p2, p1)
    # # l2_fit_2_8 = integ_fit(L2.reshape((-1,zf),order='F').T, p2, p1)
    #
    # p1 = ppm2p(4, len(dataset_hsvd),tsf, dt)
    # p2 = ppm2p(5.2, len(dataset_hsvd), tsf, dt)
    # dataset_hsvd_fit_4 = integ_fit(dataset_hsvd, p2, p1)
    # sigs_fit_4 = integ_fit(sigs, p2, p1)
    # # l2_fit_4 = integ_fit(L2.reshape((-1,zf),order='F').T, p2, p1)

    max = np.max((dataset_hsvd_fit_1_8[mask.reshape(-1,order='F') == 1], sigs_fit_1_8[mask.reshape(-1,order='F') == 1]))
    plt.imshow(mask_it(dataset_hsvd_fit_1_8.reshape((32, 24), order='F') , mask), cmap=cmap,vmax=max)
    plt.colorbar()
    savefig(dir + "hsvd_1_8")
    plt.imshow(mask_it(sigs_fit_1_8.reshape((32, 24), order='F'),mask),cmap=cmap,vmax=max)
    plt.colorbar()
    savefig(dir + "csvd_1_8")
    # plt.imshow(l2_fit_1_8.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "l2_1_8")

    # max = np.max((dataset_hsvd_fit_2_8[mask.reshape(-1,order='F') == 1], sigs_fit_2_8[mask.reshape(-1,order='F') == 1]))
    # plt.imshow(dataset_hsvd_fit_2_8.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "hsvd_2_8")
    # plt.imshow(sigs_fit_2_8.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "csvd_2_8")
    # # plt.imshow(l2_fit_2_8.reshape((32, 24), order='F') * mask,vmax=150)
    # # plt.colorbar()
    # # savefig(dir + "l2_2_8")
    #
    # max = np.max((dataset_hsvd_fit_4[mask.reshape(-1,order='F') == 1],sigs_fit_4[mask.reshape(-1,order='F') == 1]))
    # plt.imshow(dataset_hsvd_fit_4.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "hsvd_4")
    # plt.imshow(sigs_fit_4.reshape((32, 24), order='F') * mask,vmax=max)
    # plt.colorbar()
    # savefig(dir + "csvd_4")
    # plt.imshow(l2_fit_4.reshape((32, 24), order='F') * mask,vmax=150)
    # plt.colorbar()
    # savefig(dir + "l2_4")

    sns.set_style("whitegrid")
    sns.set_palette("dark")

    df = pd.DataFrame()
    df["sigs_fit_1_8"] = sigs_fit_1_8[mask.reshape(-1,order='F') == 1]
    # df["l2_fit_1_8"] = l2_fit_1_8[mask.reshape(-1,order='F') == 1]
    df["dataset_hsvd_fit_1_8"] = dataset_hsvd_fit_1_8[mask.reshape(-1,order='F') == 1]
    # df["sigs_fit_2_8"] = sigs_fit_2_8[mask.reshape(-1,order='F') == 1]
    # # df["l2_fit_2_8"] = l2_fit_2_8[mask.reshape(-1,order='F') == 1]
    # df["dataset_hsvd_fit_2_8"] = dataset_hsvd_fit_2_8[mask.reshape(-1,order='F') == 1]
    # df["sigs_fit_4"] = sigs_fit_4[mask.reshape(-1,order='F') == 1]
    # # df["l2_fit_4"] = l2_fit_4[mask.reshape(-1,order='F') == 1]
    # df["dataset_hsvd_fit_4"] = dataset_hsvd_fit_4[mask.reshape(-1,order='F') == 1]

    rslt = pd.DataFrame(columns = ["R2", "intrcept", "coef"],index=["1.8"],dtype=object)
    model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_1_8"].values.reshape((-1, 1)),
                                                     df["sigs_fit_1_8"].values.reshape((-1, 1)))
    rslt.iloc[0] = [model.score(df["dataset_hsvd_fit_1_8"].values.reshape((-1, 1)),df["sigs_fit_1_8"].values.reshape((-1, 1)))
                    ,model.intercept_[0]
                    ,model.coef_[0][0]]
    rslt.to_csv(dir+"regression")
    with open(dir+"t_test.txt",'w') as f:
        f.write("{}\n".format(stats.ttest_ind(df["dataset_hsvd_fit_1_8"].values.reshape((-1, 1)),
                                              df["sigs_fit_1_8"].values.reshape((-1, 1)), equal_var=False)))

    # model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_2_8"].values.reshape((-1, 1)),
    #                                                  df["sigs_fit_2_8"].values.reshape((-1, 1)))
    # rslt.iloc[1] = [model.score(df["dataset_hsvd_fit_2_8"].values.reshape((-1, 1)),df["sigs_fit_2_8"].values.reshape((-1, 1)))
    # ,model.intercept_[0]
    # ,model.coef_[0][0]]
    #
    # model = LinearRegression(fit_intercept=True).fit(df["dataset_hsvd_fit_4"].values.reshape((-1, 1)),
    #                                                  df["sigs_fit_4"].values.reshape((-1, 1)))
    # rslt.iloc[2] = [model.score(df["dataset_hsvd_fit_4"].values.reshape((-1, 1)),df["sigs_fit_4"].values.reshape((-1, 1)))
    # ,model.intercept_[0],
    # model.coef_[0][0]]

    ax = sns.lmplot(x="sigs_fit_1_8", y="dataset_hsvd_fit_1_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
                    line_kws={'lw': 1}, ci=False)
    ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    savefig(dir + "hsvd_scater_1_8")

    sns.set_style("white")
    bland_altman_plot(df['sigs_fit_1_8'], df['dataset_hsvd_fit_1_8'])
    savefig(dir + "hsvd_bland_altman_1_4")

    # ax = sns.lmplot(x="sigs_fit_1_8", y="l2_fit_1_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "l2_scater_1_8")

    # ax = sns.lmplot(x="sigs_fit_2_8", y="dataset_hsvd_fit_2_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "hsvd_scater_2_8")

    # ax = sns.lmplot(x="sigs_fit_2_8", y="l2_fit_2_8", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "l2_scater_2_8")

    # ax = sns.lmplot(x="sigs_fit_4", y="dataset_hsvd_fit_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "hsvd_scater_4")
    # ax = sns.lmplot(x="sigs_fit_4", y="l2_fit_4", data=df, legend=True, scatter_kws={'alpha': 0.6},
    #                 line_kws={'lw': 1}, ci=False)
    # ax.axes[0, 0].axline((max, max), slope=1, ls="--", zorder=0, color='silver')
    # savefig(dir + "l2_scater_4")

    sns.set_style("white")
    dataset_ = np.reshape(dataset.T, (32, 24, zf), order='F') * np.expand_dims(mask,2)
    sigs_ = np.reshape(sigs.T, (32, 24, zf), order='F')* np.expand_dims(mask,2)
    dataset_hsvd_ = np.reshape(dataset_hsvd.T, (32, 24, zf), order='F')* np.expand_dims(mask,2)
    x = [20, 20, 12, 9]
    y = [14, 7, 14, 9]

    for i, j in zip(x,y):
        soi = [i,j]
        plotppm(dataset_[soi[0],soi[1],:], 0, 6, False, tsf, dt)
        # plotppm(L2[soi[0], soi[1], :], 1, 6, False, tsf, dt)
        plotppm(dataset_hsvd_[soi[0],soi[1],:], 0, 6, False, tsf, dt)
        plotppm(sigs_[soi[0],soi[1],:], 0, 6, True, tsf, dt)
        plt.legend(["orginal","hlsvd", "csvd"])
        plt.title("x = {}, y = {}".format(i, j))
        plt.ylim(-100, 800)
        savefig(dir + "comparison" + str(soi))


        # plotppm(L2[soi[0], soi[1], :], 3, 6, False, tsf, dt)
        plotppm(dataset_hsvd_[soi[0], soi[1], :], 3, 6, False, tsf, dt)
        plotppm(sigs_[soi[0], soi[1], :], 3, 6, True, tsf, dt)
        plt.legend([ "hlsvd", "csvd"])
        plt.title("x = {}, y = {}".format(i,j))
        savefig(dir + "zoom comparison" + str(soi))
    #
    plt.imshow(mask_it(dataset_[:, :, 0].__abs__(), mask), cmap=cmap)
    plt.colorbar()
    plt.title("orginal")
    savefig(dir + "orginal")

    plt.imshow(mask_it(dataset_hsvd_[:, :, 0].__abs__(), mask), cmap=cmap)
    plt.colorbar()
    plt.title("hlsvd")
    savefig(dir + "hlsvd" )

    plt.imshow(mask_it(sigs_[:, :, 0].__abs__(), mask), cmap=cmap)
    plt.colorbar()
    plt.title("csvd")
    savefig(dir + "csvd" )

    # plt.imshow((fftshift(fft(L2, axis=2), axes=2)[:, :, 0].__abs__()))
    # plt.colorbar()
    # plt.title("l2")
    # savefig(dir + "l2")
    # L2 = fftshift(ifft(L2, axis=2), axes=2)

    write_nifti(np.expand_dims(np.conj(dataset_),2),dt, tsf, "1H", dir + "orginal")
    write_nifti(np.expand_dims(np.conj(dataset_hsvd_), 2), dt, tsf, "1H", dir + "hsvd")
    write_nifti(np.expand_dims(np.conj(sigs_), 2), dt, tsf, "1H", dir + "csvd")
    # write_nifti(np.expand_dims(L2, 2), dt, tsf, "1H", dir + "l2")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # testBiggaba()
    # for WS in [5]:#,25,125]:
    #     testsimulatedCSI(ws=WS,hlsvd_r = False)
    testsimulatedCSI_rank_analyze()
    # for vol in ["vol1","vol2","vol3","vol4"]:
    #     testCSIrudy(vol)
    # testCSI()
    # testfmrs() # done
    # test3dcsi()
    # combine_rudy()
    # test_GN()
    # testZenon()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
