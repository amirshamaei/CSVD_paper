import nibabel
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, mask_it, RMSE
fontsize = 18
met_name= ["Ala", "Asc", "Asp", "Cr", "GABA", "GPC", "GSH", "Glc", "Gln", "Glu",
             "Gly", "Ins", "Lac", "NAA", "NAAG", "PCho", "PCr", "PE", "Tau", "sIns", "_mm"]
_, _, ampl_t, _, white, gray, _, _, _, _, _, _, _, _ = load_data(
    "data/golden brain/ismrm challenge/HomoDeus_01/" + "data_ws_" + str(25) + "_False" + ".npz")
# csvd_25 = nibabel.load("testsimulatedCSI/25/hsvd.nii")

mask = white + gray
with open("testsimulatedCSI/25/csvd nifti/csvd.json", "r") as read_content:
    csvd_25 = json.load(read_content)

ampl_c = np.array(csvd_25["compAmplitude"])
ampl_c = ampl_c.reshape((16, 16, -1))


with open("testsimulatedCSI/25/hsvd/hsvd.json", "r") as read_content:
    hsvd_25 = json.load(read_content)

ampl_h = np.array(hsvd_25["compAmplitude"])
ampl_h = ampl_h.reshape((16, 16, -1))
cmap= 'autumn'

for i, name in enumerate(hsvd_25["peakNames"]):
    print("i:{}, name:{}".format(i,name))
    print("i:{}, name:{}".format(i,met_name.index(name)))
    idx = met_name.index(name)

    fig, ax = plt.subplots(1,3,figsize=(15,5))

    pcm1 = ax[0].imshow(mask_it(np.abs(ampl_t[:, :, idx]), np.squeeze(mask)), cmap=cmap)
    ax[0].set_title("{}".format(name),fontsize=fontsize)
    cb=fig.colorbar(pcm1, ax=ax[0])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=fontsize)

    pcm2= ax[1].imshow(mask_it(np.abs(ampl_h[:, :, i] - ampl_t[:, :, idx]), np.squeeze(mask)), vmax=np.max(ampl_t[:, :, idx]),
               cmap=cmap)
    ax[1].set_title("RMSE = {:.2f}".format(RMSE(ampl_h[:, :, i] ,ampl_t[:, :, idx])),fontsize=fontsize)
    cb=fig.colorbar(pcm2, ax=ax[1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=fontsize)

    pcm3=ax[2].imshow(mask_it(np.abs(ampl_c[:, :, i] - ampl_t[:, :, idx]), np.squeeze(mask)), vmax=np.max(ampl_t[:, :, idx]),
               cmap=cmap)
    ax[2].set_title("RMSE = {:.2f}".format(RMSE(ampl_c[:, :, i], ampl_t[:, :, idx])),fontsize=fontsize)
    cb=fig.colorbar(pcm3, ax=ax[2])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=fontsize)
    # plt.show()
    plt.savefig("testsimulatedCSI/25/" + "compare_25"+name+".svg")
#     print(np.sum(np.abs(ampl_h[:,:,1]-ampl_t[:,:,3])))
#     print(np.sum(np.abs(ampl_c[:,:,1]-ampl_t[:,:,3])))

