import json

import numpy as np
from matplotlib import pyplot as plt

with open("testCSI/hsvd_amars/rs.results.json") as f:
    json_ = json.load(f)
print(json_)
amp = np.array(json_["compAmplitude"])
plt.plot(amp[:,0].reshape((32,24,512),order='F'))
