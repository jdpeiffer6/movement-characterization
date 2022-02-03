# %%
import os
import TrialObject
import matplotlib.pyplot as plt
from scipy.misc import derivative

from TrialObject import TrialObject
path_to_trial = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
trial_obj = TrialObject(path_to_trial)

# %%
data = trial_obj.markerless_data['Gait Markerless 1']
pelvisz = data['Pelvis_WRT_LabZ']
dpelvisz = derivative(pelvisz)
# %%
import numpy as np
from rms import rms
n=100
print(rms(np.array([10+n,4+n,6+n,8+n])))

# %%
