# %%
import os
import TrialObject
import matplotlib.pyplot as plt

from TrialObject import TrialObject
path_to_trial = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
trial_obj = TrialObject(path_to_trial)

# %%
plt.plot(trial_obj.markerless_data['Gait Markerless 1']['R_HEELZ'])
plt.plot(trial_obj.markerless_data['Gait Markerless 1']['RTOES_DISTALZ'])

# %%
