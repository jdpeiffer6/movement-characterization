# %%
import os
import TrialObject
import matplotlib.pyplot as plt
import numpy as np

from TrialObject import TrialObject
path_to_trial = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
trial_obj = TrialObject(path_to_trial)

# %%
data = trial_obj.markerless_data['Gait Markerless 1']
pelvisz = data['Pelvis_WRT_LabZ']


# %%
import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-2,2,num=100)
y=x*x
plt.plot(x,y)
dydx = np.gradient(y,4/100)
plt.plot(x,dydx)
plt.show()
# %%
