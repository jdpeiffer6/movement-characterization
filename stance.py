# %%
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=True,ng=False)
path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\003\\2022-02-11"
subject = SessionDataObject(path_to_subject,False,1.7,walking=True,ng=False)

# %% double stance 
control_ds=miguel.double_stances
subject_ds=subject.double_stances
data=[control_ds,subject_ds]
bplot_ds = plt.boxplot(data,labels=["Control DS","Subject DS"],patch_artist=True)
plt.title("Double Stance Comparison")
plt.ylabel("Fraction of Gait Cycle")
a=ranksums(control_ds,subject_ds)
print("Double Stance p: %.5f"%a[1])
plt.show()
# %%
