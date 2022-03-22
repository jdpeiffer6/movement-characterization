# %%
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=True,ng=False)
path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\003\\2022-02-11"
subject = SessionDataObject(path_to_subject,False,1.7,walking=True,ng=False)

# %% tandem pelvis jerk
control_jerk=miguel.tandem_pelvis_jerks
subject_jerk=subject.tandem_pelvis_jerks
# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlie
data=[control_jerk,subject_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","Subject"],patch_artist=True)
plt.title("Pelvis Jerk Comparison\nTandem")
plt.ylabel("Normalized Jerk")
a=ranksums(control_jerk,subject_jerk)
print("Pelvis Jerk: %.5f"%a[1])
plt.show()

# %% tandem thorax jerk
control_jerk=miguel.tandem_thorax_jerks
subject_jerk=subject.tandem_thorax_jerks
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlie
data=[control_jerk,subject_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","Subject"],patch_artist=True)
plt.title("Thorax Jerk Comparison\nTandem")
plt.ylabel("Normalized Jerk")
a=ranksums(control_jerk,subject_jerk)
print("Thorax Jerk: %.5f"%a[1])
plt.show()