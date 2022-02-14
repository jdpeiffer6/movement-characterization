# %%
import os
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

from functions.SessionDataObject import SessionDataObject
path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
miguel = SessionDataObject(path_to_miguel,False,1.75)
for i in range(miguel.markerless_step_length.size-1,0,-1):
    if miguel.markerless_step_length[i] > 1:
        miguel.markerless_step_length = np.delete(miguel.markerless_step_length,i)
path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\003\\2022-02-11"
subject = SessionDataObject(path_to_subject,False,1.7)

# %%

# plt.hist(miguel.markerless_step_height,alpha=0.5)
# plt.hist(subject.markerless_step_height,alpha=0.5)
# plt.legend(["Control","Subject"])
# plt.title("Step Height")
# plt.show()

# plt.hist(miguel.markerless_step_length,alpha=0.5)
# plt.hist(subject.markerless_step_length,alpha=0.5)
# plt.legend(["Control","Subject"])
# plt.title("Step Length")
# plt.show()

# plt.hist(miguel.markerless_step_width,alpha=0.5)
# plt.hist(subject.markerless_step_width,alpha=0.5)
# plt.legend(["Control","Subject"])
# plt.title("Step Width")
# plt.show()
# %%
# plt.boxplot([miguel.markerless_step_height,subject.markerless_step_height],labels=["Control","Subject"])
# a=ranksums(miguel.markerless_step_height,subject.markerless_step_height)
# print("Step Height: %.6f\n"%a[1])
# plt.ylabel("Normalized Step Height")
# plt.title("Step Height Comparison")
# plt.show()

# plt.boxplot([miguel.markerless_step_length,subject.markerless_step_length],labels=["Control","Subject"])
# a=ranksums(miguel.markerless_step_length,subject.markerless_step_length)
# print("Step Length: %.6f\n"%a[1])
# plt.ylabel("Normalized Step Length")
# plt.title("Step Length Comparison")
# plt.show()

# plt.boxplot([miguel.markerless_step_width,subject.markerless_step_width],labels=["Control","Subject"])
# a=ranksums(miguel.markerless_step_width,subject.markerless_step_width)
# print("Step Width: %.6f\n"%a[1])
# plt.ylabel("Normalized Step Width")
# plt.title("Step Width Comparison")
# plt.show()
# %%
bplot = plt.boxplot([miguel.markerless_step_height,subject.markerless_step_height,miguel.markerless_step_length,subject.markerless_step_length,miguel.markerless_step_width,subject.markerless_step_width],labels=["Control Step Height","Subject Step Height","Control Step Length","Subject Step Length","Control Step Width","Subject Step Width"],patch_artist=True)
a=ranksums(miguel.markerless_step_height,subject.markerless_step_height)
print("Step Height: %.6f\n"%a[1])
a=ranksums(miguel.markerless_step_length,subject.markerless_step_length)
print("Step Length: %.6f\n"%a[1])
a=ranksums(miguel.markerless_step_width,subject.markerless_step_width)
print("Step Width: %.6f\n"%a[1])
plt.ylabel("Normalized Distance")
labels=['pink','lightblue','pink','lightblue','pink','lightblue']
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()