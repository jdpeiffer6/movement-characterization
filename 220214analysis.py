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
plt.ylabel("Normalized Distance (m/m)")
labels=['lightblue','pink','lightblue','pink','lightblue','pink']
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()

# %% joint angles
miguel_angles = []
miguel_angles.append(miguel.r_hip_walking)
miguel_angles.append(miguel.l_hip_walking)
miguel_angles.append(miguel.r_knee_walking)
miguel_angles.append(miguel.l_knee_walking)

subject_angles = []
subject_angles.append(subject.r_hip_walking)
subject_angles.append(subject.l_hip_walking)
subject_angles.append(subject.r_knee_walking)
subject_angles.append(subject.l_knee_walking)

plt.subplot(1,2,1)
bplot2 = plt.boxplot([miguel_angles[0],miguel_angles[1],subject_angles[0],subject_angles[1]],labels=["Control R Hip","Control L Hip","Subject R Hip","Subject L Hip"],patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Hip Angles Comparison while walking")
j=0
for i in bplot2['boxes']:
    i.set_facecolor(labels[j])
    j+=1

plt.subplot(1,2,2)
bplot3 = plt.boxplot([miguel_angles[2],miguel_angles[3],subject_angles[2],subject_angles[3]],labels=["Control R Knee","Control L Knee","Subject R Knee","Subject L Knee"],patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Knee Angles Comparison while walking")
j=0
for i in bplot3['boxes']:
    i.set_facecolor(labels[j])
    j+=1
#intra subject
print("Control R/L Hip: %.6f\nSubject R/L Hip: %.6f\n"%(ranksums(miguel_angles[0],miguel_angles[1])[1],ranksums(subject_angles[0],subject_angles[1])[1]))
print("Control R/L Knee: %.6f\nSubject R/L Knee: %.6f\n"%(ranksums(miguel_angles[2],miguel_angles[3])[1],ranksums(subject_angles[2],subject_angles[3])[1]))
#inter subject
print("Control/Subject R Hip: %.6f\nControl/Subject L Hip: %.6f\n"%(ranksums(miguel_angles[0],subject_angles[0])[1],ranksums(miguel_angles[1],subject_angles[1])[1]))
print("Control/Subject R Knee: %.6f\nControl/Subject L Knee: %.6f\n"%(ranksums(miguel_angles[2],subject_angles[2])[1],ranksums(miguel_angles[3],subject_angles[3])[1]))

plt.show()

# %%
