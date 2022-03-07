# %%
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\003\\2022-02-11"
subject = SessionDataObject(path_to_subject,False,1.7)
path_to_control =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
control = SessionDataObject(path_to_control,False,1.7)
# %%
# labels=['lightblue','pink','lightblue','pink','lightblue','pink']
labels=['#5ab4ac','#d8b365','#5ab4ac','#d8b365','#5ab4ac','#d8b365']
plt.subplot(1,5,1)
sub_pointer=subject.marker_output_data['Pointer Dist'].values
sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
sub_thumb=subject.marker_output_data['Thumb Dist'].values
sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
ctl_pointer=control.marker_output_data['Pointer Dist'].values
ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
ctl_thumb=control.marker_output_data['Thumb Dist'].values
ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
a = ranksums(data[0],data[1])
print("Path Length:")
print("Pointer p: %.5f"%a[1])
a = ranksums(data[2],data[3])
print("Thumb p: %.5f"%a[1])
bplot1 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# plt.violinplot(data)
j=0
for i in bplot1['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Distance\n(% of straight line)")
plt.box(False)
plt.subplot(1,5,2)
sub_pointer=subject.marker_output_data['Pointer Accel'].values
sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
sub_thumb=subject.marker_output_data['Thumb Accel'].values
sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
ctl_pointer=control.marker_output_data['Pointer Accel'].values
ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
ctl_thumb=control.marker_output_data['Thumb Accel'].values
ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
a = ranksums(data[0],data[1])
print("\nAccel:")
print("Pointer p: %.5f"%a[1])
a = ranksums(data[2],data[3])
print("Thumb p: %.5f"%a[1])
bplot2 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# bplot2 = plt.violinplot(data)
j=0
for i in bplot2['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Acceleration\n($mm^2/s$)")
plt.box(False)

plt.subplot(1,5,3)
sub_pointer=subject.marker_output_data['Pointer Jerk'].values
sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
sub_thumb=subject.marker_output_data['Thumb Jerk'].values
sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
ctl_pointer=control.marker_output_data['Pointer Jerk'].values
ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
ctl_thumb=control.marker_output_data['Thumb Jerk'].values
ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
a = ranksums(data[0],data[1])
print("\nJerk:")
print("Pointer p: %.5f"%a[1])
a = ranksums(data[2],data[3])
print("Thumb p: %.5f"%a[1])
bplot3 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
j=0
for i in bplot3['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Jerk\n($mm^3/s$)")
plt.box(False)

plt.subplot(1,5,4)
sub_mga = subject.marker_output_data['MGA'].values
sub_mga = sub_mga[~np.isnan(sub_mga)]
ctl_mga = control.marker_output_data['MGA'].values
ctl_mga = ctl_mga[~np.isnan(ctl_mga)]
data=[ctl_mga,sub_mga]
a = ranksums(data[0],data[1])
print("MGA p: %.5f"%a[1])
bplot4 = plt.boxplot(data,labels=['Control','Subject'],patch_artist=True,widths=0.5)
j=0
for i in bplot4['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("MGA\n(mm)")
plt.box(False)

plt.subplot(1,5,5)
sub_mgat = subject.marker_output_data['MGA_t'].values
sub_mgat = sub_mgat[~np.isnan(sub_mgat)]
sub_mgat = sub_mgat[~(sub_mgat<0.55)]
ctl_mgat = control.marker_output_data['MGA_t'].values
ctl_mgat = ctl_mgat[~np.isnan(ctl_mgat)]
ctl_mgat = ctl_mgat[~(ctl_mgat<0.55)]
data=[ctl_mgat,sub_mgat]
a = ranksums(data[0],data[1])
print("MGA Time p: %.5f"%a[1])
bplot5 = plt.boxplot(data,labels=['Control','Subject'],patch_artist=True,widths=0.5)
j=0
for i in bplot5['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("MGA Time\n% of trial")
plt.box(False)
plt.show()