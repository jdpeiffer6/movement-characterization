# %% Import data
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\005\\2022-05-19"
subject = SessionDataObject(path_to_subject,False,1.7,walking=True,ng=False)
path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=True,ng=False)

# %% Pelvis jerk
control_jerk=miguel.getOutput('Walking','pelvis_jerk_step_normalized')
subject_jerk=subject.getOutput('Walking','pelvis_jerk_step_normalized')

# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier

data=[control_jerk,subject_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","Subject"],patch_artist=True)
plt.title("Pelvis Jerk Comparison")
plt.ylabel("Normalized Jerk")
a=ranksums(control_jerk,subject_jerk)
print("Pelvis Jerk p: %.5f"%a[1])
plt.show()

# # %% Thorax jerk
# control_jerk=miguel.getOutput('Walking','thorax_jerk_step_normalized')
# subject_jerk=subject.getOutput('Walking','thorax_jerk_step_normalized')

# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier

# data=[control_jerk,subject_jerk]
# bplot_jerk = plt.boxplot(data,labels=["Control","Subject"],patch_artist=True)
# plt.title("Thorax Jerk Comparison")
# plt.ylabel("Normalized Jerk")
# a=ranksums(control_jerk,subject_jerk)
# print("Thorax Jerk p: %.5f"%a[1])
# plt.show()
# # %% Walking paths
# control_devangle=miguel.getOutput('Walking','walking_angle_deviation')
# subject_devangle=subject.getOutput('Walking','walking_angle_deviation')
# control_devangle=control_devangle[~np.isnan(control_devangle)]
# subject_devangle=subject_devangle[~np.isnan(subject_devangle)]
# data=[control_devangle,subject_devangle]
# bplot_jerk = plt.boxplot(data,labels=["Control","Subject"],patch_artist=True)
# plt.title("Wakling Deviation Angles Comparison")
# plt.ylabel("Angle (Degrees)")
# a=ranksums(control_devangle,subject_devangle)
# print("Angle Dev p: %.5f"%a[1])
# plt.show()

# # %% Joint angles
# labels=['#5ab4ac','#d8b365','#5ab4ac','#d8b365','#5ab4ac','#d8b365']
# miguel_angles = []
# miguel_angles.append(miguel.getOutput('Walking','Hip_Angle_R'))
# miguel_angles.append(miguel.getOutput('Walking','Hip_Angle_L'))
# miguel_angles.append(miguel.getOutput('Walking','Knee_Angle_R'))
# miguel_angles.append(miguel.getOutput('Walking','Knee_Angle_L'))

# subject_angles = []
# subject_angles.append(subject.getOutput('Walking','Hip_Angle_R'))
# subject_angles.append(subject.getOutput('Walking','Hip_Angle_L'))
# subject_angles.append(subject.getOutput('Walking','Knee_Angle_R'))
# subject_angles.append(subject.getOutput('Walking','Knee_Angle_L'))

# plt.subplot(1,2,1)
# bplot2 = plt.boxplot([miguel_angles[0],miguel_angles[1],subject_angles[0],subject_angles[1]],labels=["Control R Hip","Control L Hip","Subject R Hip","Subject L Hip"],patch_artist=True)
# plt.ylabel("Max Joint Angle (deg)")
# plt.title("Hip Angles Comparison while walking")
# j=0
# for i in bplot2['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1

# plt.subplot(1,2,2)
# bplot3 = plt.boxplot([miguel_angles[2],miguel_angles[3],subject_angles[2],subject_angles[3]],labels=["Control R Knee","Control L Knee","Subject R Knee","Subject L Knee"],patch_artist=True)
# plt.ylabel("Max Joint Angle (deg)")
# plt.title("Knee Angles Comparison while walking")
# j=0
# for i in bplot3['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# #intra subject
# print("Control R/L Hip: %.6f\nSubject R/L Hip: %.6f\n"%(ranksums(miguel_angles[0],miguel_angles[1])[1],ranksums(subject_angles[0],subject_angles[1])[1]))
# print("Control R/L Knee: %.6f\nSubject R/L Knee: %.6f\n"%(ranksums(miguel_angles[2],miguel_angles[3])[1],ranksums(subject_angles[2],subject_angles[3])[1]))
# #inter subject
# print("Control/Subject R Hip: %.6f\nControl/Subject L Hip: %.6f\n"%(ranksums(miguel_angles[0],subject_angles[0])[1],ranksums(miguel_angles[1],subject_angles[1])[1]))
# print("Control/Subject R Knee: %.6f\nControl/Subject L Knee: %.6f\n"%(ranksums(miguel_angles[2],subject_angles[2])[1],ranksums(miguel_angles[3],subject_angles[3])[1]))

# plt.show()
# # %% Step dims
# bplot = plt.boxplot([miguel.getOutput('Walking','step_height'),subject.getOutput('Walking','step_height'),miguel.getOutput('Walking','step_length'),subject.getOutput('Walking','step_length'),miguel.getOutput('Walking','step_width'),subject.getOutput('Walking','step_width')],labels=["Control Step Height","Subject Step Height","Control Step Length","Subject Step Length","Control Step Width","Subject Step Width"],patch_artist=True)
# a=ranksums(miguel.getOutput('Walking','step_height'),subject.getOutput('Walking','step_height'))
# print("Step Height: %.6f\n"%a[1])
# a=ranksums(miguel.getOutput('Walking','step_length'),subject.getOutput('Walking','step_length'))
# print("Step Length: %.6f\n"%a[1])
# a=ranksums(miguel.getOutput('Walking','step_width'),subject.getOutput('Walking','step_width'))
# print("Step Width: %.6f\n"%a[1])
# plt.ylabel("Normalized Distance (m/m)")
# j=0
# for i in bplot['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.show()

# # %% Double support
# control_ds=miguel.getOutput('Walking','double_stance')
# subject_ds=subject.getOutput('Walking','double_stance')
# data=[control_ds,subject_ds]
# bplot_ds = plt.boxplot(data,labels=["Control DS","Subject DS"],patch_artist=True)
# plt.title("Double Stance Comparison")
# plt.ylabel("Fraction of Gait Cycle")
# a=ranksums(control_ds,subject_ds)
# print("Double Stance p: %.5f"%a[1])
# plt.show()

# # %% tandem pelvis jerk
# control_jerk=miguel.getOutput('Tandem','pelvis_jerk_step_normalized')
# subject_jerk=subject.getOutput('Tandem','pelvis_jerk_step_normalized')
# # subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlie
# data=[control_jerk,subject_jerk]
# bplot_jerk = plt.boxplot(data,labels=["Control","Subject"],patch_artist=True)
# plt.title("Pelvis Jerk Comparison\nTandem")
# plt.ylabel("Normalized Jerk")
# a=ranksums(control_jerk,subject_jerk)
# print("Pelvis Jerk: %.5f"%a[1])
# plt.show()

# # %% tandem thorax jerk
# control_jerk=miguel.getOutput('Tandem','thorax_jerk_step_normalized')
# subject_jerk=subject.getOutput('Tandem','thorax_jerk_step_normalized')
# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlie
# data=[control_jerk,subject_jerk]
# bplot_jerk = plt.boxplot(data,labels=["Control","Subject"],patch_artist=True)
# plt.title("Thorax Jerk Comparison\nTandem")
# plt.ylabel("Normalized Jerk")
# a=ranksums(control_jerk,subject_jerk)
# print("Thorax Jerk: %.5f"%a[1])
# plt.show()

# # %% NATURAL GRASPING
# path_to_control =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
# control = SessionDataObject(path_to_control,False,1.7)
# # %%
# # labels=['lightblue','pink','lightblue','pink','lightblue','pink']
# labels=['#5ab4ac','#d8b365','#5ab4ac','#d8b365','#5ab4ac','#d8b365']
# plt.subplot(1,5,1)
# sub_pointer=subject.getOutput('NG','Pointer Dist')
# sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
# sub_thumb=subject.getOutput('NG','Thumb Dist')
# sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
# ctl_pointer=control.getOutput('NG','Pointer Dist')
# ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
# ctl_thumb=control.getOutput('NG','Thumb Dist')
# ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
# data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
# a = ranksums(data[0],data[1])
# print("Path Length:")
# print("Pointer p: %.5f"%a[1])
# a = ranksums(data[2],data[3])
# print("Thumb p: %.5f"%a[1])
# bplot1 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# # plt.violinplot(data)
# j=0
# for i in bplot1['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("Distance\n(% of straight line)")
# plt.box(False)
# plt.subplot(1,5,2)
# sub_pointer=subject.getOutput('NG','Pointer Accel')
# sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
# sub_thumb=subject.getOutput('NG','Thumb Accel')
# sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
# ctl_pointer=control.getOutput('NG','Pointer Accel')
# ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
# ctl_thumb=control.getOutput('NG','Thumb Accel')
# ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
# data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
# a = ranksums(data[0],data[1])
# print("\nAccel:")
# print("Pointer p: %.5f"%a[1])
# a = ranksums(data[2],data[3])
# print("Thumb p: %.5f"%a[1])
# bplot2 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# # bplot2 = plt.violinplot(data)
# j=0
# for i in bplot2['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("Acceleration\n($mm^2/s$)")
# plt.box(False)

# plt.subplot(1,5,3)
# sub_pointer=subject.getOutput('NG','Pointer Jerk')
# sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
# sub_thumb=subject.getOutput('NG','Thumb Jerk')
# sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
# ctl_pointer=control.getOutput('NG','Pointer Jerk')
# ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
# ctl_thumb=control.getOutput('NG','Thumb Jerk')
# ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
# data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
# a = ranksums(data[0],data[1])
# print("\nJerk:")
# print("Pointer p: %.5f"%a[1])
# a = ranksums(data[2],data[3])
# print("Thumb p: %.5f"%a[1])
# bplot3 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot3['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("Jerk\n($mm^3/s$)")
# plt.box(False)

# plt.subplot(1,5,4)
# sub_mga = subject.getOutput('NG','MGA')
# sub_mga = sub_mga[~np.isnan(sub_mga)]
# ctl_mga = control.getOutput('NG','MGA')
# ctl_mga = ctl_mga[~np.isnan(ctl_mga)]
# data=[ctl_mga,sub_mga]
# a = ranksums(data[0],data[1])
# print("MGA p: %.5f"%a[1])
# bplot4 = plt.boxplot(data,labels=['Control','Subject'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot4['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("MGA\n(mm)")
# plt.box(False)

# plt.subplot(1,5,5)
# sub_mgat = subject.getOutput('NG','MGA_t')
# sub_mgat = sub_mgat[~np.isnan(sub_mgat)]
# sub_mgat = sub_mgat[~(sub_mgat<0.55)]
# ctl_mgat = control.getOutput('NG','MGA_t')
# ctl_mgat = ctl_mgat[~np.isnan(ctl_mgat)]
# ctl_mgat = ctl_mgat[~(ctl_mgat<0.55)]
# data=[ctl_mgat,sub_mgat]
# a = ranksums(data[0],data[1])
# print("MGA Time p: %.5f"%a[1])
# bplot5 = plt.boxplot(data,labels=['Control','Subject'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot5['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("MGA Time\n% of trial")
# plt.box(False)
# plt.show()