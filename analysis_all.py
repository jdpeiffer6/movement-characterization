# %% Import data
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums
from scipy import stats

path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\003\\2022-02-11"
subject = SessionDataObject(path_to_subject,False,1.7,walking=False,ng=False)
path_to_subject2 =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\004\\2022-04-12"
subject2 = SessionDataObject(path_to_subject2,False,1.7,walking=False,ng=False)
path_to_subject3 =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\005\\2022-05-19"
subject3 = SessionDataObject(path_to_subject3,False,1.829,walking=False,ng=False)
path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=False,ng=False)
labels=['#1BD9DE','#FCBF29','#FE8821','#d8b365','#5ab4ac','#d8b365']

# %% Pelvis jerk
control_jerk=miguel.getOutput('Walking','pelvis_jerk_step_normalized')
subject_jerk=subject.getOutput('Walking','pelvis_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Walking','pelvis_jerk_step_normalized')
subject3_jerk=subject3.getOutput('Walking','pelvis_jerk_step_normalized')

subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier

data=[control_jerk,subject_jerk,subject2_jerk,subject3_jerk]
# plt.subplot(1,6,1)
bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2","S3"],patch_artist=True)
plt.title("Pelvis Jerk")
plt.ylabel("Normalized Jerk")
print("Pelvis Jerk: ")
print(f'C: {np.mean(control_jerk)} ({np.std(control_jerk)})')
print(f'S1: {np.mean(subject_jerk)} ({np.std(subject_jerk)})')
print(f'S2: {np.mean(subject2_jerk)} ({np.std(subject2_jerk)})')
print(f'S3: {np.mean(subject3_jerk)} ({np.std(subject3_jerk)})')

j=0
for i in bplot_jerk['boxes']:
    i.set_facecolor(labels[j])
    j+=1

# %% QQ
# stats.probplot(data[0],dist="norm",plot=plt)
# plt.show()
# stats.probplot(data[1],dist="norm",plot=plt)
# plt.show()
a=stats.ttest_ind(control_jerk,subject_jerk)
print("Pelvis Jerk S1: %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject2_jerk)
print("Pelvis Jerk S2: %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject3_jerk)
print("Pelvis Jerk S3: %.5f"%a[1])
# plt.show()

# %% Thorax jerk
control_jerk=miguel.getOutput('Walking','thorax_jerk_step_normalized')
subject_jerk=subject.getOutput('Walking','thorax_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Walking','thorax_jerk_step_normalized')
subject3_jerk=subject3.getOutput('Walking','thorax_jerk_step_normalized')

subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier

data=[control_jerk,subject_jerk,subject2_jerk,subject3_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2","S3"],patch_artist=True)
plt.title("Thorax Jerk Comparison")
plt.ylabel("Normalized Jerk")
a=stats.ttest_ind(control_jerk,subject_jerk)
print("Thorax Jerk p: %.5f"%a[1])
plt.show()
# %% Walking paths
control_devangle=miguel.getOutput('Walking','walking_angle_deviation')
subject_devangle=subject.getOutput('Walking','walking_angle_deviation')
subject2_devangle=subject2.getOutput('Walking','walking_angle_deviation')
subject3_devangle=subject3.getOutput('Walking','walking_angle_deviation')

control_devangle=control_devangle[~np.isnan(control_devangle)]
subject_devangle=subject_devangle[~np.isnan(subject_devangle)]
subject2_devangle=subject2_devangle[~np.isnan(subject2_devangle)]
subject3_devangle=subject3_devangle[~np.isnan(subject3_devangle)]
data=[control_devangle,subject_devangle,subject2_devangle,subject3_devangle]
bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2","S3"],patch_artist=True)
plt.title("Wakling Deviation Angles Comparison")
plt.ylabel("Angle (Degrees)")
a=stats.ttest_ind(control_devangle,subject_devangle)
print("Angle Dev S1 p: %.5f"%a[1])
a=stats.ttest_ind(control_devangle,subject3_devangle)
print("Angle Dev S3 p: %.5f"%a[1])
plt.show()

# %% Joint angles
miguel_angles = []
miguel_angles.append(miguel.getOutput('Walking','Hip_Angle_R'))
miguel_angles.append(miguel.getOutput('Walking','Hip_Angle_L'))
miguel_angles.append(miguel.getOutput('Walking','Knee_Angle_R'))
miguel_angles.append(miguel.getOutput('Walking','Knee_Angle_L'))

subject_angles = []
subject_angles.append(subject.getOutput('Walking','Hip_Angle_R'))
subject_angles.append(subject.getOutput('Walking','Hip_Angle_L'))
subject_angles.append(subject.getOutput('Walking','Knee_Angle_R'))
subject_angles.append(subject.getOutput('Walking','Knee_Angle_L'))
print(f'S1: {np.mean(subject_angles[2])} - {np.mean(subject_angles[3])}')

subject2_angles = []
subject2_angles.append(subject2.getOutput('Walking','Hip_Angle_R'))
subject2_angles.append(subject2.getOutput('Walking','Hip_Angle_L'))
subject2_angles.append(subject2.getOutput('Walking','Knee_Angle_R'))
subject2_angles.append(subject2.getOutput('Walking','Knee_Angle_L'))
print(f'S2: {np.mean(subject2_angles[2])} - {np.mean(subject2_angles[3])}')

subject3_angles = []
subject3_angles.append(subject3.getOutput('Walking','Hip_Angle_R'))
subject3_angles.append(subject3.getOutput('Walking','Hip_Angle_L'))
subject3_angles.append(subject3.getOutput('Walking','Knee_Angle_R'))
subject3_angles.append(subject3.getOutput('Walking','Knee_Angle_L'))
print(f'S3: {np.mean(subject3_angles[2])} - {np.mean(subject3_angles[3])}')


bplot2 = plt.boxplot([np.concatenate((miguel_angles[0],miguel_angles[1])),np.concatenate((subject_angles[0],subject_angles[1])),np.concatenate((subject2_angles[0],subject2_angles[1])),np.concatenate((subject3_angles[0],subject3_angles[1]))],labels=["Control","S1","S2","S3"],patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Hip Angles")
j=0
for i in bplot2['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()

bplot3 = plt.boxplot([np.concatenate((miguel_angles[2],miguel_angles[3])),np.concatenate((subject_angles[2],subject_angles[3])),np.concatenate((subject2_angles[2],subject2_angles[3])),np.concatenate((subject3_angles[2],subject3_angles[3]))],labels=["Control","S1","S2","S3"],patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Knee Angles")
j=0
for i in bplot3['boxes']:
    i.set_facecolor(labels[j])
    j+=1

#TODO: THIS IS WHERE I STOPPED
print('\nAngles:')
#intra subject
print("Control R/L Hip: %.6f\nSubject R/L Hip: %.6f\nSubject 2 R/L Hip: %.6f\n"%(stats.ttest_ind(miguel_angles[0],miguel_angles[1])[1],stats.ttest_ind(subject_angles[0],subject_angles[1])[1],stats.ttest_ind(subject2_angles[0],subject2_angles[1])[1]))
print("Control R/L Knee: %.6f\nSubject R/L Knee: %.6f\nSubject 2 R/L Knee: %.6f\n"%(stats.ttest_ind(miguel_angles[2],miguel_angles[3])[1],stats.ttest_ind(subject_angles[2],subject_angles[3])[1],stats.ttest_ind(subject2_angles[2],subject2_angles[3])[1]))
#inter subject
print("Control/Subject Hip: %.6f\nControl/Subject2 Hip: %.6f\n"%(stats.ttest_ind(np.concatenate((miguel_angles[0],miguel_angles[1])),np.concatenate((subject_angles[0],subject_angles[1])))[1],stats.ttest_ind(np.concatenate((miguel_angles[0],miguel_angles[1])),np.concatenate((subject2_angles[0],subject2_angles[1])))[1]))
print("Control/Subject Knee: %.6f\nControl/Subject2 Knee: %.6f\n"%(stats.ttest_ind(np.concatenate((miguel_angles[2],miguel_angles[3])),np.concatenate((subject_angles[2],subject_angles[3])))[1],stats.ttest_ind(np.concatenate((miguel_angles[2],miguel_angles[3])),np.concatenate((subject2_angles[2],subject2_angles[3])))[1]))

# plt.show()
# %% Step dims
plt.subplot(1,6,4)
plt.title("Step Length")
bplot = plt.boxplot([miguel.getOutput('Walking','step_length'),subject.getOutput('Walking','step_length'),subject2.getOutput('Walking','step_length')],labels=["Control","S1","S2"],patch_artist=True)

a=stats.ttest_ind(miguel.getOutput('Walking','step_length'),subject.getOutput('Walking','step_length'))
print("Step Length: %.6f\n"%a[1])
a=stats.ttest_ind(miguel.getOutput('Walking','step_length'),subject2.getOutput('Walking','step_length'))
print("Step Length2: %.6f\n"%a[1])

plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
# plt.show()

# %% Double support
plt.subplot(1,6,5)
control_ds=miguel.getOutput('Walking','double_stance')
subject_ds=subject.getOutput('Walking','double_stance')
subject2_ds=subject2.getOutput('Walking','double_stance')
data=[control_ds,subject_ds,subject2_ds]
bplot_ds = plt.boxplot(data,patch_artist=True)
plt.title("Double Stance")
plt.ylabel("Fraction of Gait Cycle")
a=stats.ttest_ind(control_ds,subject_ds)
print("Double Stance p: %.5f"%a[1])
a=stats.ttest_ind(control_ds,subject2_ds)
print("Double Stance2 p: %.5f"%a[1])
# bplot_ds
j=0
for i in bplot_ds['boxes']:
    i.set_facecolor(labels[j])
    j+=1
# plt.show()

# %% tandem pelvis jerk
plt.subplot(1,6,6)
control_jerk=miguel.getOutput('Tandem','pelvis_jerk_step_normalized')
subject_jerk=subject.getOutput('Tandem','pelvis_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Tandem','pelvis_jerk_step_normalized')
subject2_jerk=np.delete(subject2_jerk,np.argmax(subject2_jerk))  #removes outlie
subject2_jerk=np.delete(subject2_jerk,np.argmax(subject2_jerk))  #removes outlie
data=[control_jerk,subject_jerk,subject2_jerk]
bplot_jerk = plt.boxplot(data,patch_artist=True)
plt.title("Pelvis Jerk\nTandem")
plt.ylabel("Normalized Jerk")
a=stats.ttest_ind(control_jerk,subject_jerk)
print("Pelvis Jerk: %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject2_jerk)
print("Pelvis Jerk2: %.5f"%a[1])
j=0
for i in bplot_jerk['boxes']:
    i.set_facecolor(labels[j])
    j+=1
import matplotlib.patches as mpatches
pop_a = mpatches.Patch(color=labels[0], label='Control')
pop_b = mpatches.Patch(color=labels[1], label='Subject 1')
pop_c = mpatches.Patch(color=labels[2], label='Subject 2')
plt.legend(handles=[pop_a,pop_b,pop_c])
plt.show()

# %% tandem thorax jerk
# control_jerk=miguel.getOutput('Tandem','thorax_jerk_step_normalized')
# subject_jerk=subject.getOutput('Tandem','thorax_jerk_step_normalized')
# subject2_jerk=subject2.getOutput('Tandem','thorax_jerk_step_normalized')
# subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlie
# data=[control_jerk,subject_jerk,subject2_jerk]
# bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2"],patch_artist=True)
# plt.title("Thorax Jerk\nTandem")
# plt.ylabel("Normalized Jerk")
# a=stats.ttest_ind(control_jerk,subject_jerk)
# print("Thorax Jerk: %.5f"%a[1])
# plt.show()

# # %% NATURAL GRASPING
# path_to_control =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
# control = SessionDataObject(path_to_control,False,1.7)
# # %%
# # labels=['lightblue','pink','lightblue','pink','lightblue','pink']
# labels=['#5ab4ac','#d8b365','#5ab4ac','#d8b365','#5ab4ac','#d8b365']
# plt.subplot(1,6,1)
# sub_pointer=subject.getOutput('NG','Pointer Dist')
# sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
# sub_thumb=subject.getOutput('NG','Thumb Dist')
# sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
# ctl_pointer=control.getOutput('NG','Pointer Dist')
# ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
# ctl_thumb=control.getOutput('NG','Thumb Dist')
# ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
# data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
# a = stats.ttest_ind(data[0],data[1])
# print("Path Length:")
# print("Pointer p: %.5f"%a[1])
# a = stats.ttest_ind(data[2],data[3])
# print("Thumb p: %.5f"%a[1])
# bplot1 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# # plt.violinplot(data)
# j=0
# for i in bplot1['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("Distance\n(% of straight line)")
# plt.box(False)
# plt.subplot(1,6,2)
# sub_pointer=subject.getOutput('NG','Pointer Accel')
# sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
# sub_thumb=subject.getOutput('NG','Thumb Accel')
# sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
# ctl_pointer=control.getOutput('NG','Pointer Accel')
# ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
# ctl_thumb=control.getOutput('NG','Thumb Accel')
# ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
# data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
# a = stats.ttest_ind(data[0],data[1])
# print("\nAccel:")
# print("Pointer p: %.5f"%a[1])
# a = stats.ttest_ind(data[2],data[3])
# print("Thumb p: %.5f"%a[1])
# bplot2 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# # bplot2 = plt.violinplot(data)
# j=0
# for i in bplot2['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("Acceleration\n($mm^2/s$)")
# plt.box(False)

# plt.subplot(1,6,3)
# sub_pointer=subject.getOutput('NG','Pointer Jerk')
# sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
# sub_thumb=subject.getOutput('NG','Thumb Jerk')
# sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
# ctl_pointer=control.getOutput('NG','Pointer Jerk')
# ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
# ctl_thumb=control.getOutput('NG','Thumb Jerk')
# ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
# data = [ctl_pointer,sub_pointer,ctl_thumb,sub_thumb]
# a = stats.ttest_ind(data[0],data[1])
# print("\nJerk:")
# print("Pointer p: %.5f"%a[1])
# a = stats.ttest_ind(data[2],data[3])
# print("Thumb p: %.5f"%a[1])
# bplot3 = plt.boxplot(data, labels=['Control\nPointer','Subject\nPointer','Control\nThumb','Subject\nThumb'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot3['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("Jerk\n($mm^3/s$)")
# plt.box(False)

# plt.subplot(1,6,4)
# sub_mga = subject.getOutput('NG','MGA')
# sub_mga = sub_mga[~np.isnan(sub_mga)]
# ctl_mga = control.getOutput('NG','MGA')
# ctl_mga = ctl_mga[~np.isnan(ctl_mga)]
# data=[ctl_mga,sub_mga]
# a = stats.ttest_ind(data[0],data[1])
# print("MGA p: %.5f"%a[1])
# bplot4 = plt.boxplot(data,labels=['Control','Subject'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot4['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("MGA\n(mm)")
# plt.box(False)

# plt.subplot(1,6,5)
# sub_mgat = subject.getOutput('NG','MGA_t')
# sub_mgat = sub_mgat[~np.isnan(sub_mgat)]
# sub_mgat = sub_mgat[~(sub_mgat<0.55)]
# ctl_mgat = control.getOutput('NG','MGA_t')
# ctl_mgat = ctl_mgat[~np.isnan(ctl_mgat)]
# ctl_mgat = ctl_mgat[~(ctl_mgat<0.55)]
# data=[ctl_mgat,sub_mgat]
# a = stats.ttest_ind(data[0],data[1])
# print("MGA Time p: %.5f"%a[1])
# bplot5 = plt.boxplot(data,labels=['Control','Subject'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot5['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title("MGA Time\n% of trial")
# plt.box(False)
# plt.show()