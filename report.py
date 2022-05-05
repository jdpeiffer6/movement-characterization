# %% Import data
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind 
from scipy import stats

path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=False,ng=False)
path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\003\\2022-02-11"
subject = SessionDataObject(path_to_subject,False,1.7,walking=False,ng=False)
path_to_subject2 =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\004\\2022-04-12"
subject2 = SessionDataObject(path_to_subject2,False,1.7,walking=False,ng=False)

# %% Joint angles
labels=['#130AF1','#17ACE8','#FE8821','#FEA832','#FE8821','#FEA832']
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

subject2_angles[2]=np.delete(subject2_angles[2],np.argmin(subject2_angles[2]))  #removes outlier

print('\nAngles:')
print(f'Hip:\nCR: {np.mean(miguel_angles[0]):.1f}({np.std(miguel_angles[0])}\nCL: {np.mean(miguel_angles[1]):.1f}({np.std(miguel_angles[1])}\n\nS1R: {np.mean(subject_angles[0]):.1f}({np.std(subject_angles[0])}\nS1L: {np.mean(subject_angles[1]):.1f}({np.std(subject_angles[1])}\n\nS2R: {np.mean(subject2_angles[0]):.1f}({np.std(subject2_angles[0])}\nS2L: {np.mean(subject2_angles[1]):.1f}({np.std(subject2_angles[1])}\n\n')
print(f'Knee:\nCR: {np.mean(miguel_angles[2]):.1f}({np.std(miguel_angles[2])}\nCL: {np.mean(miguel_angles[3]):.1f}({np.std(miguel_angles[3])}\n\nS1R: {np.mean(subject_angles[2]):.1f}({np.std(subject_angles[2])}\nS1L: {np.mean(subject_angles[3]):.1f}({np.std(subject_angles[3])}\n\nS2R: {np.mean(subject2_angles[2]):.1f}({np.std(subject2_angles[2])}\nS2L: {np.mean(subject2_angles[3]):.1f}({np.std(subject2_angles[3])}\n\n')
#intra subject
print("P values:\n")
print("Control R/L Hip: %.6f\nSubject R/L Hip: %.6f\nSubject 2 R/L Hip: %.6f\n"%(stats.ttest_ind(miguel_angles[0],miguel_angles[1])[1],stats.ttest_ind(subject_angles[0],subject_angles[1])[1],stats.ttest_ind(subject2_angles[0],subject2_angles[1])[1]))
print("Control R/L Knee: %.6f\nSubject R/L Knee: %.6f\nSubject 2 R/L Knee: %.6f\n"%(stats.ttest_ind(miguel_angles[2],miguel_angles[3])[1],stats.ttest_ind(subject_angles[2],subject_angles[3])[1],stats.ttest_ind(subject2_angles[2],subject2_angles[3])[1]))
#inter subject
print("Control/Subject Hip: %.6f\nControl/Subject2 Hip: %.6f\n"%(stats.ttest_ind(np.concatenate((miguel_angles[0],miguel_angles[1])),np.concatenate((subject_angles[0],subject_angles[1])))[1],stats.ttest_ind(np.concatenate((miguel_angles[0],miguel_angles[1])),np.concatenate((subject2_angles[0],subject2_angles[1])))[1]))
print("Control/Subject Knee: %.6f\nControl/Subject2 Knee: %.6f\n"%(stats.ttest_ind(np.concatenate((miguel_angles[2],miguel_angles[3])),np.concatenate((subject_angles[2],subject_angles[3])))[1],stats.ttest_ind(np.concatenate((miguel_angles[2],miguel_angles[3])),np.concatenate((subject2_angles[2],subject2_angles[3])))[1]))

plt.subplot(1,2,1)
hips = plt.boxplot([miguel_angles[0],miguel_angles[1],subject_angles[0],subject_angles[1],subject2_angles[0],subject2_angles[1]],positions=[1,2,4,5,7,8],labels=["Control L","Control R","S1 L","S1 R","S2 L","S2 R"], patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Hip Angles")
j=0
for i in hips['boxes']:
    i.set_facecolor(labels[j])
    j+=1
    
plt.subplot(1,2,2)
knees = plt.boxplot([miguel_angles[2],miguel_angles[3],subject_angles[2],subject_angles[3],subject2_angles[2],subject2_angles[3]],positions=[1,2,4,5,7,8],labels=["Control R","Control L","S1 R","S1 L","S2 R","S2 L"],patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Knee Angles")
j=0
for i in knees['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()

# %% Step dims
labels=['#130AF1','#FE8821','#FE8821','#FEA832']
plt.subplot(1,3,1)
bplot = plt.boxplot([miguel.getOutput('Walking','step_length'),subject.getOutput('Walking','step_length'),subject2.getOutput('Walking','step_length')],labels=["Control","S1","S2"],patch_artist=True)
plt.title("Step Length")
a=stats.ttest_ind(miguel.getOutput('Walking','step_length'),subject.getOutput('Walking','step_length'))
print("Step Length p = %.6f\n"%a[1])
a=stats.ttest_ind(miguel.getOutput('Walking','step_length'),subject2.getOutput('Walking','step_length'))
print("Step Length 2 p = %.6f\n"%a[1])
plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1

plt.subplot(1,3,2)
bplot = plt.boxplot([miguel.getOutput('Walking','step_width'),subject.getOutput('Walking','step_width'),subject2.getOutput('Walking','step_width')],labels=["Control","S1","S2"],patch_artist=True)
plt.title("Step width")
a=stats.ttest_ind(miguel.getOutput('Walking','step_width'),subject.getOutput('Walking','step_width'))
print("Step width p = %.6f\n"%a[1])
a=stats.ttest_ind(miguel.getOutput('Walking','step_width'),subject2.getOutput('Walking','step_width'))
print("Step width 2 p = %.6f\n"%a[1])
plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1

plt.subplot(1,3,3)
bplot = plt.boxplot([miguel.getOutput('Walking','step_height'),subject.getOutput('Walking','step_height'),subject2.getOutput('Walking','step_height')],labels=["Control","S1","S2"],patch_artist=True)
plt.title("Step height")
a=stats.ttest_ind(miguel.getOutput('Walking','step_height'),subject.getOutput('Walking','step_height'))
print("Step height p = %.6f\n"%a[1])
a=stats.ttest_ind(miguel.getOutput('Walking','step_height'),subject2.getOutput('Walking','step_height'))
print("Step height 2 p = %.6f\n"%a[1])
plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()
# %% Pelvis jerk
control_jerk=miguel.getOutput('Walking','pelvis_jerk_step_normalized')
subject_jerk=subject.getOutput('Walking','pelvis_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Walking','pelvis_jerk_step_normalized')

subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier

plt.subplot(1,2,1)
data=[control_jerk,subject_jerk,subject2_jerk]
jerkz = plt.boxplot(data,labels=["Control","S1","S2"],patch_artist=True)
j=0
for i in jerkz['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Pelvis Jerk")
plt.ylabel("Normalized Jerk")
print("Pelvis Jerk: ")
print(f'C: {np.mean(control_jerk)} ({np.std(control_jerk)})')
print(f'S1: {np.mean(subject_jerk)} ({np.std(subject_jerk)})')
print(f'S2: {np.mean(subject2_jerk)} ({np.std(subject2_jerk)})')

# QQ
# stats.probplot(data[0],dist="norm",plot=plt)
# plt.show()
# stats.probplot(data[1],dist="norm",plot=plt)
# plt.show()
a=stats.ttest_ind(control_jerk,subject_jerk)
print("Pelvis Jerk S1 p = %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject2_jerk)
print("Pelvis Jerk S2 p = %.5f"%a[1])
# plt.show()

# %% Thorax jerk
plt.subplot(1,2,2)
control_jerk=miguel.getOutput('Walking','thorax_jerk_step_normalized')
subject_jerk=subject.getOutput('Walking','thorax_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Walking','thorax_jerk_step_normalized')

subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject2_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject2_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier

data=[control_jerk,subject_jerk,subject2_jerk]
print("Thorax Jerk: ")
print(f'C: {np.mean(control_jerk):.0f} ({np.std(control_jerk):.0f})')
print(f'S1: {np.mean(subject_jerk):.0f} ({np.std(subject_jerk):.0f})')
print(f'S2: {np.mean(subject2_jerk):.0f} ({np.std(subject2_jerk):.0f})')
bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2"],patch_artist=True)
j=0
for i in bplot_jerk['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Thorax Jerk Comparison")
plt.ylabel("Normalized Jerk")
a=stats.ttest_ind(control_jerk,subject_jerk)
print("Thorax Jerk p: %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject2_jerk)
print("Thorax Jerk 2 p: %.5f"%a[1])
plt.show()
# # %% Walking paths
# control_devangle=miguel.getOutput('Walking','walking_angle_deviation')
# subject_devangle=subject.getOutput('Walking','walking_angle_deviation')
# subject2_devangle=subject2.getOutput('Walking','walking_angle_deviation')
# control_devangle=control_devangle[~np.isnan(control_devangle)]
# subject_devangle=subject_devangle[~np.isnan(subject_devangle)]
# subject2_devangle=subject2_devangle[~np.isnan(subject2_devangle)]
# data=[control_devangle,subject_devangle,subject2_devangle]
# bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2"],patch_artist=True)
# plt.title("Wakling Deviation Angles Comparison")
# plt.ylabel("Angle (Degrees)")
# a=stats.ttest_ind(control_devangle,subject_devangle)
# print("Angle Dev p: %.5f"%a[1])
# plt.show()



# %% Double support
control_ds=miguel.getOutput('Walking','double_stance')
subject_ds=subject.getOutput('Walking','double_stance')
subject2_ds=subject2.getOutput('Walking','double_stance')
print("Thorax Jerk: ")
print(f'C: {np.mean(control_ds):.2f} ({np.std(control_ds):.2f})')
print(f'S1: {np.mean(subject_ds):.2f} ({np.std(subject_ds):.2f})')
print(f'S2: {np.mean(subject2_ds):.2f} ({np.std(subject2_ds):.2f})')
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
plt.show()

# %% tandem pelvis jerk
plt.subplot(1,2,1)
control_jerk=miguel.getOutput('Tandem','pelvis_jerk_step_normalized')
subject_jerk=subject.getOutput('Tandem','pelvis_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Tandem','pelvis_jerk_step_normalized')
subject2_jerk=np.delete(subject2_jerk,np.argmax(subject2_jerk))  #removes outlie
subject2_jerk=np.delete(subject2_jerk,np.argmax(subject2_jerk))  #removes outlie
print("Tandem Pelvis Jerk: ")
print(f'C: {np.mean(control_jerk):.0f} ({np.std(control_jerk):.0f})')
print(f'S1: {np.mean(subject_jerk):.0f} ({np.std(subject_jerk):.0f})')
print(f'S2: {np.mean(subject2_jerk):.0f} ({np.std(subject2_jerk):.0f})')
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
# plt.show()

# %% tandem thorax jerk
plt.subplot(1,2,2)
control_jerk=miguel.getOutput('Tandem','thorax_jerk_step_normalized')
subject_jerk=subject.getOutput('Tandem','thorax_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Tandem','thorax_jerk_step_normalized')
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlie
print("Tandem Thorax Jerk: ")
print(f'C: {np.mean(control_jerk):.0f} ({np.std(control_jerk):.0f})')
print(f'S1: {np.mean(subject_jerk):.0f} ({np.std(subject_jerk):.0f})')
print(f'S2: {np.mean(subject2_jerk):.0f} ({np.std(subject2_jerk):.0f})')
data=[control_jerk,subject_jerk,subject2_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2"],patch_artist=True)
j=0
for i in bplot_jerk['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Thorax Jerk\nTandem")
plt.ylabel("Normalized Jerk")
a=stats.ttest_ind(control_jerk,subject_jerk)
print("Thorax Jerk: %.5f"%a[1])
plt.show()

# %% NATURAL GRASPING
path_to_control =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
control = SessionDataObject(path_to_control,False,1.7)
# labels=['lightblue','pink','lightblue','pink','lightblue','pink']
# labels=['#5ab4ac','#d8b365','#5ab4ac','#d8b365','#5ab4ac','#d8b365']
# labels=['#130AF1','#17ACE8','#FE8821','#FEA832','#FE8821','#FEA832']
labels=['#130AF1','#FE8821','#FE8821','#17ACE8','#FEA832','#FEA832']
# plt.subplot(1,6,1)
sub_pointer=subject.getOutput('NG','Pointer Dist')
sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
sub_thumb=subject.getOutput('NG','Thumb Dist')
sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
ctl_pointer=control.getOutput('NG','Pointer Dist')
ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
ctl_thumb=control.getOutput('NG','Thumb Dist')
ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
sub2_pointer=subject2.getOutput('NG','Pointer Dist')
sub2_pointer=sub2_pointer[~np.isnan(sub2_pointer)]
sub2_thumb=subject2.getOutput('NG','Thumb Dist')
sub2_thumb=sub2_thumb[~np.isnan(sub2_thumb)]

data = [ctl_pointer,sub_pointer,sub2_pointer,ctl_thumb,sub_thumb,sub2_thumb]
print("Pointer Path: ")
print(f'C: {np.mean(ctl_pointer):.2f} ({np.std(ctl_pointer):.2f})')
print(f'S1: {np.mean(sub_pointer):.2f} ({np.std(sub_pointer):.2f})')
print(f'S2: {np.mean(sub2_pointer):.2f} ({np.std(sub2_pointer):.2f})')
print("Thumb Path: ")
print(f'C: {np.mean(ctl_thumb):.2f} ({np.std(ctl_thumb):.2f})')
print(f'S1: {np.mean(sub_thumb):.2f} ({np.std(sub_thumb):.2f})')
print(f'S2: {np.mean(sub2_thumb):.2f} ({np.std(sub2_thumb):.2f})')
a = stats.ttest_ind(data[0],data[1])
print("Path Length:")
print("Pointer p: %.5f"%a[1])
a = stats.ttest_ind(data[2],data[3])
print("Thumb p: %.5f"%a[1])
bplot1 = plt.boxplot(data, labels=['C P','S1 P','S2 P','C T','S1 T','S2 T'],patch_artist=True,widths=0.5)
# plt.violinplot(data)
j=0
for i in bplot1['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Distance\n(% of straight line)")
plt.box(False)
plt.show()
# # plt.subplot(1,6,2)
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
# plt.show()

# plt.subplot(1,6,3)
sub_pointer=subject.getOutput('NG','Pointer Jerk')
sub_pointer=sub_pointer[~np.isnan(sub_pointer)]
sub_thumb=subject.getOutput('NG','Thumb Jerk')
sub_thumb=sub_thumb[~np.isnan(sub_thumb)]
sub2_pointer=subject2.getOutput('NG','Pointer Jerk')
sub2_pointer=sub2_pointer[~np.isnan(sub2_pointer)]
sub2_thumb=subject2.getOutput('NG','Thumb Jerk')
sub2_thumb=sub2_thumb[~np.isnan(sub2_thumb)]
ctl_pointer=control.getOutput('NG','Pointer Jerk')
ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
ctl_thumb=control.getOutput('NG','Thumb Jerk')
ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
data = [ctl_pointer,sub_pointer,sub2_pointer,ctl_thumb,sub_thumb,sub2_thumb]
print("Pointer Jerk: ")
print(f'C: {np.mean(ctl_pointer):.0f} ({np.std(ctl_pointer):.0f})')
print(f'S1: {np.mean(sub_pointer):.0f} ({np.std(sub_pointer):.0f})')
print(f'S2: {np.mean(sub2_pointer):.0f} ({np.std(sub2_pointer):.0f})')
print("Thumb Jerk: ")
print(f'C: {np.mean(ctl_thumb):.0f} ({np.std(ctl_thumb):.0f})')
print(f'S1: {np.mean(sub_thumb):.0f} ({np.std(sub_thumb):.0f})')
print(f'S2: {np.mean(sub2_thumb):.0f} ({np.std(sub2_thumb):.0f})')
print("\nJerk:")
a = stats.ttest_ind(data[0],data[1])
print("Pointer p: %.5f"%a[1])
a = stats.ttest_ind(data[0],data[2])
print("Pointer p: %.5f"%a[1])
a = stats.ttest_ind(data[3],data[4])
print("Thumb p: %.5f"%a[1])
a = stats.ttest_ind(data[3],data[5])
print("Thumb p: %.5f"%a[1])

bplot3 = plt.boxplot(data, labels=['C P','S1 P','S2 P','C T','S1 T','S2 T'],patch_artist=True,widths=0.5)
j=0
for i in bplot3['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("Jerk")
plt.box(False)
plt.show()

# plt.subplot(1,6,4)
sub_mga = subject.getOutput('NG','MGA')
sub_mga = sub_mga[~np.isnan(sub_mga)]
sub2_mga = subject2.getOutput('NG','MGA')
sub2_mga = sub2_mga[~np.isnan(sub2_mga)]
ctl_mga = control.getOutput('NG','MGA')
ctl_mga = ctl_mga[~np.isnan(ctl_mga)]
data=[ctl_mga,sub_mga,sub2_mga]
a = stats.ttest_ind(data[0],data[1])
print("MGA p: %.5f"%a[1])
a = stats.ttest_ind(data[0],data[2])
print("MGA p: %.5f"%a[1])
bplot4 = plt.boxplot(data,labels=['Control','Subject 1', 'Subject 2'],patch_artist=True,widths=0.5)
j=0
for i in bplot4['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("MGA\n(mm)")
plt.box(False)
plt.show()

# plt.subplot(1,6,5)
sub_mgat = subject.getOutput('NG','MGA_t')
sub_mgat = sub_mgat[~np.isnan(sub_mgat)]
sub_mgat = sub_mgat[~(sub_mgat<0.55)]
sub2_mgat = subject2.getOutput('NG','MGA_t')
sub2_mgat = sub2_mgat[~np.isnan(sub2_mgat)]
sub2_mgat = sub2_mgat[~(sub2_mgat<0.55)]
ctl_mgat = control.getOutput('NG','MGA_t')
ctl_mgat = ctl_mgat[~np.isnan(ctl_mgat)]
ctl_mgat = ctl_mgat[~(ctl_mgat<0.55)]
data=[ctl_mgat,sub_mgat,sub2_mgat]
print("MGA t: ")
print(f'C: {np.mean(ctl_mgat):.2f} ({np.std(ctl_mgat):.2f})')
print(f'S1: {np.mean(sub_mgat):.2f} ({np.std(sub_mgat):.2f})')
print(f'S2: {np.mean(sub2_mgat):.2f} ({np.std(sub2_mgat):.2f})')
a = stats.ttest_ind(data[0],data[1])
print("MGA Time p: %.5f"%a[1])
a = stats.ttest_ind(data[0],data[2])
print("MGA Time p: %.5f"%a[1])
bplot5 = plt.boxplot(data,labels=['Control','Subject 1','Subject 2'],patch_artist=True,widths=0.5)
j=0
for i in bplot5['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("MGA Time\n% of trial")
plt.box(False)
plt.show()

# %% 9 HOLE PEG
path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\004\\2022-04-12"
subject = SessionDataObject(path_to_subject,False,1.7,walking=True,ng=False)
path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=True,ng=False)
# %% TMT
labels=['#130AF1','#17ACE8','#FE8821','#FEA832']
tmt = [miguel.output_data.data['9HPL']['TMT'],miguel.output_data.data['9HPR']['TMT'],subject.output_data.data['9HPL']['TMT'],subject.output_data.data['9HPR']['TMT']]
bplot = plt.boxplot(tmt,labels=["Control L","Control R","CM L","CM R"],patch_artist=True)
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.ylabel("Time (s)")
plt.title("Total Movement Time")
print("TMT: ")
print(f'C L: {np.mean(tmt[0]):.2f} ({np.std(tmt[0]):.2f})')
print(f'C R: {np.mean(tmt[1]):.2f} ({np.std(tmt[1]):.2f})')
print(f'S L: {np.mean(tmt[2]):.2f} ({np.std(tmt[2]):.2f})')
print(f'S R: {np.mean(tmt[3]):.2f} ({np.std(tmt[3]):.2f})')
plt.show()
a=ttest_ind(tmt[0],tmt[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(tmt[2],tmt[3])
print(f'CM R/L p: {a[1]}')

a=ttest_ind(tmt[0],tmt[2])
print(f'L p: {a[1]}')
a=ttest_ind(tmt[1],tmt[3])
print(f'R p: {a[1]}')

# %% Single peg MT
spt = [miguel.getOutput('9HPL','Single Peg MT'),miguel.getOutput('9HPR','Single Peg MT'),subject.getOutput('9HPL','Single Peg MT'),subject.getOutput('9HPR','Single Peg MT')]
bplot = plt.boxplot(spt,labels=["Control L","Control R","CM L","CM R"],patch_artist=True)
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.ylabel("Time (s)")
plt.title("Single Peg Movement Time")
print("TMT: ")
print(f'C L: {np.mean(spt[0]):.2f} ({np.std(spt[0]):.2f})')
print(f'C R: {np.mean(spt[1]):.2f} ({np.std(spt[1]):.2f})')
print(f'S L: {np.mean(spt[2]):.2f} ({np.std(spt[2]):.2f})')
print(f'S R: {np.mean(spt[3]):.2f} ({np.std(spt[3]):.2f})')
plt.show()
a=ttest_ind(spt[0],spt[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(spt[2],spt[3])
print(f'CM R/L p: {a[1]}')

# %% GR Ratio 
grr = [miguel.getOutput('9HPL','GR Ratio'),miguel.getOutput('9HPR','GR Ratio'),subject.getOutput('9HPL','GR Ratio'),subject.getOutput('9HPR','GR Ratio')]
bplot = plt.boxplot(grr,labels=["Control L","Control R","CM L","CM R"],patch_artist=True)
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.ylabel("Ratio s/s")
plt.title("Grasp-Reach Ratio")
print("GRR: ")
print(f'C L: {np.mean(grr[0]):.2f} ({np.std(grr[0]):.2f})')
print(f'C R: {np.mean(grr[1]):.2f} ({np.std(grr[1]):.2f})')
print(f'S L: {np.mean(grr[2]):.2f} ({np.std(grr[2]):.2f})')
print(f'S R: {np.mean(grr[3]):.2f} ({np.std(grr[3]):.2f})')
plt.show()
a=ttest_ind(grr[0],grr[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(grr[2],grr[3])
print(f'CM R/L p: {a[1]}')
a=ttest_ind(grr[0],grr[2])
print(f'L compare p: {a[1]}')
a=ttest_ind(grr[1],grr[2])
print(f'R compare p: {a[1]}')
# %%
ps = [miguel.getOutput('9HPL','Peak Speeds'),miguel.getOutput('9HPR','Peak Speeds'),subject.getOutput('9HPL','Peak Speeds'),subject.getOutput('9HPR','Peak Speeds')]
plt.boxplot(ps,labels=["Control L","Control R","CM L","CM R"])
plt.ylabel("Speed")
plt.title("Peak Speeds")
plt.show()
# %%
ps_t = [miguel.getOutput('9HPL','Peak Speed Time'),miguel.getOutput('9HPR','Peak Speed Time'),subject.getOutput('9HPL','Peak Speed Time'),subject.getOutput('9HPR','Peak Speed Time')]
plt.boxplot(ps_t,labels=["Control L","Control R","CM L","CM R"])
plt.ylabel("Time (s)")
plt.title("Peak Speed Time")
print("Peak Speed: ")
print(f'C L: {np.mean(ps_t[0]):.2f} ({np.std(ps_t[0]):.2f})')
print(f'C R: {np.mean(ps_t[1]):.2f} ({np.std(ps_t[1]):.2f})')
print(f'S L: {np.mean(ps_t[2]):.2f} ({np.std(ps_t[2]):.2f})')
print(f'S R: {np.mean(ps_t[3]):.2f} ({np.std(ps_t[3]):.2f})')
plt.show()
a=ttest_ind(ps_t[0],ps_t[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(ps_t[2],ps_t[3])
print(f'CM R/L p: {a[1]}')
a=ttest_ind(ps_t[0],ps_t[2])
print(f'L p: {a[1]}')
a=ttest_ind(ps_t[1],ps_t[3])
print(f'R p: {a[1]}')
# %%
jerkz = [miguel.getOutput('9HPL','Peak Jerk'),miguel.getOutput('9HPR','Peak Jerk'),subject.getOutput('9HPL','Peak Jerk'),subject.getOutput('9HPR','Peak Jerk')]
jerkz[0]=np.delete(jerkz[0],np.argmax(jerkz[0]))  #removes outlier
bplot = plt.boxplot(jerkz,labels=["Control L","Control R","CM L","CM R"],patch_artist=True)
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.ylabel("Normalized Jerk")
plt.title("Pointer Jerk")
print("Pointer Jerk: ")
print(f'C L: {np.mean(jerkz[0]):.0f} ({np.std(jerkz[0]):.0f})')
print(f'C R: {np.mean(jerkz[1]):.0f} ({np.std(jerkz[1]):.0f})')
print(f'S L: {np.mean(jerkz[2]):.0f} ({np.std(jerkz[2]):.0f})')
print(f'S R: {np.mean(jerkz[3]):.0f} ({np.std(jerkz[3]):.0f})')
plt.show()
a=ttest_ind(jerkz[0],jerkz[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(jerkz[2],jerkz[3])
print(f'CM R/L p: {a[1]}')
a=ttest_ind(jerkz[0],jerkz[2])
print(f'L compare p: {a[1]}')
a=ttest_ind(jerkz[1],jerkz[3])
print(f'R compare p: {a[1]}')
# %%
from functions.helper import getIndexNames
namez = getIndexNames('9HP',subject.marker_task_labels)
for trial in namez:
    data = subject.marker_data[trial]
    data = data * 10**-3
    # remove the Right/Left from data. could just not name the markers this lol
    new_names = []
    for colname in data.columns:
        if colname[0] == 'R':
            colname = colname[1:]
            right = True
            # if right side, then the first peak will be positivie
        elif colname[0] == 'L':
            colname = colname[1:]
            right = False
        new_names.append(colname)
    data.columns = new_names
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(data['THU X'], data['THU Y'], data['THU Z'], 'red')
    plt.show()

namez = getIndexNames('9HP',miguel.marker_task_labels)
for trial in namez:
    data = miguel.marker_data[trial]
    data = data * 10**-3
    # remove the Right/Left from data. could just not name the markers this lol
    new_names = []
    for colname in data.columns:
        if colname[0] == 'R':
            colname = colname[1:]
            right = True
            # if right side, then the first peak will be positivie
        elif colname[0] == 'L':
            colname = colname[1:]
            right = False
        new_names.append(colname)
    data.columns = new_names
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(data['THU X'], data['THU Y'], data['THU Z'], 'blue')
    ax.set_xlabel("")
    plt.show()
# %%



# %%
jerkz = [miguel.getOutput('9HPL','Peak Jerk'),miguel.getOutput('9HPR','Peak Jerk'),subject.getOutput('9HPL','Peak Jerk'),subject.getOutput('9HPR','Peak Jerk')]
jerkz[0]=np.delete(jerkz[0],np.argmax(jerkz[0]))  #removes outlier
bplot = plt.boxplot(jerkz,labels=["Control L","Control R","CM L","CM R"],patch_artist=True)
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.ylabel("Normalized Jerk")
plt.title("Pointer Jerk")
plt.show()
a=ttest_ind(jerkz[0],jerkz[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(jerkz[2],jerkz[3])
print(f'CM R/L p: {a[1]}')
a=ttest_ind(jerkz[0],jerkz[2])
print(f'L compare p: {a[1]}')
a=ttest_ind(jerkz[1],jerkz[3])
print(f'R compare p: {a[1]}')
# %%
from functions.helper import getIndexNames
namez = getIndexNames('9HP',subject.marker_task_labels)
for trial in namez:
    data = subject.marker_data[trial]
    data = data * 10**-3
    # remove the Right/Left from data. could just not name the markers this lol
    new_names = []
    for colname in data.columns:
        if colname[0] == 'R':
            colname = colname[1:]
            right = True
            # if right side, then the first peak will be positivie
        elif colname[0] == 'L':
            colname = colname[1:]
            right = False
        new_names.append(colname)
    data.columns = new_names
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(data['THU X'], data['THU Y'], data['THU Z'], 'red')
    plt.show()

namez = getIndexNames('9HP',miguel.marker_task_labels)
for trial in namez:
    data = miguel.marker_data[trial]
    data = data * 10**-3
    # remove the Right/Left from data. could just not name the markers this lol
    new_names = []
    for colname in data.columns:
        if colname[0] == 'R':
            colname = colname[1:]
            right = True
            # if right side, then the first peak will be positivie
        elif colname[0] == 'L':
            colname = colname[1:]
            right = False
        new_names.append(colname)
    data.columns = new_names
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot3D(data['THU X'], data['THU Y'], data['THU Z'], 'blue')
    ax.set_xlabel("")
    plt.show()
# %%


