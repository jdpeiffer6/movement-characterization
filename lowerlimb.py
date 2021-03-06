# %%
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=True,ng=False)
path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\003\\2022-02-11"
subject = SessionDataObject(path_to_subject,False,1.7,walking=True,ng=False)
path_to_subject2 =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\004\\2022-04-12"
subject2 = SessionDataObject(path_to_subject2,False,1.7,walking=False,ng=False)

# %% pelvis jerk
control_jerk=miguel.getOutput('Walking','pelvis_jerk_step_normalized')
subject_jerk=subject.getOutput('Walking','pelvis_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Walking','pelvis_jerk_step_normalized')

subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier
subject_jerk=np.delete(subject_jerk,np.argmax(subject_jerk))  #removes outlier

data=[control_jerk,subject_jerk,subject2_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","S1","S2"],patch_artist=True)
plt.title("Pelvis Jerk Comparison")
plt.ylabel("Normalized Jerk")
a=ranksums(control_jerk,subject_jerk)
print("Pelvis Jerk: %.5f"%a[1])
a=ranksums(control_jerk,subject2_jerk)
print("Pelvis Jerk: %.5f"%a[1])
plt.show()
# %% thorax jerk
control_jerk=miguel.thorax_jerks
subject_jerk=subject.thorax_jerks
data=[control_jerk,subject_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control RMS","Subject RMS"],patch_artist=True)
plt.title("Thorax Jerk Comparison")
plt.ylabel("Normalized Jerk RMS\n($m^4/s$)")
plt.show()


# %% joint angles
# labels=['lightblue','pink','lightblue','pink','lightblue','pink']
labels=['#5ab4ac','#d8b365','#5ab4ac','#d8b365','#5ab4ac','#d8b365']
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
bplot = plt.boxplot([miguel.markerless_step_height,subject.markerless_step_height,miguel.markerless_step_length,subject.markerless_step_length,miguel.markerless_step_width,subject.markerless_step_width],labels=["Control Step Height","Subject Step Height","Control Step Length","Subject Step Length","Control Step Width","Subject Step Width"],patch_artist=True)
a=ranksums(miguel.markerless_step_height,subject.markerless_step_height)
print("Step Height: %.6f\n"%a[1])
a=ranksums(miguel.markerless_step_length,subject.markerless_step_length)
print("Step Length: %.6f\n"%a[1])
a=ranksums(miguel.markerless_step_width,subject.markerless_step_width)
print("Step Width: %.6f\n"%a[1])
plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()


# %%
# plt.subplot(1,5,1)
# bplot1 = plt.boxplot([miguel_angles[0],subject_angles[0],miguel_angles[1],subject_angles[1]],labels=['Control\nR','Subject\nR','Control\nL','Subject\nL'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot1['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title('Hip Flexion Angles\n(degrees)')
# plt.box(False)

# plt.subplot(1,5,2)
# bplot2 = plt.boxplot([miguel_angles[2],subject_angles[2],miguel_angles[3],subject_angles[3]],labels=['Control\nR','Subject\nR','Control\nL','Subject\nL'],patch_artist=True,widths=0.5)
# j=0
# for i in bplot2['boxes']:
#     i.set_facecolor(labels[j])
#     j+=1
# plt.title('Knee Flexion Angles\n(degrees)')
# plt.box(False)

plt.subplot(1,5,1)
bplot1 = plt.boxplot([list(miguel_angles[0])+list(miguel_angles[1]),list(subject_angles[0])+list(subject_angles[1])],labels=['Control','Subject'],patch_artist=True,widths=0.5)
j=0
for i in bplot1['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title('Hip Flexion Angles\n(degrees)')
plt.box(False)
print(ranksums(list(miguel_angles[0])+list(miguel_angles[1]),list(subject_angles[0])+list(subject_angles[1])))

plt.subplot(1,5,2)
bplot2 = plt.boxplot([list(miguel_angles[2])+list(miguel_angles[3]),list(subject_angles[2])+list(subject_angles[3])],labels=['Control','Subject'],patch_artist=True,widths=0.5)
j=0
for i in bplot2['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title('Knee Flexion Angles\n(degrees)')
plt.box(False)
print(ranksums(list(miguel_angles[2])+list(miguel_angles[3]),list(subject_angles[2])+list(subject_angles[3])))

plt.subplot(1,5,3)
control_jerk=miguel.markerless_output_data['Thorax_Jerk'].values
control_jerk=control_jerk[~np.isnan(control_jerk)]
subject_jerk=subject.markerless_output_data['Thorax_Jerk'].values
subject_jerk=subject_jerk[~np.isnan(subject_jerk)]
bplot3 = plt.boxplot([control_jerk,subject_jerk],labels=['Control','Subject'],patch_artist=True,widths=0.5)
j=0
for i in bplot3['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title('Thorax Jerk\n($m^3/s)$')
plt.box(False)
print(ranksums(control_jerk,subject_jerk))

plt.subplot(1,5,4)
control_jerk=miguel.markerless_output_data['Pelvis_Jerk'].values
control_jerk=control_jerk[~np.isnan(control_jerk)]
subject_jerk=subject.markerless_output_data['Pelvis_Jerk'].values
subject_jerk=subject_jerk[~np.isnan(subject_jerk)]
bplot4 = plt.boxplot([control_jerk,subject_jerk],labels=['Control','Subject'],patch_artist=True,widths=0.5)
j=0
for i in bplot4['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title('Pelvis Jerk\n($m^3/s)$')
plt.box(False)
print(ranksums(control_jerk,subject_jerk))

plt.subplot(1,5,5)
bplot5 = plt.boxplot([miguel.markerless_step_length,subject.markerless_step_length],labels=['Control','Subject'],patch_artist=True,widths=0.5)
j=0
for i in bplot5['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title('Step Length\n(Normalized to height)')
plt.box(False)
print(ranksums(miguel.markerless_step_length,subject.markerless_step_length))
plt.show()
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
