# %% Import data
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

path_to_subject2 =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\004\\2022-04-12"
subject2 = SessionDataObject(path_to_subject2,False,1.7,walking=False,ng=False)
path_to_subject2_post =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\006\\2022-06-20"
subject2_post = SessionDataObject(path_to_subject2_post,False,1.7,walking=False,ng=False)
path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\002\\2022-01-28"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=False,ng=False)
labels=['#1BD9DE','#FCBF29','#FE8821','#d8b365','#5ab4ac','#d8b365']

# %% Pelvis jerk
control_jerk=miguel.getOutput('Walking','pelvis_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Walking','pelvis_jerk_step_normalized')
subject2_post_jerk=subject2_post.getOutput('Walking','pelvis_jerk_step_normalized')

data=[control_jerk,subject2_jerk,subject2_post_jerk]
# plt.subplot(1,6,1)
bplot_jerk = plt.boxplot(data,labels=["Control","S2 Pre","S2 Post"],patch_artist=True)
plt.title("Pelvis Jerk")
plt.ylabel("Normalized Jerk")
print("Pelvis Jerk: ")
print(f'C: {np.mean(control_jerk)} ({np.std(control_jerk)})')
print(f'S2: {np.mean(subject2_jerk)} ({np.std(subject2_jerk)})')
print(f'S2 post: {np.mean(subject2_post_jerk)} ({np.std(subject2_post_jerk)})')

j=0
for i in bplot_jerk['boxes']:
    i.set_facecolor(labels[j])
    j+=1

# %% QQ
# stats.probplot(data[0],dist="norm",plot=plt)
# plt.show()
# stats.probplot(data[1],dist="norm",plot=plt)
# plt.show()
a=stats.ttest_ind(control_jerk,subject2_jerk)
print("Pelvis Jerk control/Pre: %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject2_post_jerk)
print("Pelvis Jerk S2 control/Post: %.5f"%a[1])
a=stats.ttest_ind(subject2_jerk,subject2_post_jerk)
print("Pelvis Jerk S2 pre/post: %.5f"%a[1])
plt.show()

# %% Thorax jerk
control_jerk=miguel.getOutput('Walking','thorax_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Walking','thorax_jerk_step_normalized')
subject2_post_jerk=subject2_post.getOutput('Walking','thorax_jerk_step_normalized')

data=[control_jerk,subject2_jerk,subject2_post_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","S2 Pre","S2 Post"],patch_artist=True)
plt.title("Thorax Jerk Comparison")
plt.ylabel("Normalized Jerk")
a=stats.ttest_ind(subject2_jerk,subject2_post_jerk)
print("Thorax Jerk pre/post p: %.5f"%a[1])
plt.show()

# %% Walking paths
control_devangle=miguel.getOutput('Walking','walking_angle_deviation')
subject2_devangle=subject2.getOutput('Walking','walking_angle_deviation')
subject2_post_devangle=subject2_post.getOutput('Walking','walking_angle_deviation')

control_devangle=control_devangle[~np.isnan(control_devangle)]
subject2_devangle=subject2_devangle[~np.isnan(subject2_devangle)]
subject2_post_devangle=subject2_post_devangle[~np.isnan(subject2_post_devangle)]
data=[control_devangle,subject2_devangle,subject2_post_devangle]
bplot_jerk = plt.boxplot(data,labels=["Control","S2 Pre","S2 Post"],patch_artist=True)

plt.title("Wakling Deviation Angles Comparison")
plt.ylabel("Angle (Degrees)")
a=stats.ttest_ind(control_devangle,subject2_devangle)
print("Angle Dev ctl/s2 pre p: %.5f"%a[1])
a=stats.ttest_ind(control_devangle,subject2_post_devangle)
print("Angle Dev ctl/s2 post p: %.5f"%a[1])
a=stats.ttest_ind(subject2_devangle,subject2_post_devangle)
print("Angle Dev s2 pre/s2 post p: %.5f"%a[1])
plt.show()

# %% Joint angles
miguel_angles = []
miguel_angles.append(miguel.getOutput('Walking','Hip_Angle_R'))
miguel_angles.append(miguel.getOutput('Walking','Hip_Angle_L'))
miguel_angles.append(miguel.getOutput('Walking','Knee_Angle_R'))
miguel_angles.append(miguel.getOutput('Walking','Knee_Angle_L'))

subject2_angles = []
subject2_angles.append(subject2.getOutput('Walking','Hip_Angle_R'))
subject2_angles.append(subject2.getOutput('Walking','Hip_Angle_L'))
subject2_angles.append(subject2.getOutput('Walking','Knee_Angle_R'))
subject2_angles.append(subject2.getOutput('Walking','Knee_Angle_L'))
print(f'S2 pre: {np.mean(subject2_angles[2])} - {np.mean(subject2_angles[3])}')

subject2_post_angles = []
subject2_post_angles.append(subject2_post.getOutput('Walking','Hip_Angle_R'))
subject2_post_angles.append(subject2_post.getOutput('Walking','Hip_Angle_L'))
subject2_post_angles.append(subject2_post.getOutput('Walking','Knee_Angle_R'))
subject2_post_angles.append(subject2_post.getOutput('Walking','Knee_Angle_L'))
print(f'S2 post: {np.mean(subject2_post_angles[2])} - {np.mean(subject2_post_angles[3])}')


bplot2 = plt.boxplot([np.concatenate((miguel_angles[0],miguel_angles[1])),np.concatenate((subject2_angles[0],subject2_angles[1])),np.concatenate((subject2_post_angles[0],subject2_post_angles[1]))],labels=["Control","S2 Pre","S2 Post"],patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Hip Angles")
j=0
for i in bplot2['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()

bplot3 = plt.boxplot([np.concatenate((miguel_angles[2],miguel_angles[3])),np.concatenate((subject2_angles[2],subject2_angles[3])),np.concatenate((subject2_post_angles[2],subject2_post_angles[3]))],labels=["Control","S2 Pre","S2 Post"],patch_artist=True)
plt.ylabel("Max Joint Angle (deg)")
plt.title("Knee Angles")
j=0
for i in bplot3['boxes']:
    i.set_facecolor(labels[j])
    j+=1

print('\nAngles:')
#intra subject
print("Control R/L Hip: %.6f\nSubject 2 Pre R/L Hip: %.6f\nSubject 2 Post R/L Hip: %.6f\n"% \
    (stats.ttest_ind(miguel_angles[0],miguel_angles[1])[1], \
    stats.ttest_ind(subject2_angles[0],subject2_angles[1])[1], \
    stats.ttest_ind(subject2_post_angles[0],subject2_post_angles[1])[1]))

print("Control R/L Knee: %.6f\nSubject 2 Pre R/L Knee: %.6f\nSubject 2 Post R/L Knee: %.6f\n"% \
    (stats.ttest_ind(miguel_angles[2],miguel_angles[3])[1], \
    stats.ttest_ind(subject2_angles[2],subject2_angles[3])[1], \
    stats.ttest_ind(subject2_post_angles[2],subject2_post_angles[3])[1]))
#inter subject
print("Control/Subject 2 Pre Hip: %.6f\nControl/Subject 2 Post Hip: %.6f\n"% \
    (stats.ttest_ind(np.concatenate((miguel_angles[0],miguel_angles[1])), \
    np.concatenate((subject2_angles[0],subject2_angles[1])))[1], \
    stats.ttest_ind(np.concatenate((miguel_angles[0],miguel_angles[1])), \
    np.concatenate((subject2_post_angles[0],subject2_post_angles[1])))[1]))
print("Control/Subject 2 Pre Knee: %.6f\nControl/Subject 2 Post Knee: %.6f\n"% \
    (stats.ttest_ind(np.concatenate((miguel_angles[2],miguel_angles[3])),
    np.concatenate((subject2_angles[2],subject2_angles[3])))[1],
    stats.ttest_ind(np.concatenate((miguel_angles[2],miguel_angles[3])),
    np.concatenate((subject2_post_angles[2],subject2_post_angles[3])))[1]))
# pre/post
print("Pre/Post Knee: %.6f\nPre/Post Hip: %.6f\n"% \
    (stats.ttest_ind(np.concatenate((subject2_post_angles[2],subject2_post_angles[3])),
    np.concatenate((subject2_angles[2],subject2_angles[3])))[1],
    stats.ttest_ind(np.concatenate((subject2_angles[0],subject2_angles[1])),
    np.concatenate((subject2_post_angles[0],subject2_post_angles[1])))[1]))
plt.show()
# %% Step dims
plt.title("Step Length")
bplot = plt.boxplot([miguel.getOutput('Walking','step_length'),
    subject2.getOutput('Walking','step_length'),
    subject2_post.getOutput('Walking','step_length')],
    labels=["Control","S2 Pre","S2 Post"],patch_artist=True)

a=stats.ttest_ind(miguel.getOutput('Walking','step_length'),subject2.getOutput('Walking','step_length'))
print("Step length Ctrl/Pre: %.6f\n"%a[1])
a=stats.ttest_ind(miguel.getOutput('Walking','step_length'),subject2_post.getOutput('Walking','step_length'))
print("Step length Ctrl/Post: %.6f\n"%a[1])
a=stats.ttest_ind(subject2.getOutput('Walking','step_length'),subject2_post.getOutput('Walking','step_length'))
print("Step length Pre/Post: %.6f\n"%a[1])

plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()

# %% Step width
plt.title("Step Width")
bplot = plt.boxplot([miguel.getOutput('Walking','step_width'),
    subject2.getOutput('Walking','step_width'),
    subject2_post.getOutput('Walking','step_width')],
    labels=["Control","S2 Pre","S2 Post"],patch_artist=True)

a=stats.ttest_ind(miguel.getOutput('Walking','step_width'),subject2.getOutput('Walking','step_width'))
print("Step width ctl/pre: %.6f\n"%a[1])
a=stats.ttest_ind(miguel.getOutput('Walking','step_width'),subject2_post.getOutput('Walking','step_width'))
print("Step width ctl/post: %.6f\n"%a[1])
a=stats.ttest_ind(subject2.getOutput('Walking','step_width'),subject2_post.getOutput('Walking','step_width'))
print("Step width pre/post: %.6f\n"%a[1])

plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.show()
# %% step height
plt.title("Step height")
bplot = plt.boxplot([miguel.getOutput('Walking','step_height'),
    subject2.getOutput('Walking','step_height'),
    subject2_post.getOutput('Walking','step_height')],
    labels=["Control","S2 Pre","S2 Post"],patch_artist=True)

a=stats.ttest_ind(miguel.getOutput('Walking','step_height'),subject2.getOutput('Walking','step_height'))
print("Step height ctl/pre: %.6f\n"%a[1])
a=stats.ttest_ind(miguel.getOutput('Walking','step_height'),subject2_post.getOutput('Walking','step_height'))
print("Step height ctl/post: %.6f\n"%a[1])
a=stats.ttest_ind(subject2.getOutput('Walking','step_height'),subject2_post.getOutput('Walking','step_height'))
print("Step height pre/post: %.6f\n"%a[1])

plt.ylabel("Normalized Distance (m/m)")
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1

plt.show()
# %% Double support
control_ds=miguel.getOutput('Walking','double_stance')
subject2_ds=subject2.getOutput('Walking','double_stance')
subject2_post_ds=subject2_post.getOutput('Walking','double_stance')
data=[control_ds,subject2_ds,subject2_post_ds]
bplot_ds = plt.boxplot(data,patch_artist=True)
plt.title("Double Stance")
plt.ylabel("Fraction of Gait Cycle")
a=stats.ttest_ind(control_ds,subject2_ds)
print("Double Stance ctl/pre p: %.5f"%a[1])
a=stats.ttest_ind(control_ds,subject2_post_ds)
print("Double Stance ctl/post p: %.5f"%a[1])
a=stats.ttest_ind(subject2_ds,subject2_post_ds)
print("Double Stance pre/post p: %.5f"%a[1])
# bplot_ds
j=0
for i in bplot_ds['boxes']:
    i.set_facecolor(labels[j])
    j+=1
# plt.show()
plt.show()
# %% tandem pelvis jerk
control_jerk=miguel.getOutput('Tandem','pelvis_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Tandem','pelvis_jerk_step_normalized')
subject2_post_jerk=subject2_post.getOutput('Tandem','pelvis_jerk_step_normalized')
# subject2_jerk=np.delete(subject2_jerk,np.argmax(subject2_jerk))  #removes outlie
# subject2_jerk=np.delete(subject2_jerk,np.argmax(subject2_jerk))  #removes outlie
data=[control_jerk,subject2_jerk,subject2_post_jerk]
bplot_jerk = plt.boxplot(data,patch_artist=True)
plt.title("Pelvis Jerk\nTandem")
plt.ylabel("Normalized Jerk")
a=stats.ttest_ind(control_jerk,subject2_jerk)
print("Pelvis Jerk ctl/pre: %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject2_post_jerk)
print("Pelvis Jerk ctl/post: %.5f"%a[1])
a=stats.ttest_ind(subject2_jerk,subject2_post_jerk)
print("Pelvis Jerk pre/post: %.5f"%a[1])
j=0
for i in bplot_jerk['boxes']:
    i.set_facecolor(labels[j])
    j+=1
import matplotlib.patches as mpatches
pop_a = mpatches.Patch(color=labels[0], label='Control')
pop_b = mpatches.Patch(color=labels[1], label='Subject 2 Pre')
pop_c = mpatches.Patch(color=labels[2], label='Subject 2 Post')
plt.legend(handles=[pop_a,pop_b,pop_c])
plt.show()

# %% tandem thorax jerk
control_jerk=miguel.getOutput('Tandem','thorax_jerk_step_normalized')
subject2_jerk=subject2.getOutput('Tandem','thorax_jerk_step_normalized')
subject2_post_jerk=subject2_post.getOutput('Tandem','thorax_jerk_step_normalized')
subject2_post_jerk=np.delete(subject2_post_jerk,np.argmax(subject2_post_jerk))  #removes outlie
data=[control_jerk,subject2_jerk,subject2_post_jerk]
bplot_jerk = plt.boxplot(data,labels=["Control","S2 Pre","S2 Post"],patch_artist=True)
plt.title("Thorax Jerk\nTandem")
plt.ylabel("Normalized Jerk")
j=0
for i in bplot_jerk['boxes']:
    i.set_facecolor(labels[j])
    j+=1
a=stats.ttest_ind(control_jerk,subject2_jerk)
print("Thorax ctl/pre Jerk: %.5f"%a[1])
a=stats.ttest_ind(control_jerk,subject2_post_jerk)
print("Thorax ctl/post Jerk: %.5f"%a[1])
a=stats.ttest_ind(subject2_jerk,subject2_post_jerk)
print("Thorax pre/post Jerk: %.5f"%a[1])
plt.show()

# %% 9 HOLE PEG
path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"  # this is actually not miguel :)
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=True,ng=False)
# CM subject 1 did not complete the right 9HP
# %% TMT
labels=['#130AF1','#17ACE8','#FE8821','#FEA832','#FE8821','#FEA832']
tmt = [miguel.output_data.data['9HPL']['TMT'],miguel.output_data.data['9HPR']['TMT'],subject2.output_data.data['9HPL']['TMT'],subject2.output_data.data['9HPR']['TMT'],subject2_post.output_data.data['9HPL']['TMT'],subject2_post.output_data.data['9HPR']['TMT']]
bplot = plt.boxplot(tmt,labels=["Control L","Control R","CM 2 Pre L","CM 2 Pre R","CM 2 Post L","CM 2 Post R"],patch_artist=True)
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
a=stats.ttest_ind(tmt[0],tmt[1])
print(f'Control R/L p: {a[1]}')
a=stats.ttest_ind(tmt[2],tmt[3])
print(f'CM 2 R/L p: {a[1]}')
a=stats.ttest_ind(tmt[4],tmt[5])
print(f'CM 3 R/L p: {a[1]}')

a=stats.ttest_ind(tmt[0],tmt[2])
print(f'L p: {a[1]}')
a=stats.ttest_ind(tmt[1],tmt[3])
print(f'R p: {a[1]}')

# %% Single peg MT
spt = [miguel.getOutput('9HPL','Single Peg MT'),miguel.getOutput('9HPR','Single Peg MT'),
    subject2.getOutput('9HPL','Single Peg MT'),subject2.getOutput('9HPR','Single Peg MT'),
    subject2_post.getOutput('9HPL','Single Peg MT'),subject2_post.getOutput('9HPR','Single Peg MT')]
spt[-1]=np.delete(spt[-1],np.argmax(spt[-1]))  #removes outlie
bplot = plt.boxplot(spt,labels=["Control L","Control R","CM 2 Pre L","CM 2 Pre R","CM 2 Post L","CM 2 Post R"],patch_artist=True)
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
a=stats.ttest_ind(spt[0],spt[1])
print(f'Control R/L p: {a[1]}')
a=stats.ttest_ind(spt[2],spt[3])
print(f'CM 2 R/L p: {a[1]}')
a=stats.ttest_ind(spt[4],spt[5])
print(f'CM 2 R/L p: {a[1]}')

# %% GR Ratio 
grr = [miguel.getOutput('9HPL','GR Ratio'),miguel.getOutput('9HPR','GR Ratio'), \
    subject2.getOutput('9HPL','GR Ratio'),subject2.getOutput('9HPR','GR Ratio'), \
    subject2_post.getOutput('9HPL','GR Ratio'),subject2_post.getOutput('9HPR','GR Ratio')]
grr[-1]=np.delete(grr[-1],np.argmax(grr[-1]))  #removes outlie
bplot = plt.boxplot(grr,labels=["Control L","Control R","CM 2 L","CM 2 R","CM 3 L","CM 3 R"],patch_artist=True)
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
a=stats.ttest_ind(grr[0],grr[1])
print(f'Control R/L p: {a[1]}')
a=stats.ttest_ind(grr[2],grr[3])
print(f'CM R/L p: {a[1]}')
a=stats.ttest_ind(grr[0],grr[2])
print(f'L compare p: {a[1]}')
a=stats.ttest_ind(grr[1],grr[2])
print(f'R compare p: {a[1]}')
# %%
ps = [miguel.getOutput('9HPL','Peak Speeds'),miguel.getOutput('9HPR','Peak Speeds'),
    subject2.getOutput('9HPL','Peak Speeds'),subject2.getOutput('9HPR','Peak Speeds'),
    subject2_post.getOutput('9HPL','Peak Speeds'),subject2_post.getOutput('9HPR','Peak Speeds')]
bplot = plt.boxplot(ps,labels=["Control L","Control R","CM 2 L","CM 2 R","CM 3 L","CM 3 R"],patch_artist=True)
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.ylabel("Speed")
plt.title("Peak Speeds")
plt.show()
# %%
ps_t = [miguel.getOutput('9HPL','Peak Speed Time'),miguel.getOutput('9HPR','Peak Speed Time'),
    subject2.getOutput('9HPL','Peak Speed Time'),subject2.getOutput('9HPR','Peak Speed Time'),
    subject2_post.getOutput('9HPL','Peak Speed Time'),subject2_post.getOutput('9HPR','Peak Speed Time')]
bplot=plt.boxplot(ps_t,labels=["Control L","Control R","CM 2 L","CM 2 R","CM 3 L","CM 3 R"],patch_artist=True)
j=0
for i in bplot['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.ylabel("Time (s)")
plt.title("Peak Speed Time")
print("Peak Speed: ")
print(f'C L: {np.mean(ps_t[0]):.2f} ({np.std(ps_t[0]):.2f})')
print(f'C R: {np.mean(ps_t[1]):.2f} ({np.std(ps_t[1]):.2f})')
print(f'S L: {np.mean(ps_t[2]):.2f} ({np.std(ps_t[2]):.2f})')
print(f'S R: {np.mean(ps_t[3]):.2f} ({np.std(ps_t[3]):.2f})')
plt.show()
a=stats.ttest_ind(ps_t[0],ps_t[1])
print(f'Control R/L p: {a[1]}')
a=stats.ttest_ind(ps_t[2],ps_t[3])
print(f'CM R/L p: {a[1]}')
a=stats.ttest_ind(ps_t[0],ps_t[2])
print(f'L p: {a[1]}')
a=stats.ttest_ind(ps_t[1],ps_t[3])
print(f'R p: {a[1]}')
# %%
jerkz = [miguel.getOutput('9HPL','Peak Jerk'),miguel.getOutput('9HPR','Peak Jerk'),
    subject2.getOutput('9HPL','Peak Jerk'),subject2.getOutput('9HPR','Peak Jerk'),
    subject2_post.getOutput('9HPL','Peak Jerk'),subject2_post.getOutput('9HPR','Peak Jerk')]
jerkz[0]=np.delete(jerkz[0],np.argmax(jerkz[0]))  #removes outlier
bplot = plt.boxplot(jerkz,labels=["Control L","Control R","CM 2 L","CM 2 R","CM 3 L","CM 3 R"],patch_artist=True)
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
a=stats.ttest_ind(jerkz[0],jerkz[1])
print(f'Control R/L p: {a[1]}')
a=stats.ttest_ind(jerkz[2],jerkz[3])
print(f'CM 2 R/L p: {a[1]}')
a=stats.ttest_ind(jerkz[4],jerkz[5])
print(f'CM 3 R/L p: {a[1]}')

a=stats.ttest_ind(jerkz[0],jerkz[2])
print(f'L compare p: {a[1]}')
a=stats.ttest_ind(jerkz[1],jerkz[3])
print(f'R compare p: {a[1]}')
# %%
from functions.helper import getIndexNames
namez = getIndexNames('9HP',subject2_post.marker_task_labels)
for trial in namez:
    data = subject2_post.marker_data[trial]
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

# %% NATURAL GRASPING
path_to_control =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
control = SessionDataObject(path_to_control,False,1.7)
labels=['#130AF1','#FE8821','#FE8821','#FE8821','#17ACE8','#FEA832','#FEA832','#FEA832']
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
sub3_pointer=subject2_post.getOutput('NG','Pointer Dist')
sub3_pointer=sub3_pointer[~np.isnan(sub3_pointer)]
sub3_thumb=subject2_post.getOutput('NG','Thumb Dist')
sub3_thumb=sub3_thumb[~np.isnan(sub3_thumb)]

data = [ctl_pointer,sub_pointer,sub2_pointer,sub3_pointer,ctl_thumb,sub_thumb,sub2_thumb,sub3_thumb]
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
bplot1 = plt.boxplot(data, labels=['C P','S1 P','S2 P','S3 P','C T','S1 T','S2 T','S3 T'],patch_artist=True,widths=0.5)
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
sub3_pointer=subject2_post.getOutput('NG','Pointer Jerk')
sub3_pointer=sub3_pointer[~np.isnan(sub3_pointer)]
sub3_pointer=np.delete(sub3_pointer,np.argmax(sub3_pointer))  #removes outlier
sub3_thumb=subject2_post.getOutput('NG','Thumb Jerk')
sub3_thumb=sub3_thumb[~np.isnan(sub3_thumb)]
ctl_pointer=control.getOutput('NG','Pointer Jerk')
ctl_pointer=ctl_pointer[~np.isnan(ctl_pointer)]
ctl_thumb=control.getOutput('NG','Thumb Jerk')
ctl_thumb=ctl_thumb[~np.isnan(ctl_thumb)]
data = [ctl_pointer,sub_pointer,sub2_pointer,sub3_pointer,ctl_thumb,sub_thumb,sub2_thumb,sub3_thumb]
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

bplot3 = plt.boxplot(data, labels=['C P','S1 P','S2 P','S3 P','C T','S1 T','S2 T','S3 T'],patch_artist=True,widths=0.5)
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
sub3_mga = subject2_post.getOutput('NG','MGA')
sub3_mga = sub3_mga[~np.isnan(sub3_mga)]
ctl_mga = control.getOutput('NG','MGA')
ctl_mga = ctl_mga[~np.isnan(ctl_mga)]
data=[ctl_mga,sub_mga,sub2_mga,sub3_mga]
a = stats.ttest_ind(data[0],data[1])
print("MGA p: %.5f"%a[1])
a = stats.ttest_ind(data[0],data[2])
print("MGA p: %.5f"%a[1])
bplot4 = plt.boxplot(data,labels=['Control','Subject 1', 'Subject 2', 'Subject 3'],patch_artist=True,widths=0.5)
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
sub3_mgat = subject2_post.getOutput('NG','MGA_t')
sub3_mgat = sub3_mgat[~np.isnan(sub3_mgat)]
sub3_mgat = sub3_mgat[~(sub3_mgat<0.55)]
ctl_mgat = control.getOutput('NG','MGA_t')
ctl_mgat = ctl_mgat[~np.isnan(ctl_mgat)]
ctl_mgat = ctl_mgat[~(ctl_mgat<0.55)]
data=[ctl_mgat,sub_mgat,sub2_mgat,sub3_mgat]
print("MGA t: ")
print(f'C: {np.mean(ctl_mgat):.2f} ({np.std(ctl_mgat):.2f})')
print(f'S1: {np.mean(sub_mgat):.2f} ({np.std(sub_mgat):.2f})')
print(f'S2: {np.mean(sub2_mgat):.2f} ({np.std(sub2_mgat):.2f})')
a = stats.ttest_ind(data[0],data[1])
print("MGA Time p: %.5f"%a[1])
a = stats.ttest_ind(data[0],data[2])
print("MGA Time p: %.5f"%a[1])
a = stats.ttest_ind(data[0],data[3])
print("MGA Time p: %.5f"%a[1])
bplot5 = plt.boxplot(data,labels=['Control','Subject 1','Subject 2','Subject 3'],patch_artist=True,widths=0.5)
j=0
for i in bplot5['boxes']:
    i.set_facecolor(labels[j])
    j+=1
plt.title("MGA Time\n% of trial")
plt.box(False)
plt.show()

# %%
