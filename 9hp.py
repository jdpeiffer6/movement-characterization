# %%
from functions.SessionDataObject import SessionDataObject
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums,ttest_ind

path_to_subject =  "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\004\\2022-04-12"
subject = SessionDataObject(path_to_subject,False,1.7,walking=True,ng=False)
path_to_miguel = "C:\\Users\\jd\\Box\\Movement-Characterization\\data\\output\\001\\2022-02-22"
miguel = SessionDataObject(path_to_miguel,False,1.75,walking=True,ng=False)
# %% TMT
tmt = [miguel.output_data.data['9HPL']['TMT'],miguel.output_data.data['9HPR']['TMT'],subject.output_data.data['9HPL']['TMT'],subject.output_data.data['9HPR']['TMT']]
plt.boxplot(tmt,labels=["Control L","Control R","CM L","CM R"])
plt.ylabel("Time (s)")
plt.title("Total Movement Time")
plt.show()
a=ttest_ind(tmt[0],tmt[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(tmt[2],tmt[3])
print(f'CM R/L p: {a[1]}')

# %% Single peg MT
spt = [miguel.getOutput('9HPL','Single Peg MT'),miguel.getOutput('9HPR','Single Peg MT'),subject.getOutput('9HPL','Single Peg MT'),subject.getOutput('9HPR','Single Peg MT')]
plt.boxplot(spt,labels=["Control L","Control R","CM L","CM R"])
plt.ylabel("Time (s)")
plt.title("Single Peg Movement Time")
plt.show()
a=ttest_ind(spt[0],spt[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(spt[2],spt[3])
print(f'CM R/L p: {a[1]}')

# %% GR Ratio 
grr = [miguel.getOutput('9HPL','GR Ratio'),miguel.getOutput('9HPR','GR Ratio'),subject.getOutput('9HPL','GR Ratio'),subject.getOutput('9HPR','GR Ratio')]
plt.boxplot(grr,labels=["Control L","Control R","CM L","CM R"])
plt.ylabel("Ratio s/s")
plt.title("Grasp-Reach Ratio")
plt.show()
a=ttest_ind(spt[0],spt[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(spt[2],spt[3])
print(f'CM R/L p: {a[1]}')
a=ttest_ind(spt[0],spt[2])
print(f'L compare p: {a[1]}')
a=ttest_ind(spt[1],spt[2])
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
plt.show()
a=ttest_ind(ps_t[0],ps_t[1])
print(f'Control R/L p: {a[1]}')
a=ttest_ind(ps_t[2],ps_t[3])
print(f'CM R/L p: {a[1]}')
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
    plt.show()
# %%
