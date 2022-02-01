# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

files = os.listdir()


small_layout = pd.read_csv(files[2],sep='\t',skiprows=11)
small_means = np.mean(small_layout,axis=0)
small_std = np.std(small_layout,axis=0)
big_layout = pd.read_csv(files[1],sep='\t',skiprows=11)
big_means = np.mean(big_layout,axis=0)
big_std = np.std(big_layout,axis=0)

layout = small_means
x = layout[range(2,28,3)]
y = layout[range(3,29 ,3)]
z = layout[range(4,30,3)]

plt.scatter(x,y,s=200,alpha=0.5,edgecolors="black")

#paths
path = pd.read_csv(files[3],sep='\t',skiprows=11)
tpts = range(110)
plt.plot(path['Wrist X'][tpts],path['Wrist Y'][tpts],c='grey')
plt.plot(path['Pointer X'][tpts],path['Pointer Y'][tpts],c='blue')
plt.plot(path['Thumb X'][tpts],path['Thumb Y'][tpts],c='red')
plt.legend(['Peg','Wrist','Pointer','Thumb'])
plt.show() 

tpts = range(200,380)
plt.scatter(x,y,s=200,alpha=0.5,edgecolors="black")
plt.plot(path['Wrist X'][tpts],path['Wrist Y'][tpts],c='grey')
plt.plot(path['Pointer X'][tpts],path['Pointer Y'][tpts],c='blue')
plt.plot(path['Thumb X'][tpts],path['Thumb Y'][tpts],c='red')
plt.legend(['Peg','Wrist','Pointer','Thumb'])

# %%
