import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,1000,1000)
x = 100*np.sin(t)
y = (t-500)**2
z = t

# Method 1: Individual Gradients
dx = np.gradient(x,1)
dy = np.gradient(y,1)
dz = np.gradient(z,1)

print(f'Method 1: {dx[5]+dy[5]+dz[5]}')

# Method 2: Add together
dists = []
for i in range(len(x)-1):
    dists.append((x[i+1]-x[i])+(y[i+1]-y[i])+z[i+1]-z[i])
print(f'Method 2: {dists[5]}')

plt.plot(dx+dy+dz)
plt.plot(dists)
plt.show()
