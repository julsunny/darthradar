import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
data = h5py.File('data.h5', 'r')

X = np.array(data["rdms"][:])

# Get coordinates of all corner points of targets
cornerCoordinates = np.array([]).reshape(0,5)
for idx in range(len(X)):
    for target in data['labels'][str(idx)]:
        cornerCoordinates = np.vstack([cornerCoordinates, target]) if target.size else cornerCoordinates

# Reshape coordinates
xmin = cornerCoordinates[:,0:1]
ymin = cornerCoordinates[:,1:2]
xmax = cornerCoordinates[:,2:3]
ymax = cornerCoordinates[:,3:4]
labels = cornerCoordinates[:,4:5]

p1 = np.concatenate((xmin, ymin, labels),  axis = 1) 
p2 = np.concatenate((xmin, ymax, labels), axis = 1) 
p3 = np.concatenate((xmax, ymin, labels), axis = 1) 
p4 = np.concatenate((xmax, ymax, labels), axis = 1) 

data = np.vstack((p1, p2, p3, p4))

x = data[:,0]
y = data[:,1]
labels = data[:,2]

# Scatter Plot
colors = ['red','green','blue','purple']

fig = plt.figure(figsize=(8,8))
plt.scatter(x, y, c=labels, cmap=matplotlib.colors.ListedColormap(colors)) # alpha=0.3, for transparency


cb = plt.colorbar()
loc = np.arange(0,max(labels),max(labels)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(['red: pedestrians','green: cars','blue: trucks','purple: no object'])

plt.show()