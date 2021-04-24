import h5py
import matplotlib.pyplot as plt
import numpy as np
data = h5py.File('data.h5', 'r')

X = np.array(data["rdms"][:])

cornerCoordinates = np.array([]).reshape(0,5) #, dtype=np.int64
for idx in range(len(X)):
    # labelsAtIndex = np.array(data['labels'][str(idx)])
    for target in data['labels'][str(idx)]:
        cornerCoordinates = np.vstack([cornerCoordinates, target]) if target.size else cornerCoordinates
    



test = 1