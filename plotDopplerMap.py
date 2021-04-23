#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This example plots the range Doppler map as colored image and shows positions
and types of the different labels within this image. This example should help
to get a better understanding about the dataset.
"""

"""
Dependencies: h5py, matplotlib
install with: pip3 install h5py matplotlib
"""

import h5py # Required to read the radar data.
import matplotlib.pyplot as plt # Default plotting module for Python.
import matplotlib.patches as patches

# Read the radar data into a variable.
data = h5py.File('data.h5', 'r')

classes = {
    '0':'pedestrian',
    '1':'car',
    '2':'truck',
    '3':'no object'
}

fig, ax = plt.subplots()
for idx,dmap in enumerate(data['rdms']):
    ax.clear()
    plt.imshow(dmap.T, origin='lower', interpolation='bilinear', cmap='prism')#, aspect=256/32*0.5)
    title = 'Sample ' + str(idx) + ": "
    for target in data['labels'][str(idx)]:
        rect = patches.Rectangle((target[1], target[0]), target[3]-target[1], target[2]-target[0], linewidth=1, edgecolor='b', facecolor='none')
        target_class = classes[str(int(target[4]))]
        ax.annotate(target_class, (target[3], target[2]), color='w', weight='bold', fontsize=6, ha='left', va='bottom')
        ax.add_patch(rect)
    plt.title(title)
    plt.draw()
    plt.show()
    plt.pause(20)
    plt.close()