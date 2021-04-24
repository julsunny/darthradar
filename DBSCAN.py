import numpy as np
import sklearn.cluster as skc
import h5py
import torch
from sklearn import preprocessing

import matplotlib.patches as patches
from matplotlib import pyplot as plt
#%%

def height_map(rdms):
    fm_rdms = []
    for i in range(rdms.shape[0]):
        for j in range(rdms.shape[1]):
            fm_rdms.append([i, j, float(rdms[i,j])])
    
    return torch.Tensor(fm_rdms)

class Clustering:
    def __init__(self, clusters):
        self.clusters = np.asarray(clusters, dtype = Cluster)
    
    def get_centres(self):
        return np.asarray([c.mean for c in self.clusters])

class Cluster:
    def __init__(self, c_points, label):
        self.label = label
        self.c_points = c_points
        self.mean = np.mean(c_points, axis = 0)
        self.var = np.var(c_points, axis = 0)
        self.freq = c_points.shape[0]
        self.box = self.get_box()
        
    def average_var(self):
        return np.mean(self.var)
    
    def get_box(self):
        x0 = np.min(self.c_points[:,0])
        y0 = np.min(self.c_points[:,1])
        x1 = np.max(self.c_points[:,0])
        y1 = np.max(self.c_points[:,1])
        
        return [x0, y0, x1, y1]

#%%

data_path = "M:/programming/projects/TUM_Hackathon_Infineon_Radar_Challenge/darthradar/data.h5"

data = h5py.File(data_path, 'r')

classes = {
    '0' : 'pedestrian',
    '1' : 'car',
    '2' : 'truck',
    '3' : 'no object'
    }


filt = np.asarray([0,1,2,3,4,5,6,7,8])+200
ncols = 3
nrows = round(len(filt)/ncols)
fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols*3.5 + 2, nrows*1.5), dpi = 600)
ax_ref = axs.flatten()
    
for idx, rdms in enumerate(data['rdms'][filt]):
    
    rdms = torch.Tensor(rdms)
    rdms[126:131,:] = 30
    fm_rdms = height_map(rdms)
    
    fmt_rdms = fm_rdms.numpy()
    fmt_rdms2 = preprocessing.scale(fmt_rdms[:,2], axis = 0)
    fmt_rdms[:,2] = fmt_rdms2
    fmt_rdms[:,2] *= 1.5
    
    epsilon = 4
    #use dbscan to cluster data
    dbs = skc.DBSCAN(eps = epsilon, min_samples = 3, n_jobs = -1)
    dbs.fit(fmt_rdms)
    
    #evaluate clustering
    #create clustering object
    labels = dbs.labels_
    
    dbs_clustering = Clustering([Cluster(fm_rdms.numpy()[labels == l], l) for l in set(labels)])
    
    boxes = [cluster.box for cluster in dbs_clustering.clusters]

    #change cmap and v_range here
    ax_ref[idx].imshow(rdms.T, origin='lower', interpolation='bilinear', cmap='gist_ncar', vmin = 500, vmax = 6000) 
    
    for i, box in enumerate(boxes):
        rect = patches.Rectangle((box[0]-1, box[1]-1), box[2]-box[0]+2, box[3]-box[1]+2, linewidth=1, edgecolor='yellow', facecolor='none')
        ax_ref[idx].annotate(str(i), (box[3], box[2]), color='pink', weight='bold', fontsize=5, ha='right', va='bottom')
        ax_ref[idx].add_patch(rect)
    
    for target in data['labels'][str(filt[idx])]:
        rect = patches.Rectangle((target[1], target[0]), target[3]-target[1], target[2]-target[0], linewidth=0.5, edgecolor='red', facecolor='none')
        target_class = classes[str(int(target[4]))]
        ax_ref[idx].annotate(target_class, (target[3], target[2]), color='w', weight='bold', fontsize=5, ha='left', va='bottom')
        ax_ref[idx].add_patch(rect)
