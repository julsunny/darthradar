import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py

#%%
def plot_rdms(rdms, target, ax, cmap = "gist_ncar", v_range = (500, 6000)):
    """
    

    Parameters
    ----------
    rdms : 2d numpy array or 2d torch tensor
        the radar doppler image
    target : dictionary
        target['boxes'] is a numpy (N, 4) array or a torch.FloatTensor[N, 4] where N is the number of boxes
    ax : TYPE
        DESCRIPTION.
    cmap : TYPE, optional
        DESCRIPTION. The default is "gist_ncar".
    v_range : TYPE, optional
        DESCRIPTION. The default is (500, 6000).

    Returns
    -------
    None.

    """
    
    classes = {
    '1' : 'pedestrian',
    '2' : 'car',
    '3' : 'truck',
    '4' : 'no object'
    }
    print(target["refined_boxes"])
    ax.imshow(rdms.T[()], origin = 'lower', interpolation = 'bilinear', cmap = cmap, vmin = v_range[0], vmax = v_range[1])
    #plot predicted boxes
    for idx, box in enumerate(target["refined_boxes"]):
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth = 2, edgecolor = 'red', facecolor = 'none')
        ax.add_patch(rect)
        #box_class = classes[str(int(target["labels"][idx]))]
        #ax.annotate(box_class, (box[3], box[2]), color = 'w', weight = 'bold', fontsize = 5, ha = 'left', va = 'bottom')
    
    #draw prelabeled boxes
    for idx, box in enumerate(target["boxes"]):
        rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth = 1, edgecolor = 'pink', facecolor = 'none')
        box_class = classes[str(int(target["labels"][idx]))]
        ax.annotate(box_class, (box[3], box[2]), color = 'w', weight = 'bold', fontsize = 5, ha = 'left', va = 'bottom')
        ax.add_patch(rect)
   
def grid_plot(samples, ncols = 3, cmap = "gist_ncar", v_range = (500, 6000)):
    """
    High level function for plotting samples on a grid with ncols. 

    Parameters
    ----------
    samples : (image, target) pair
        DESCRIPTION.
    ncols : TYPE, optional
        DESCRIPTION. The default is 3.
    cmap : TYPE, optional
        DESCRIPTION. The default is "gist_ncar".
    v_range : TYPE, optional
        DESCRIPTION. The default is (500, 6000).

    Returns
    -------
    None.

    """
    
    nrows = round(len(samples)/ncols)
    fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols*3.5, nrows*1.5), dpi = 400)
    ax_ref = axs.flatten()
    for i, sample in enumerate(samples): #sample = (rdms, target)
        plot_rdms(sample[0], sample[1], ax_ref[i], cmap, v_range)
        
#%%       
"""
data_path = "./data.h5"

data = h5py.File(data_path, 'r')

classes = {
    '0' : 'pedestrian',
    '1' : 'car',
    '2' : 'truck',
    '3' : 'no object'
    }

#%%
#this works for old format
filt = [0,1,2,3,4,5]
ncols = 3
nrows = round(len(filt)/ncols + 0.5)

fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (ncols*3.5 + 2, nrows*1.5), dpi = 600)
ax_ref = axs.flatten()

for idx,dmap in enumerate(data['rdms'][filt]):
    #change cmap and v_range here
    ax_ref[idx].imshow(dmap.T, origin='lower', interpolation='bilinear', cmap='gist_ncar', vmin = 500, vmax = 6000) 
    
    for target in data['labels'][str(idx)]:
        rect = patches.Rectangle((target[1], target[0]), target[3]-target[1], target[2]-target[0], linewidth=1, edgecolor='pink', facecolor='none')
        target_class = classes[str(int(target[4]))]
        ax_ref[idx].annotate(target_class, (target[3], target[2]), color='w', weight='bold', fontsize=5, ha='left', va='bottom')
        ax_ref[idx].add_patch(rect)
"""