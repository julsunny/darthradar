import h5py # Required to read the radar data.
import matplotlib.pyplot as plt # Default plotting module for Python.
import matplotlib.patches as patches
import scipy.ndimage
import numpy as np

def detect_peaks(d, tx, ty, gx, gy, rate_fa):
    """
    :param d: radar image – numpy.ndarray, shape(N_doppler, N_range)
    :param tx: train cells x – float
    :param ty: train cells y – float
    :param gx: guard cells x _ float
    :param gy: guard cells y - float
    :param rate_fa: false alarm rate - float
    :return:
    numpeaks: number of peaks - int
    x: x coordinates of peaks – numpy.ndarray, shape(N_peaks,)
    y: y coordinates of peaks – numpy.ndarray, shape(N_peaks,)
    strength: strengths of peaks –numpy.ndarray, shape(N_peaks,)

    For example usage, see test_run
    """
    d = scipy.ndimage.gaussian_filter(d, sigma=(1,0.25))
    n_train = 4*tx*ty + 2*tx*(2*gy+1) + 2*ty*(2*gx+1)
    alpha = n_train * (rate_fa**(-1/n_train) - 1) #threshold factor

    f = np.ones((2*(gx + tx)+1, 2*(gy + ty)+1))
    cond = d == scipy.ndimage.maximum_filter(
    d, footprint=f, mode='constant', cval=-np.inf)

    f[tx:tx+2*gx+1, ty:ty+2*gy+1] = 0
    f /= n_train
    diff = d - alpha * scipy.ndimage.convolve(d, f, mode='constant', cval = np.inf)

    x, y = np.argwhere(cond * (diff > 0)).T
    strength = diff[x,y]
    numpeaks = x.size

    return numpeaks, x, y, strength

def cutout_middle_strip(d, cutout_lower, cutout_upper):
    # Cuts out the middle strip corresponding to static objects
    d = np.delete(d, np.arange(cutout_lower,cutout_upper+1), axis=0)
    return d

def return_box_bounds(x, y, halfwidth, halfheight, limits):
    x0 = np.maximum(x - halfwidth, np.repeat(0, x.shape[0]))
    x1 = np.minimum(x + halfwidth, np.repeat(limits[0], x.shape[0]))
    y0 = np.maximum(y - halfheight,np.repeat(0, y.shape[0]))
    y1 = np.minimum(y + halfheight, np.repeat(limits[1], y.shape[0]))

    bounds = np.stack((x0, y0, x1, y1), axis = 1)

    return bounds

def test_run(delay):
    # Read the radar data into a variable.
    data = h5py.File('data.h5', 'r')

    classes = {
        '0': 'pedestrian',
        '1': 'car',
        '2': 'truck',
        '3': 'no object'
    }

    fig, ax = plt.subplots()

    for idx,dmap in enumerate(data['rdms']):
        ax.clear()
        dmap = cutout_middle_strip(dmap, 122, 134)
        numpeaks, x, y, strength = detect_peaks(dmap, tx=10, ty=3, gx=2, gy=1, rate_fa=1e-2)

        plt.imshow(dmap.T, origin='lower', interpolation='bilinear', cmap='viridis')
        plt.scatter(x,y,color='red')
        title = '# Peaks: ' + str(numpeaks)

        for target in data['labels'][str(idx)]:
            rect = patches.Rectangle((target[1], target[0]), target[2]-target[0], target[3]-target[1], linewidth=1, edgecolor='b', facecolor='none')
            target_class = classes[str(int(target[4]))]
            ax.annotate(target_class, (target[3], target[2]), color='w', weight='bold', fontsize=6, ha='left', va='bottom')
            ax.add_patch(rect)

        bounds = return_box_bounds(x,y,11,7,dmap.shape)
        print(bounds)
        for i in bounds:
            rect = patches.Rectangle((i[0], i[1]), i[2]-i[0]+1, i[3]-i[1]+1, linewidth=1,
                                     edgecolor='y', facecolor='none')
            ax.add_patch(rect)

        plt.title(title)
        plt.draw()
        plt.pause(delay)

test_run(0.5)