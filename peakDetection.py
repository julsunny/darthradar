import h5py # Required to read the radar data.
import matplotlib.pyplot as plt # Default plotting module for Python.
import matplotlib.patches as patches
import scipy.ndimage
import numpy as np
from skimage import measure
STANDARD_BOXSIZE_X = 11
STANDARD_BOXSIZE_Y = 7
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
    """

    # This implements cell-averaging CFAR in 2 dimensions, using boolean filters
    d = scipy.ndimage.gaussian_filter(d, sigma=(5,1))
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

def return_box_bounds(x, y, data, halfwidth, halfheight):
    """
    :param x: np.ndarray, shape(N_peaks,)
    :param y: np.ndarray, shape(N_peaks,)
    :param data: np.ndarray, shape(N_doppler, N_range)
    :param halfwidth: float
    :param halfheight: float
    :return:
    bounds: np.ndarray, shape(N_peaks,4)
    """

    # Return box bounds that are adapted to the image statistics

    x0 = np.maximum(x - halfwidth, np.repeat(0, x.shape[0]))
    x1 = np.minimum(x + halfwidth, np.repeat(data.shape[0]-1, x.shape[0]))
    y0 = np.maximum(y - halfheight,np.repeat(0, y.shape[0]))
    y1 = np.minimum(y + halfheight, np.repeat(data.shape[1]-1, y.shape[0]))

    bounds = np.stack((x0, y0, x1, y1), axis = 1)

    for b in bounds:
        image = data[b[0]:b[2], b[1]:b[3]]
        M =  measure.moments(image)
        centroid = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
        MC = measure.moments_central(image, centroid)
        var = (MC[2,0] / M[0, 0], MC[0,2] / M[0, 0])
        b[0] = max(b[0], b[0] + centroid[0] - 0.25*var[0])
        b[1] = max(b[1], b[1] + centroid[1] - 0.25*var[1])
        b[2] = min(b[2], b[0] + centroid[0] + 0.25*var[0])
        b[3] = min(b[3], b[1] + centroid[1] + 0.25*var[1])

    return bounds

def return_static_box_bounds(x, y, data, halfwidth, halfheight):

    # Return uniformly sized box bounds

    x0 = np.maximum(x - halfwidth, np.repeat(0, x.shape[0]))
    x1 = np.minimum(x + halfwidth, np.repeat(data.shape[0] - 1, x.shape[0]))
    y0 = np.maximum(y - halfheight, np.repeat(0, y.shape[0]))
    y1 = np.minimum(y + halfheight, np.repeat(data.shape[1] - 1, y.shape[0]))

    bounds = np.stack((x0, y0, x1, y1), axis=1)

    return bounds

def return_box_stats(data, bounds):
    # Return spacial moments, pixel value statistics and box position
    box_stats = []
    for b in bounds:
        image = data[b[0]:b[2], b[1]:b[3]]
        norm = measure.moments(image)[0,0]
        if np.isclose(norm, 0):
            stats = measure.moments(image, order = 3).flatten()
        else:
            stats = (measure.moments_central(image, order = 3)/norm).flatten()
        stats = np.append(stats, 0.5*(b[0]+b[2]))
        stats = np.append(stats, 0.5*(b[1]+b[3]))
        stats = np.append(stats, np.mean(image))
        stats = np.append(stats, np.var(image))
        box_stats.append(stats)

    return box_stats

def bounds_overlap(b1, b2):
    # Standard box overlap criterion
    width = 2*STANDARD_BOXSIZE_X+1
    height = 2*STANDARD_BOXSIZE_Y+1
    overlap = max((width - 2*abs(b1[0]-b2[0])) * (height - 2*abs(b1[1]-b2[1])), 0)
    return ( overlap / (2*width*height - overlap) > 0.5)