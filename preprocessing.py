import h5py
import numpy as np
import torch
import pickle

def generate_dataset():
    '''
    Receives data in hdf5 format, returns doppler maps as numpy array (number maps x width x height) and targets
    (dictionary with keys 0 to 375 (datapoint index)). Each targets[index] is a dictionary with keys "boxes" and "labels".
    :return:
    imgs (numpy array)
    targets (dict)
    '''
    data = h5py.File('data.h5', 'r')
    imgs = [data['rdms'][index][()] for index in range(len(data['rdms']))]
    imgs = np.array(imgs)
    imgs = np.concatenate((imgs[:, :127, :], imgs[:, 130:, :]), axis=1)   # cut out the artifacts in the middle
    targets = {}
    for index in range(len(data['rdms'])):
        for index2 in range(data['labels'][str(index)][()].shape[0]):  # iterate through objects
            targets[str(index)] = {}
            targets[str(index)]["boxes"] = data["labels"][str(index)][()][:, :4]
            targets[str(index)]["labels"] = data["labels"][str(index)][()][:, 4]

    return imgs, targets

