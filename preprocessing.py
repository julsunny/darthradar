import h5py
import numpy as np
import torch
import pickle

def generate_dataset():
    data = h5py.File('data.h5', 'r')
    imgs = [data['rdms'][index][()] for index in range(len(data['rdms']))]
    imgs = np.array(imgs)
    imgs = np.concatenate((imgs[:, :127, :], imgs[:, 130:, :]), axis=1)   # cut out the artifacts
    targets = {}
    for index in range(len(data['rdms'])):
        for index2 in range(data['labels'][str(index)][()].shape[0]):  # iterate through objects
            targets[str(index)] = {}
            targets[str(index)]["boxes"] = data["labels"][str(index)][()][:, :4]
            targets[str(index)]["labels"] = data["labels"][str(index)][()][:, 4]
    np.save("doppler_data.npy", imgs)
    f = open("label_data.pkl", "wb")
    pickle.dump(targets, f)
    f.close()

    return

generate_dataset()