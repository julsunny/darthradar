from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import h5py
import numpy as np
import pickle
import torch
from torchvision.ops import box_convert
from typing import List, Dict
from transformations import ComposeDouble, ComposeSingle


class RadarDetectionDataSet(Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a pickled file of a dict
    and should contain at least a 'boxes' and a 'labels' key.
    inputs and targets are expected to be a list of pathlib.Path objects.
    In case your labels are strings, you can use mapping (a dict) to int-encode them.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(self,
                 input_start,
                 input_stop,
                 use_cache: bool = False,
                 convert_to_format: str = None,
                 mapping: bool = True,
                 transform: ComposeDouble = None,
                 ):
        self.inputs = range(input_start,input_stop)
        self.targets = range(input_start,input_stop)
        self.input_images = np.load("doppler_data.npy")
        with open("label_data.pkl","rb") as f:
            self.target_dicts = pickle.load(f)

        self.use_cache = use_cache
        self.convert_to_format = convert_to_format
        self.mapping = mapping
        self.transform = transform

        if self.use_cache:
            # Use multiprocessing to load images and targets into RAM
            from multiprocessing import Pool
            with Pool() as pool:
                self.cached_data = pool.starmap(self.read_doppler, zip(self.inputs, self.targets))

    def read_doppler(self, inp, tar):
        im = self.input_images[inp, :, :]
        try:
            tar = self.target_dicts[str(tar)]
        except:
            tar = {"boxes": np.array([]), "labels": np.array([])}
        return im, tar

    def __len__(self):
        return len(self.inputs)

    def map_class_to_int(self,labels):
        translation = {0: 1, 1: 2, 2: 3, 3: 4}
        new_list = [translation[l] for l in labels]
        return np.array(new_list)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = self.read_doppler(input_ID, target_ID)

        # Read boxes
        try:
            boxes = torch.from_numpy(y['boxes']).to(torch.float32)
        except TypeError:
            boxes = torch.tensor(y['boxes']).to(torch.float32)

        # Label Mapping
        if self.mapping:
            labels = self.map_class_to_int(y['labels'])
        else:
            labels = y['labels']

        # Read labels
        try:
            labels = torch.from_numpy(labels).to(torch.int64)
        except TypeError:
            labels = torch.tensor(labels).to(torch.int64)

        # Convert format
        #if self.convert_to_format == 'xyxy':
        #    boxes = box_convert(boxes, in_fmt='xywh', out_fmt='xyxy')  # transforms boxes from xywh to xyxy format
        #elif self.convert_to_format == 'xywh':
        #    boxes = box_convert(boxes, in_fmt='xyxy', out_fmt='xywh')  # transforms boxes from xyxy to xywh format

        # Create target
        target = {'boxes': boxes,
                  'labels': labels}

        # Preprocessing
        target = {key: value.numpy() for key, value in target.items()}  # all tensors should be converted to np.ndarrays

        #if self.transform is not None:
        #    x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {key: torch.from_numpy(value) for key, value in target.items()}

        return {'x': x, 'y': target, 'x_name': '', 'y_name': ''}

