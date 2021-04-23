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
from transformations import ComposeDouble, ComposeSingle, Clip, FunctionWrapperDouble, normalize, normalize_01
from dataloader import RadarDetectionDataSet


transforms = ComposeDouble([
    Clip(),
    # AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
    # AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
    # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
    FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
    FunctionWrapperDouble(normalize_01)   # or normalize
])

dataset = RadarDetectionDataSet(transform=transforms,
                                use_cache=False,
                                mapping=True)
