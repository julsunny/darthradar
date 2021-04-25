from torch.utils.data import Dataset
import numpy as np
import pickle
import torch
from libs.transformations import ComposeDouble
from preprocessing import generate_dataset


class RadarDetectionDataSet(Dataset):
    """
    Builds a dataset with images and their respective targets.
    A target is expected to be a pickled file of a dict
    and should contain at least a 'boxes' and a 'labels' key.
    inputs and targets are expected to be ranges or lists of indices.
    Returns a dict with the following keys: 'x', 'x_name', 'y', 'y_name'
    """

    def __init__(self,
                 input_indices = None,
                 use_cache: bool = False,
                 convert_to_format: str = None,
                 mapping: bool = True,
                 transform: ComposeDouble = None,
                 ):
        if input_indices is None:
            input_indices = range(375)
        else:
            self.inputs = input_indices
        self.targets = input_indices

        # bring the dataset into the format accepted by the faster rCNN
        inp_images, targets = generate_dataset()
        self.input_images = inp_images
        self.target_dicts = targets

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
        '''
        load one doppler map (im) with corresponding labels (tar)
        :param inp:
        :param tar:
        :return:
        '''
        im = self.input_images[inp, :, :]
        im = im.T
        # normalize data
        im = (im-im.mean())/im.std()
        im = im.reshape(1, im.shape[0], im.shape[1])
        im = np.concatenate((im, im, im),axis=0)   # fake an RGB image since rCNN only works with RGB tensors
        try:
            tar = self.target_dicts[str(tar)]
        except:
            # if nothing is in the doppler map, the only label is the background (corresponds to label 0)
            tar = {"boxes": np.array([0, 0, im.shape[0], im.shape[1]]).reshape(1,4), "labels": np.array([0])}
        return im, tar

    def __len__(self):
        return len(self.inputs)

    def map_class_to_int(self, labels):
        '''Translate the dataset labels to labels compatible with rCNN'''
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

        # Create target
        target = {'boxes': boxes,
                  'labels': labels}

        # Preprocessing
        target = {key: value.numpy() for key, value in target.items()}  # all tensors should be converted to np.ndarrays

        if self.transform is not None:
            x, target = self.transform(x, target)  # returns np.ndarrays

        # Typecasting
        x = torch.from_numpy(x).type(torch.float32)
        target = {key: torch.from_numpy(value) for key, value in target.items()}

        return {'x': x, 'y': target, 'x_name': '', 'y_name': ''}


class RadarImageTargetSet():
    """
    Images are stored as numpy arrays.
    Target dictionaries have format: {
        'boxes'  : (N, 4) numpy array
        'labels' : (N,) numpy array
    }
        
    __getitem__ Returns a tuple consisting of an image and a target dictionary.
    
    Example:
        ds = RadarImageTargetSet()
        ds[17][0] # image: 2d numpy array
        ds[17][1] # target dictionary
    """

    def __init__(self):
        imgs, targets = generate_dataset()
        self.input_images = imgs
        assert isinstance(self.input_images, np.ndarray)
        self.target_dicts = targets

    def __getitem__(self, index: int):
        """Read image and target-dict given indexes"""
        image = self.input_images[index, :, :]
        try:
            target = self.target_dicts[str(index)]
        except:
            target = {"boxes": np.array([]), "labels": np.array([])}
        return image, target

    def __len__(self):
        return len(self.input_images)


def get_img_tgt_tuple_by_id(id: int):
    ds = RadarDetectionDataSet()
    return (ds[0]['x'], ds[0]['y'])
