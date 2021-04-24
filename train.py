from torch.utils.data import Dataset
import pandas as pd
import os
import torch
import h5py
import albumentations as A
import numpy as np

from dataloader import RadarDetectionDataSet
from torch.utils.data import DataLoader

from transformations import ComposeDouble, Clip, AlbumentationWrapper, FunctionWrapperDouble
from transformations import normalize_01, normalize
from utils import collate_double

if __name__ ==  '__main__':
    params = {'BATCH_SIZE': 1,
              'LR': 0.001,
              'PRECISION': 32,
              'CLASSES': 5,
              'SEED': 42,
              'PROJECT': 'Heads',
              'EXPERIMENT': 'heads',
              'MAXEPOCHS': 10,
              'BACKBONE': 'resnet18',
              'FPN': False,
              'ANCHOR_SIZE': ((2, 4, 8, 16, 32),),
              'ASPECT_RATIOS': ((0.5, 1.0, 2.0),),
              'MIN_SIZE': 1024,
              'MAX_SIZE': 1024,
              'IMG_MEAN': [0., 0., 0.],
              'IMG_STD': [1., 1., 1.],
              'IOU_THRESHOLD': 0.5,
              'CHECKPOINT': 'results_resnet'
              }

    # training transformations and augmentations
    transforms_training = ComposeDouble([
        Clip(),
        AlbumentationWrapper(albumentation=A.HorizontalFlip(p=0.5)),
        AlbumentationWrapper(albumentation=A.RandomScale(p=0.5, scale_limit=0.5)),
        # AlbuWrapper(albu=A.VerticalFlip(p=0.5)),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    # validation transformations
    transforms_validation = ComposeDouble([
        Clip(),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    # test transformations
    transforms_test = ComposeDouble([
        Clip(),
        FunctionWrapperDouble(np.moveaxis, source=-1, destination=0),
        FunctionWrapperDouble(normalize_01)
    ])

    from pytorch_lightning import seed_everything
    seed_everything(params['SEED'])

    input_start_train = 0
    input_stop_train = 300
    input_start_dev = 300
    input_stop_dev = 330
    input_start_test = 330
    input_stop_test = 375

    # dataset training
    dataset_train = RadarDetectionDataSet(input_start=input_start_train,
                                           input_stop=input_stop_train,
                                           transform=transforms_training,
                                           use_cache=True,
                                           mapping=True)

    # dataset validation
    dataset_valid = RadarDetectionDataSet(input_start=input_start_dev,
                                           input_stop=input_stop_dev,
                                           transform=transforms_validation,
                                           use_cache=True,
                                           mapping=True)

    # dataset test
    dataset_test = RadarDetectionDataSet(input_start=input_start_test,
                                           input_stop=input_stop_test,
                                           transform=transforms_test,
                                           use_cache=True,
                                           mapping=True)

    # dataloader training
    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=params['BATCH_SIZE'],
                                  shuffle=True,
                                  num_workers=8,
                                  collate_fn=collate_double)

    # dataloader validation
    dataloader_valid = DataLoader(dataset=dataset_valid,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=8,
                                  collate_fn=collate_double)

    # dataloader test
    dataloader_test = DataLoader(dataset=dataset_test,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=8,
                                 collate_fn=collate_double)

    print("-----------------done creating datasets!-----------------")


    from faster_RCNN import get_fasterRCNN_resnet

    model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                                  backbone_name=params['BACKBONE'],
                                  anchor_size=params['ANCHOR_SIZE'],
                                  aspect_ratios=params['ASPECT_RATIOS'],
                                  fpn=params['FPN'],
                                  min_size=params['MIN_SIZE'],
                                  max_size=params['MAX_SIZE'])

    from faster_RCNN import FasterRCNN_lightning

    task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])

    # callbacks
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

    checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
    learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
    early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=50, mode='max')

    # trainer init
    from pytorch_lightning import Trainer

    trainer = Trainer(gpus=0,
                      precision=params['PRECISION'],  # try 16 with enable_pl_optimizer=False
                      default_root_dir=params["CHECKPOINT"],  # where checkpoints are saved to
                      log_every_n_steps=1,
                      num_sanity_val_steps=0,
                      enable_pl_optimizer=False,  # False seems to be necessary for half precision
                      )

    trainer.max_epochs = params['MAXEPOCHS']
    print("---------------------Start training!---------------------")
    trainer.fit(task,
                train_dataloader=dataloader_train,
                val_dataloaders=dataloader_valid)


    trainer.test(ckpt_path='best', test_dataloaders=dataloader_test)

