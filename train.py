from dataloader import RadarDetectionDataSet
from torch.utils.data import DataLoader
from libs.utils import collate_double
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from random import shuffle


'''
Script for training and testing the faster rCNN for radar target detection and classification.
For rCNN, see https://arxiv.org/abs/1506.01497
For the implementation which we use here, see https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial 
'''

if __name__ ==  '__main__':

    # set parameters for the rCNN
    params = {'BATCH_SIZE': 2,
              'LR': 0.0001,
              'PRECISION': 32,
              'CLASSES': 5,
              'SEED': 42,
              'PROJECT': 'Radar',
              'EXPERIMENT': 'radar',
              'MAXEPOCHS': 100,
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

    # set random seed
    seed_everything(params['SEED'])

    # define data indices for train, test and val data
    # shuffle the indices before splitting into train, test, dev
    shuffled_indices = list(range(0, 375))
    shuffle(shuffled_indices)
    # split the data into train, test and validation
    input_indices_train = shuffled_indices[:300]
    input_indices_test = shuffled_indices[300:340]
    input_indices_dev = shuffled_indices[340:]

    # dataset training
    dataset_train = RadarDetectionDataSet(input_indices=input_indices_train,
                                           use_cache=True,
                                           mapping=True)

    # dataset validation
    dataset_valid = RadarDetectionDataSet(input_indices=input_indices_dev,
                                           use_cache=True,
                                           mapping=True)

    # dataset test
    dataset_test = RadarDetectionDataSet(input_indices=input_indices_test,
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


    from libs.faster_RCNN import get_fasterRCNN_resnet

    model = get_fasterRCNN_resnet(num_classes=params['CLASSES'],
                                  backbone_name=params['BACKBONE'],
                                  anchor_size=params['ANCHOR_SIZE'],
                                  aspect_ratios=params['ASPECT_RATIOS'],
                                  fpn=params['FPN'],
                                  min_size=params['MIN_SIZE'],
                                  max_size=params['MAX_SIZE'])

    from libs.faster_RCNN import FasterRCNN_lightning

    task = FasterRCNN_lightning(model=model, lr=params['LR'], iou_threshold=params['IOU_THRESHOLD'])

    # callbacks
    checkpoint_callback = ModelCheckpoint(monitor='Validation_mAP', mode='max')
    learningrate_callback = LearningRateMonitor(logging_interval='step', log_momentum=False)
    early_stopping_callback = EarlyStopping(monitor='Validation_mAP', patience=50, mode='max')

    # trainer init

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

