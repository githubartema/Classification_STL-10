import os
import cv2
import collections
import time 

import tqdm
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import Dataset
import argparse


from torchvision import datasets, models, transforms

from catalyst.data import Augmentor
from catalyst.contrib.data.cv.reader import ImageReader
from catalyst.contrib.data.reader import ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl import SupervisedRunner, SchedulerCallback, TensorboardLogger
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback
from catalyst.callbacks.metrics.classification import PrecisionRecallF1SupportCallback
from catalyst.callbacks.misc import EarlyStoppingCallback
from catalyst.callbacks import EarlyStoppingCallback, CheckpointCallback

from torchvision import models
from model.finetune_model import NNetwork

from utils import utils

parser = argparse.ArgumentParser(description='Training stage')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to the train images', required=True)
parser.add_argument('-batch_size', type=int, default=32,
                    help='Batch size for training', required=False)
parser.add_argument('-num_workers', type=int, default=0,
                    help='Number of workers for training', required=False)


args = parser.parse_args()

BATCH_SIZE = args.batch_size
PATH = args.dir 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.resnet50(pretrained=True)
model = NNetwork(base_model)
model = model.to(device)

print('The model structure is:')

for _,f in enumerate(list(model.backbone_features)):
  print(_, f)


NUM_WORKERS = args.num_workers

model.freeze_backbone()
model.unfreeze_layers_from(6)

data = torchvision.datasets.ImageFolder(root=os.path.join(PATH, 'train'))

train_dataset, valid_dataset = torch.utils.data.random_split(data, [9500, 500])
train_dataset, valid_dataset = utils.CustomTransform(train_dataset, utils.get_augmentation('train')), utils.CustomTransform(valid_dataset, utils.get_augmentation('valid'))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

num_epochs = 30

conv_lr = 1e-2
dense_lr = 1e-3

optimizer = torch.optim.SGD([
    {'params': model.backbone_features.parameters(), 'lr': conv_lr},
    {'params': model.classifier.parameters(), 'lr': dense_lr}], 
    momentum=0.9
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.15, patience=2
)

criterion = nn.CrossEntropyLoss()

runner = SupervisedRunner(
    input_key='features', 
    output_key='scores', 
    target_key='targets',
    loss_key='loss'
)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    callbacks=[
        SchedulerCallback(loader_key='valid', metric_key='loss'),
        EarlyStoppingCallback(loader_key='valid', metric_key='loss', 
                              minimize=True, patience=5, min_delta=0.001),
        CheckpointCallback(
            logdir='/content/sample_data/checkpoint', 
            loader_key='valid', metric_key="loss", save_n_best=2, 
                   minimize=True),
        AccuracyCallback(input_key="scores", target_key="targets", num_classes=10),
        PrecisionRecallF1SupportCallback(
                   input_key="scores", target_key="targets", num_classes=10),
        ConfusionMatrixCallback(
                   input_key="scores", target_key="targets", num_classes=10
        )],
    loaders=loaders,
    num_epochs=num_epochs,
    verbose=True,
    loggers={'tensorboard': TensorboardLogger(logdir='./logdir/tensorboard')},
    logdir='./logdir/tensorboard'
)

print('Weights have been saved to checkpoint library.')
