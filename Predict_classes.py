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
from config import CLASSES
from torch.autograd import Variable
from model.finetune_model import NNetwork

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import utils
import argparse

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

from torchvision import models

from catalyst.contrib.data.reader import ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl import SupervisedRunner, SchedulerCallback, TensorboardLogger
from catalyst.callbacks.checkpoint import CheckpointCallback
from catalyst.callbacks.misc import EarlyStoppingCallback
from catalyst.callbacks import EarlyStoppingCallback, CheckpointCallback


parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('-dir', type=str, default=None,
                    help='Pass a directory to evaluate the images in it', required=True)
parser.add_argument('-weights_dir', type=str, default=None,
                    help='Pass a weights directory', required=True)

args = parser.parse_args()

print("Loading images from directory : ", args.dir)

PATH = args.dir
NUM_WORKERS = 0

test_data = torchvision.datasets.ImageFolder(root=os.path.join(PATH, 'test'), transform = utils.get_augmentation('test'))
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model = models.resnet50(pretrained=False)
model = NNetwork(base_model)
model = model.to(device)

state_dict = torch.load(os.path.join(args.weights_dir, 'best-last-resnet50.pth'), map_location=torch.device('cpu'))
model.load_state_dict(state_dict["model_state_dict"])

model.eval()

print('************The results are:**********')

with torch.no_grad():
    for i, data in enumerate(test_loader):
        images, labels = data
        images = images
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        filename, _ = test_loader.dataset.samples[i]
        print(filename.split('/')[-1], CLASSES[predicted.cpu().detach().numpy()[0]])
