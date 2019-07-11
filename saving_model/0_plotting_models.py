#! /usr/bin/python

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


from torchvision import transforms, datasets
import numpy as np

from matplotlib import pyplot as plt

import time
import sys
sys.path.append('../python_scripts/')
from utils import save_curves_no_smooth as save_curves

from sys import argv
# the path of the log is passed after the command
exp = argv[1]
ep = argv[2]

# define a CNN
sys.path.append('../python_scripts/')
from networks import alexnet_pytorch as Net

epoch = '_epoch_' + ep
model_path = '../saving_model/exp' + exp + '/exp' + exp + epoch + '.pth.tar'

checkpoint = torch.load(model_path)
print 'Model loaded from the following path:'
print model_path
print 'Model keys:'
print checkpoint.keys()

save_curves(checkpoint, './')
