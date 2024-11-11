from __future__ import print_function
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as tr
import torchvision.utils as vutils

from data import GetMyData2DForTest
from losses import DICELossMultiClass
from models import UNetSmall
from tqdm import tqdm
import numpy as np

# DATA_FOLDER = "dataset/Train/Image/4"
DATA_FOLDER = "ForTest/0"
batch_size = 4
size = 1024
lr = 1e-3
beta1 = 0.5
beta2 = 0.99
log_interval = 20
savename = "Unet"
epochs = 200


test_images = GetMyData2DForTest(DATA_FOLDER,transform=tr.Compose([
        tr.Resize([size, size]),
        tr.ToTensor()
        ]))

test_loader = DataLoader(test_images,batch_size=batch_size,
                          shuffle=False, num_workers=1)


model = UNetSmall()

UNET_MODEL_FILE = 'UnetSmall/unet-final-8-2121-0.0001-169'
model.load_state_dict(torch.load(UNET_MODEL_FILE))
model#.cuda()

for i, image in tqdm(enumerate(test_loader)):
    files = test_images.this_img[i * batch_size:(i + 1) * batch_size]
    output = model(Variable(image))
    for j in range(len(files)):
        vutils.save_image(output[j][:,:,:],
                            '%s/%s' % ("第七代预测结果2D-1024/P", files[j].split("/")[-1]),
                            normalize=True)
        
        vutils.save_image(output[j][0:1,:,:],
                            '%s/%s' % ("第七代预测结果2D-1024/P1", files[j].split("/")[-1]),
                            normalize=True)
        
        vutils.save_image(output[j][1:2,:,:],
                            '%s/%s' % ("第七代预测结果2D-1024/P2", files[j].split("/")[-1]),
                            normalize=True)


