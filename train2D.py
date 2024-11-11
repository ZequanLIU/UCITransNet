from __future__ import print_function

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.io as sio
import torchvision.transforms as tr
from IOU import SegmentationMetric
from PIL import Image
from data import GetUDD6
from losses import DICELossMultiClass,lovasz_softmax
from models import UNet,UNetSmall
from tqdm import tqdm
import numpy as np
import random
from NewCRFDepth import NewCRFDepth

DATA_FOLDER_FOR_TRAIN = "train.txt"
DATA_FOLDER_FOR_VAL = "val.txt"
batch_size = 6
test_batch_size = 6
size = (512, 512)
lr = 1e-4
beta1 = 0.5
beta2 = 0.99
log_interval = 20
savename = "NweCFRs"
epochs = 2001
seed = 2022
# torch.seed(seed)
random.seed(seed)
np.random.seed(seed)
dset_train = GetUDD6(DATA_FOLDER_FOR_TRAIN, train=True,
                              im_size=size)

train_loader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True, num_workers=1)

dset_test = GetUDD6(DATA_FOLDER_FOR_VAL, train=False,
                             im_size=size)

test_loader = DataLoader(dset_test,
                         batch_size=test_batch_size,
                         shuffle=False, num_workers=1)



# %% Loading in the model
model = UNet(num_classes=1)
# model = NewCRFDepth(version="tiny07", inv_depth=False, max_depth=6, pretrained=None)

UNET_MODEL_FILE = 'SaveModels/crfs-final-6-2121-0.001-394'
pretrained_dict  = torch.load(UNET_MODEL_FILE)
model_dict = model.state_dict()


pretrained_dict = {k: v for k, v in pretrained_dict.items()} #if "final.0" not in k}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.cuda()

optimizer = optim.AdamW(model.parameters(), lr=lr,
                       betas=(beta1, beta2))

# Defining Loss Function
criterion = DICELossMultiClass()


def train(epoch, loss_lsit):
    model.train()
    for batch_idx, (image, mask) in enumerate(train_loader):
        image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image), Variable(mask)

        optimizer.zero_grad()

        output, ssim_loss = model(image)
#         ssim_loss = ssim_loss
#         mask_numpy = np.argmax(mask.cpu().detach().numpy(),axis=1)
#         mask_numpy = mask.cpu().detach().numpy()
#         mask_numpy[mask_numpy > 0.5] = 1
#         loss = lovasz_softmax(output, torch.from_numpy(mask_numpy).cuda())
        loss = criterion(output, mask)
        loss_list.append((loss).data)

        (loss).backward()
        optimizer.step()

        
        
        
#         ignore_labels = [255]
#         metric = SegmentationMetric(6) # 3表示有3个分类，有几个分类就填几, 0也是1个分类
#         output_numpy = np.argmax(output.cpu().detach().numpy(),axis=1)
#         mask_numpy = np.argmax(mask.cpu().detach().numpy(),axis=1)
#         hist = metric.addBatch(torch.from_numpy(output_numpy).long(),torch.from_numpy(mask_numpy).long(),ignore_labels)
#         pa = metric.pixelAccuracy()
#         cpa = metric.classPixelAccuracy()
#         mpa = metric.meanPixelAccuracy()
#         IoU = metric.IntersectionOverUnion()
#         mIoU = metric.meanIntersectionOverUnion()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage DICE Loss: {:.10f}\tAverage ssim Loss: {:.6f}\tTotal Loss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data,ssim_loss.data,(loss + ssim_loss).data))
            
#             print('hist is :\n', hist)
#             print('PA is : %f' % pa.cpu().detach().numpy())
#             print('cPA is :', cpa.cpu().detach().numpy())  # 列表
#             print('mPA is : %f' % mpa.cpu().detach().numpy())
#             print('IoU is : ', IoU.cpu().detach().numpy())
#             print('mIoU is : ', mIoU.cpu().detach().numpy())


def test(train_accuracy=False, save_output=False):
    test_loss = 0
    model.eval()
    if train_accuracy:
        loader = train_loader
    else:
        loader = test_loader
    total_sum = 0
    total_len = 0
    total_class1 = 0
    total_class1_mask = 0
    for batch_idx, (image, mask) in tqdm(enumerate(loader)):
        image, mask = image.cuda(), mask.cuda()

        image, mask = Variable(image, volatile=True), Variable(
            mask, volatile=True)

        output, _ = model(image)
        
        
        ignore_labels = [255]
        metric = SegmentationMetric(1) # 3表示有3个分类，有几个分类就填几, 0也是1个分类
#         output_numpy = np.argmax(output.cpu().detach().numpy(),axis=1)
#         mask_numpy = np.argmax(mask.cpu().detach().numpy(),axis=1)
        
        
        
        mask_numpy = mask.cpu().detach().numpy()
        mask_numpy[mask_numpy > 0] = 1
        output_numpy = output.cpu().detach().numpy()
        output_numpy[output_numpy >= 0.5] = 1
        output_numpy[output_numpy != 1] = 0
        
        total_sum += np.sum(output_numpy == mask_numpy)
        total_class1 += np.sum(output_numpy * mask_numpy)
        total_len += len(np.reshape(output_numpy,[-1]))
        total_class1_mask += np.sum(mask_numpy)
#         acc = np.sum(output_numpy == mask_numpy) / len(np.reshape(output_numpy,[-1]))
#         print(mask_numpy.shape,output_numpy.shape)
        
        
#         hist = metric.addBatch(torch.from_numpy(output_numpy).long(),mask_numpy,ignore_labels)
#         pa = metric.pixelAccuracy()
#         cpa = metric.classPixelAccuracy()
#         mpa = metric.meanPixelAccuracy()
#         IoU = metric.IntersectionOverUnion()
#         mIoU = metric.meanIntersectionOverUnion()
        
#         # test_loss += criterion(output, mask).data[0]
#         maxes, out = torch.max(output, 1, keepdim=True)

#         if save_output and (not train_accuracy):
#             np.save('./npy-files/out-files/{}-batch-{}-outs.npy'.format(savename,
#                                                                         batch_idx),
#                     out.data.byte().cpu().numpy())
#             np.save('./npy-files/out-files/{}-batch-{}-masks.npy'.format(savename,
#                                                                          batch_idx),
#                     mask.data.byte().cpu().numpy())
#             np.save('./npy-files/out-files/{}-batch-{}-images.npy'.format(savename,
#                                                                           batch_idx),
#                     image.data.float().cpu().numpy())

#         if save_output and train_accuracy:
#             np.save('./npy-files/out-files/{}-train-batch-{}-outs.npy'.format(savename,
#                                                                               batch_idx),
#                     out.data.byte().cpu().numpy())
#             np.save('./npy-files/out-files/{}-train-batch-{}-masks.npy'.format(savename,
#                                                                                batch_idx),
#                     mask.data.byte().cpu().numpy())
#             np.save('./npy-files/out-files/{}-train-batch-{}-images.npy'.format(savename,
#                                                                                 batch_idx),
#                     image.data.float().cpu().numpy())

        test_loss += 1 - criterion(output, mask).data

    # Average Dice Coefficient
    test_loss /= len(loader)
    if train_accuracy:
        print('\nTraining Set: Average DICE Coefficient: {:.4f})\n'.format(
            test_loss))
        
#         print('hist is :\n', hist)
#         print('PA is : %f' % pa.cpu().detach().numpy())
#         print('cPA is :', cpa.cpu().detach().numpy())  # 列表
#         print('mPA is : %f' % mpa.cpu().detach().numpy())
#         print('IoU is : ', IoU.cpu().detach().numpy())
#         print('mIoU is : ', mIoU.cpu().detach().numpy())
    else:
        print('\nTest Set: Average DICE Coefficient: {:.4f})\n'.format(
            test_loss))
#         print('hist is :\n', hist)
#         print('PA is : %f' % pa.cpu().detach().numpy())
#         print('cPA is :', cpa.cpu().detach().numpy())  # 列表
#         print('mPA is : %f' % mpa.cpu().detach().numpy())
#         print('IoU is : ', IoU.cpu().detach().numpy())
#         print('mIoU is : ', mIoU.cpu().detach().numpy())
    print("ACC : ",total_sum / total_len,"Class1 : ",total_class1 / total_class1_mask)

loss_list = []
for i in tqdm(range(epochs)):
    test()
    train(i, loss_list)
    if i % 1 == 0:
        print("Save Path:",'SaveModels/crfs-final-{}-{}-{}-{}'.format(batch_size,120 + epochs,lr,i))
        torch.save(model.state_dict(), 'SaveModels/crfs-final-{}-{}-{}-{}'.format(batch_size,120 + epochs,lr,i))


np.save('./npy-files/loss-files/{}-UNet_Loss_bs={}_ep={}_lr={}.npy'.format(savename,batch_size,epochs,lr),np.asarray(loss_list))

                                                            
