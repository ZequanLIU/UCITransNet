import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
import os
from dataParser import getMaskFileName, getImg, getMask, getMasks
import torchvision.transforms as tr 
import random

class AddPepperNoise(object):
    """增加椒盐噪声
    Args:
        snr （float）: Signal Noise Rate
        p (float): 概率值，依概率执行该操作
    """
 
    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) and (isinstance(p, float))    # 2020 07 26 or --> and
        self.snr = snr
        self.p = p
 
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:      #概率判断
            img_ = np.array(img).copy()
            h, w, c = img_.shape
            signal_pct = self.snr              #信噪比。信噪比0.9，说明信号占90%
            noise_pct = (1 - self.snr)         #噪声占比0.1
            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = random.randint(0,255)   # 盐噪声
            img_[mask == 2] = random.randint(0,255)     # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img
class AddLowq(object):
 
    def __init__(self, snr, p=0.85):
        assert isinstance(snr, float) and (isinstance(p, float)) 
        self.snr = snr
        self.p = p
    def rebuild_img(self,u, sigma, v): #p表示奇异值的百分比
        m = len(u)
        n = len(v)
        a = np.zeros((m, n))
        
        count = (int)(sum(sigma))
        curSum = 0
        k = 0
        while curSum <= count * self.snr:
            uk = u[:, k].reshape(m, 1)
            vk = v[k].reshape(1, n)
            a += sigma[k] * np.dot(uk, vk)
            curSum += sigma[k]
            k += 1
        a[a < 0] = 0
        a[a > 255] = 255
        #按照最近距离取整数，并设置参数类型为uint8
        return np.rint(a).astype("uint8")
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        if random.uniform(0, 1) < self.p:      #概率判断
            a = np.array(img)
            u, sigma, v = np.linalg.svd(a[:, :, 0])
            R = self.rebuild_img(u, sigma, v)

            u, sigma, v = np.linalg.svd(a[:, :, 1])
            G = self.rebuild_img(u, sigma, v)

            u, sigma, v = np.linalg.svd(a[:, :, 2])
            B = self.rebuild_img(u, sigma, v)

            I=np.stack((R, G, B), 2)
            return Image.fromarray(I)
        else:
            return img

def get_udd6mask(file_path):
    mask = Image.open(file_path)
    mask = np.array(mask)
    masks = [np.zeros_like(mask) for _ in range(6)]
    masks = np.array(masks)
    for i in range(6):
        masks[i][mask == i] = 255
    mask1 = Image.fromarray(masks.transpose([1,2,0])[:,:,:3])
    mask2 = Image.fromarray(masks.transpose([1,2,0])[:,:,3:])
    return mask1,mask2

def get_udd1mask(file_path):
    mask = Image.open(file_path)
    mask = np.array(mask)
    
    mask[mask == 1] = 255
    return Image.fromarray(mask)
class GetUDD6(Dataset):

    def __init__(self, dataset_folders_path, train=True, im_size=[2160, 4096]):

        self.train = train
        self.dataset_folders = []
        self.im_size = im_size
        train_set = open(dataset_folders_path,"r",encoding="utf-8").read()
        for line in train_set.split("\n")[:-1]:
            self.dataset_folders.append(["train/" + line,"masks/" + line])
        self.dataset_folders = np.array(self.dataset_folders)
        
    def __getitem__(self, index):

        img = getImg(self.dataset_folders[index][0])
        mask = get_udd1mask(self.dataset_folders[index][1])
        
        if self.train:
            seed = np.random.randint(0,10000)
            if seed > 5000:
                seed = 1
            else:
                seed = 0
            transform_data=tr.Compose([tr.Resize(self.im_size),
                                  tr.RandomHorizontalFlip(p=seed),
                                  tr.RandomVerticalFlip(p=seed),
#                                   AddLowq(np.random.randint(500,999) / 1000),
#                                   AddPepperNoise(0.9),
                                  tr.ToTensor()])
            transform_mask=tr.Compose([tr.Resize(self.im_size),
                                  tr.RandomHorizontalFlip(p=seed),
                                  tr.RandomVerticalFlip(p=seed),
                                  tr.ToTensor()])
            img = transform_data(img)
            mask = transform_mask(mask)
        else:
            transform=tr.Compose([tr.Resize(self.im_size),
                            tr.ToTensor()
                            ])
            img = transform(img)
            mask = transform(mask)

        mask[mask > 0] = 1
        return img, mask

    def __len__(self):

        return len(self.dataset_folders)