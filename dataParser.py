import os
from PIL import Image
import numpy as np
from skimage import color
from skimage import io


def getMaskFileName(file):

    mask_file = file.replace("flair.png", "seg.png")
    mask_file = mask_file.replace("t1.png", "seg.png")
    mask_file = mask_file.replace("t2.png", "seg.png")
    mask_file = mask_file.replace("t1ce.png", "seg.png")

    return mask_file


def getImg(imgpathway):
    img = Image.open(imgpathway)
    return img



def getMask(imgpathway):
    img = Image.open(imgpathway)
    img = np.array(img)[:,:,0] + np.array(img)[:,:,1] + np.array(img)[:,:,2]
    img[img>0] = 255
    img = img.reshape([img.shape[0],img.shape[1],1])
    return Image.fromarray(np.concatenate([img,img,img],axis = -1))


# def getMasks(imgpathway1,imgpathway2):
#     img1 = Image.open(imgpathway1)
#     img1 = np.array(img1)[:,:,0] + np.array(img1)[:,:,1] + np.array(img1)[:,:,2]
#     img1[img1!=0] = 255
#     img1 = img1.reshape([img1.shape[0],img1.shape[1],1])
    
    
#     img2 = Image.open(imgpathway2)
#     img2 = np.array(img2)[:,:,0] + np.array(img2)[:,:,1] + np.array(img2)[:,:,2]
#     img2[img2!=0] = 255
#     img2 = img2.reshape([img2.shape[0],img2.shape[1],1])
    
#     return Image.fromarray(np.concatenate([img2,img1],axis = -1))
def getMasks(imgpathway1,imgpathway2):
    img1 = Image.open(imgpathway1)
    img1 = np.array(img1)[:,:,0] + np.array(img1)[:,:,1] + np.array(img1)[:,:,2]
    img1[img1!=0] = 255
    img1 = img1.reshape([img1.shape[0],img1.shape[1],1])
    
    
    img2 = Image.open(imgpathway2)
    img2 = np.array(img2)[:,:,0] + np.array(img2)[:,:,1] + np.array(img2)[:,:,2]
    img2[img2!=0] = 255
    img2 = img2.reshape([img2.shape[0],img2.shape[1],1])
#     img1[img2!=0] = 255
    return Image.fromarray(np.concatenate([img2,img1],axis = -1))

def File2Image(self, index):
    file = self.__file[index]
    filename_fragments = file.split("_")
    if filename_fragments[1] == '0' or filename_fragments[1] == '154':
        # Not sure what to do here
        return 0, 0

    filename1 = filename_fragments[0] + filename_fragments[1] + '_' + \
        str(int(filename_fragments[2]) - 1) + '_' + filename_fragments[3]
    filename3 = filename_fragments[0] + filename_fragments[1] + '_' + \
        str(int(filename_fragments[2]) + 1) + '_' + filename_fragments[3]

    idx1 = self.__file.index(filename1)
    idx3 = self.__file.index(filename3)
    img1 = self.__im[idx1]
    img3 = self.__im[idx3]

    return img1, img3