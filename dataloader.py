import torch
import numpy as np
import cv2
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import torch.utils.data as data

import os,sys
sys.path.append('.')
sys.path.append('..')
import random
from PIL import Image

random.seed(1143)

def correct_gt(mat):
    # clahe_images = np.zeros_like(mat)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    for j in range(3):
        mat[:, :, j] = clahe.apply(mat[:, :, j])
    return mat

class UIEBD_Dataset(data.Dataset):
    def __init__(self,path,train,size=None,format='.png'):
        super(UIEBD_Dataset,self).__init__()
        self.crop_size=size
        self.train=train
        self.format=format
        self.haze_imgs=[os.path.join(path,'input',img) for img in os.listdir(os.path.join(path,'input'))]
        self.clear_dir=os.path.join(path,'target')
    def __getitem__(self, index):
        raw_img=Image.open(self.haze_imgs[index]).convert("RGB")
        id=os.path.basename(self.haze_imgs[index])
        clear_name=id
        gt_img=Image.open(os.path.join(self.clear_dir,clear_name)).convert("RGB")


        if self.train:

            if np.random.rand(1) < 0.4:
                gt_img_arr = correct_gt(np.array(gt_img))
                gt_img = Image.fromarray(gt_img_arr)

            i, j, h, w = transforms.RandomResizedCrop(self.crop_size).get_params(raw_img, (0.08, 1.0),
                                                                                 (3. / 4., 4. / 3.))
            raw_cropped = F.resized_crop(raw_img, i, j, h, w, (self.crop_size, self.crop_size),
                                         InterpolationMode.BICUBIC)
            gt_cropped = F.resized_crop(gt_img, i, j, h, w, (self.crop_size, self.crop_size), InterpolationMode.BICUBIC)
            raw_img = transforms.ToTensor()(raw_cropped)
            gt_img = transforms.ToTensor()(gt_cropped)
            if np.random.rand(1) < 0.5:  # flip horizonly
                raw_img = torch.flip(raw_img, [2])
                gt_img = torch.flip(gt_img, [2])
            if np.random.rand(1) < 0.5:  # flip vertically
                raw_img = torch.flip(raw_img, [1])
                gt_img = torch.flip(gt_img, [1])
            rand_rot = random.randint(0, 3)
            raw_img = F.rotate(raw_img, 90 * rand_rot)
            gt_img = F.rotate(gt_img, 90 * rand_rot)
        else:
            # raw_img = transforms.Resize((raw_img.size[1], raw_img.size[0]))(raw_img)
            # gt_img = transforms.Resize((gt_img.size[1], gt_img.size[0]))(gt_img)
            # raw_img = transforms.Resize((256,256))(raw_img)
            # gt_img = transforms.Resize((256,256))(gt_img)
            raw_img = transforms.ToTensor()(raw_img)
            gt_img = transforms.ToTensor()(gt_img)
        return raw_img,gt_img,id
    def __len__(self):
        return len(self.haze_imgs)

class single_Dataset(data.Dataset):
    def __init__(self, path,  format='.png'):
        super(single_Dataset, self).__init__()


        self.format = format
        self.haze_imgs = [os.path.join(path,img) for img in os.listdir(path)]

    def __getitem__(self, index):
        raw_img = Image.open(self.haze_imgs[index]).convert("RGB")
        id = os.path.basename(self.haze_imgs[index])
        raw_img = transforms.ToTensor()(raw_img)

        return raw_img,id

    def __len__(self):
        return len(self.haze_imgs)