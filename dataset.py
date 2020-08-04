import os
import torch
import random
import PIL.Image as Image

import torch.utils.data as data

import pdb


class dataset(data.Dataset):

    def __init__(self, Config, anno, common_aug=None, totensor=None, is_train=True):
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        self.paths = anno['ImageName'].tolist()
        self.labels = anno['label'].tolist()
    
        self.common_aug = common_aug
        self.totensor = totensor
        self.cfg = Config
        self.is_train = is_train
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(img_path)
        if self.is_train:
            img = self.common_aug(img) if not self.common_aug is None else img
        img = self.totensor(img)
        label = self.labels[item] - 1
        return img, label, self.paths[item]

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


def collate_fn(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name
