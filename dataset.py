import os
import torch
import random
import PIL.Image as Image

import torch.utils.data as data

from pdb import set_trace as bp


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
        if is_train == True and (Config.module == 'OEL' or Config.module == 'LIO'):
            print('load and store positive_image_list for OEL module')
            self.positive_image_list = self.get_positive_images(self.paths, self.labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        imgpath = os.path.join(self.root_path, self.paths[item])
        img = self.pil_loader(imgpath)
        if self.is_train:
            img = self.common_aug(img) if not self.common_aug is None else img
        img = self.totensor(img)
        label = self.labels[item] - 1
        return img, label, self.paths[item]

    def pil_loader(self, imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get_positive_images(self, imgpath_list, label_list):
        # for OEL, we get positive_images first. it takes some minutes (2~3 min)!
        positive_image_list = {}
        import time
        start = time.time()
        for i in range(self.numcls):
            positive_image_list[i] = []
        
        for imgpath, label in zip(imgpath_list, label_list):
            imgpath = os.path.join(self.root_path, imgpath)
            img = self.pil_loader(imgpath)
            if self.is_train:
                img = self.common_aug(img) if not self.common_aug is None else img
            img = self.totensor(img)
            label = label - 1
            positive_image_list[label].append(img)
        print('time check', time.time() - start)

        return positive_image_list


def collate_fn(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name
