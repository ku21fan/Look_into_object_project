import os
import pandas as pd
import torch

import torchvision.transforms as transforms


def load_data_transformers(resize_reso=512, crop_reso=448):
    Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    data_transforms = {
        'common_aug': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
        ]),
        'train_totensor': transforms.Compose([
            transforms.Resize((crop_reso, crop_reso)),
            transforms.ToTensor(),
            Normalize,
        ]),
        'test_totensor': transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.CenterCrop((crop_reso, crop_reso)),
            transforms.ToTensor(),
            Normalize,
        ]),
        'None': None,
    }
    return data_transforms


class LoadConfig(object):

    def __init__(self, args, version):
        if version == 'train':
            get_list = ['train', 'test']
        elif version == 'test':
            get_list = ['test']

        if args.dataset == 'CUB':
            self.dataset = args.dataset
            self.rawdata_root = './datasets/CUB/data'
            self.anno_root = './datasets/CUB/anno'
            self.numcls = 200
        elif args.dataset == 'STCAR':
            self.dataset = args.dataset
            self.rawdata_root = './datasets/STCAR/data'
            self.anno_root = './datasets/STCAR/anno'
            self.numcls = 196
        elif args.dataset == 'AIR':
            self.dataset = args.dataset
            self.rawdata_root = './datasets/AIR/data'
            self.anno_root = './datasets/AIR/anno'
            self.numcls = 100

        if 'train' in get_list:
            self.train_anno = pd.read_csv(os.path.join(self.anno_root, 'train.txt'),
                                          sep=" ",
                                          header=None,
                                          names=['ImageName', 'label'])

        if 'test' in get_list:
            self.test_anno = pd.read_csv(os.path.join(self.anno_root, 'test.txt'),
                                         sep=" ",
                                         header=None,
                                         names=['ImageName', 'label'])
        if version == 'train':
            self.exp_name = f'./saved_models/{args.dataset}_{args.exp_name}_seed{args.seed}'
            os.makedirs(self.exp_name, exist_ok=True)
        self.backbone = args.backbone
        self.module = args.module
