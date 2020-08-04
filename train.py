# coding=utf-8
import os
import time
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from Model import MainModel
from config import LoadConfig, load_data_transformers
from dataset import collate_fn, dataset
from utils import LossRecord, eval_turn

import pdb


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          data_size=448,
          save_per_epoch=5
          ):

    step = 0
    rec_loss = []
    checkpoint_list = []
    log_file = open(os.path.join(Config.exp_name, f'log.txt'), 'a')

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    get_ce_loss = nn.CrossEntropyLoss()

    start_time = time.time()
    model.train()
    for epoch in range(start_epoch, epoch_num - 1):
        # exp_lr_scheduler.step(epoch)

        save_grad = []
        for step, data in enumerate(data_loader['train']):
            inputs, labels, img_names = data
            inputs = inputs.cuda()
            labels = torch.from_numpy(np.array(labels)).cuda()

            outputs = model(inputs)
            loss = get_ce_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            if step % 100 == 0:
                elapsed_time = int(time.time() - start_time)
                print(f'epoch: {epoch} / {epoch_num} step: {step:-8d} / {train_epoch_step:d} loss: {loss.detach().item():6.4f} lr: {current_lr:0.8f} elapsed_time: {elapsed_time}', flush=True)
            rec_loss.append(loss.detach().item())
            train_loss_recorder.update(loss.detach().item())

        # evaluation & save
        if epoch % save_per_epoch == 0:
            rec_loss = []
            print(80 * '-', flush=True)
            print(f'epoch: {epoch} step: {step:d} / {train_epoch_step:d} global_step: {1.0 * step / train_epoch_step:8.2f} train_epoch: {epoch:04d} train_loss: {train_loss_recorder.get_val():6.4f}', flush=True)
            model.eval()
            test_acc1, test_acc2, test_acc3 = eval_turn(Config, model, data_loader['test'], 'test', epoch, log_file)
            model.train()

            save_path = os.path.join(Config.exp_name, f'weights_{epoch}_{test_acc1:0.4f}_{test_acc3:0.4f}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'saved model to {save_path}', flush=True)

    log_file.close()


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--exp_name', default='tmp', type=str)
    parser.add_argument('--spe', dest='save_per_epoch', default=5, type=int)
    parser.add_argument('--data', dest='dataset', default='CUB', type=str)
    parser.add_argument('--save', dest='resume', default=None, type=str)
    parser.add_argument('--backbone', dest='backbone', default='resnet50', type=str)
    parser.add_argument('--auto_resume', dest='auto_resume', action='store_true')
    parser.add_argument('--epoch', dest='epoch', default=50, type=int)
    parser.add_argument('--tb', dest='train_batch', default=16, type=int)
    # There was no valid set in DCL repository (2020.8)
    parser.add_argument('--vb', dest='test_batch', default=128, type=int)
    parser.add_argument('--lr', dest='base_lr', default=0.0008, type=float)
    parser.add_argument('--lr_step', dest='decay_step', default=60, type=int)
    parser.add_argument('--cls_lr_ratio', dest='cls_lr_ratio', default=10.0, type=float)
    parser.add_argument('--start_epoch', dest='start_epoch', default=0,  type=int)
    parser.add_argument('--tnw', dest='train_num_workers', default=4, type=int)
    parser.add_argument('--vnw', dest='test_num_workers', default=4, type=int)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        print('For my setting, check GPU number, you should be missed CUDA_VISIBLE_DEVICES=0 or typo')
        sys.exit()

    args = parse_args()
    print(args, flush=True)
    Config = LoadConfig(args, 'train')
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)

    # inital dataloader
    train_set = dataset(Config=Config,
                        anno=Config.train_anno,
                        common_aug=transformers["common_aug"],
                        totensor=transformers["train_totensor"],
                        train=True)

    test_set = dataset(Config=Config,
                       anno=Config.test_anno,
                       common_aug=transformers["None"],
                       totensor=transformers["test_totensor"],
                       test=True)

    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(train_set,
                                                      batch_size=args.train_batch,
                                                      shuffle=True,
                                                      num_workers=args.train_num_workers,
                                                      collate_fn=collate_fn,
                                                      pin_memory=False)

    setattr(dataloader['train'], 'total_item_len', len(train_set))

    dataloader['test'] = torch.utils.data.DataLoader(test_set,
                                                     batch_size=args.test_batch,
                                                     shuffle=False,
                                                     num_workers=args.test_num_workers,
                                                     collate_fn=collate_fn,
                                                     pin_memory=False)

    setattr(dataloader['test'], 'total_item_len', len(test_set))
    setattr(dataloader['test'], 'num_cls', Config.numcls)

    cudnn.benchmark = True

    print('train from imagenet pretrained models ...', flush=True)
    model = MainModel(Config)
    model = nn.DataParallel(model).cuda()

    # optimizer prepare
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.1)

    # for super convergence with one cycle learning rate
    step_up_size = len(dataloader['train']) * args.epoch / 2
    step_down_size = len(dataloader['train']) * args.epoch / 2
    scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.base_lr * 10,
                                      step_size_up=step_up_size, step_size_down=step_down_size,
                                      cycle_momentum=False)

    # train entry
    train(Config,
          model,
          epoch_num=args.epoch,
          start_epoch=args.start_epoch,
          optimizer=optimizer,
          exp_lr_scheduler=scheduler,
          data_loader=dataloader,
          data_size=args.crop_resolution,
          save_per_epoch=args.save_per_epoch)
