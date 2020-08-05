import os
import time
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from LookIntoObject import Model
from config import LoadConfig
from dataset import collate_fn, dataset
from utils import eval_turn, load_data_transformers

from pdb import set_trace as bp


def train(Config,
          model,
          epoch_num,
          optimizer,
          exp_lr_scheduler,
          dataset,
          data_loader,
          data_size=448,
          save_per_epoch=5
          ):
    train_epoch_step = data_loader['train'].__len__()
    
    # set loss function and positive image list for OEL
    get_cls_loss = torch.nn.CrossEntropyLoss()
    if Config.module == 'LIO' or Config.module == 'OEL' or Config.module == 'SCL':
        from LookIntoObject import OEL_make_pseudo_mask, get_SCL_loss
        get_OEL_loss = torch.nn.MSELoss()
        if Config.module == 'LIO' or Config.module == 'OEL':
            positive_image_list = dataset.positive_image_list

    start_time = time.time()
    model.train()
    epoch = 0
    while(epoch < epoch_num):
        for step, data in enumerate(data_loader['train']):
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = torch.from_numpy(np.array(labels)).cuda()

            if Config.module == 'LIO':
                outputs, featuremap_7x7, oel_mask, scl_polar_coordinate = model(inputs)
                cls_loss = get_cls_loss(outputs, labels)
                pseudo_mask = OEL_make_pseudo_mask(model, featuremap_7x7.detach(), labels, positive_image_list)
                OEL_loss = get_OEL_loss(oel_mask, pseudo_mask)  # equation (4) in the paper
                SCL_loss = get_SCL_loss(scl_polar_coordinate['pred'], scl_polar_coordinate['gt'], oel_mask.detach())

                # equation (10) in the paper.
                loss = cls_loss + 0.1 * OEL_loss + 0.1 * SCL_loss

            elif Config.module == 'OEL':
                outputs, featuremap_7x7, oel_mask = model(inputs)
                cls_loss = get_cls_loss(outputs, labels)
                pseudo_mask = OEL_make_pseudo_mask(model, featuremap_7x7.detach(), labels, positive_image_list)
                OEL_loss = get_OEL_loss(oel_mask, pseudo_mask)  # equation (4) in the paper
                loss = cls_loss + 0.1 * OEL_loss

            elif Config.module == 'SCL':
                outputs, featuremap_7x7, oel_mask, scl_polar_coordinate = model(inputs)
                cls_loss = get_cls_loss(outputs, labels)
                SCL_loss = get_SCL_loss(scl_polar_coordinate['pred'], scl_polar_coordinate['gt'], oel_mask.detach())
                loss = cls_loss + 0.1 * SCL_loss

            else:
                outputs = model(inputs)
                loss = get_cls_loss(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            exp_lr_scheduler.step()
            
            # for train log
            if step % 100 == 0:
                elapsed_time = int(time.time() - start_time)
                current_lr = optimizer.param_groups[0]['lr']
                if Config.module == 'LIO':
                    train_log = f'epoch: {epoch} / {epoch_num} step: {step:-5d} / {train_epoch_step:d} loss: {loss.detach().item():6.4f}, OEL_loss: {OEL_loss.detach().item():6.4f}, SCL_loss: {SCL_loss.detach().item():6.4f} lr: {current_lr:0.8f} elapsed_time: {elapsed_time}'
                elif Config.module == 'OEL':
                    train_log = f'epoch: {epoch} / {epoch_num} step: {step:-5d} / {train_epoch_step:d} loss: {loss.detach().item():6.4f}, OEL_loss: {OEL_loss.detach().item():6.4f} lr: {current_lr:0.8f} elapsed_time: {elapsed_time}'
                elif Config.module == 'SCL':
                    train_log = f'epoch: {epoch} / {epoch_num} step: {step:-5d} / {train_epoch_step:d} loss: {loss.detach().item():6.4f}, SCL_loss: {SCL_loss.detach().item():6.4f} lr: {current_lr:0.8f} elapsed_time: {elapsed_time}'
                else:
                    train_log = f'epoch: {epoch} / {epoch_num} step: {step:-5d} / {train_epoch_step:d} loss: {loss.detach().item():6.4f} lr: {current_lr:0.8f} elapsed_time: {elapsed_time}'

                print(train_log)
                with open(os.path.join(Config.exp_name, 'log.txt'), 'a') as log_file:
                    log_file.write(train_log + '\n')
            
        epoch += 1

        # evaluation and save models. To see training progress, we also conduct evaluation when 'epoch==1'
        if epoch % save_per_epoch == 0 or epoch == 1:
            print(80 * '-')
            model.eval()
            test_acc1, test_acc2, test_acc3 = eval_turn(Config, model, data_loader['test'], 'test', epoch)
            model.train()

            if epoch != 1:
                save_path = os.path.join(Config.exp_name, f'weights_{epoch}_{test_acc1:0.4f}_{test_acc3:0.4f}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'saved model to{save_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--exp_name', default='tmp', type=str, help='experiment name')
    parser.add_argument('--seed', default=1111, type=int, help='random seed')
    parser.add_argument('--data', dest='dataset', default='CUB', type=str)
    parser.add_argument('--save', dest='resume', default=None, type=str, help='path to saved model')
    parser.add_argument('--epoch', dest='epoch', default=50, type=int)
    parser.add_argument('--spe', dest='save_per_epoch', default=5, type=int)
    parser.add_argument('--tb', dest='train_batch', default=16, type=int)
    parser.add_argument('--testb', dest='test_batch', default=128, type=int)
    parser.add_argument('--tnw', dest='train_num_workers', default=4, type=int)
    parser.add_argument('--vnw', dest='test_num_workers', default=4, type=int)
    parser.add_argument('--lr', dest='base_lr', default=0.0008, type=float)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)
    # model
    parser.add_argument('--backbone', dest='backbone', default='resnet50', type=str)
    parser.add_argument('--mo', dest='module', default='onlyCLS', type=str,
                        help='|Look-into-Object (LIO)|Object Extent Learning (OEL)|Spatial Context Learning (SCL)|')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        print('For my setting, only use 1 GPU so it should be missed CUDA_VISIBLE_DEVICES=0 or typo')
        sys.exit()

    args = parse_args()
    print(args)
    Config = LoadConfig(args, 'train')
    with open(os.path.join(Config.exp_name, 'log.txt'), 'a') as log_file:
        log_file.write(str(args) + '\n')

    """Seed and GPU setting"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    
    """ dataset & dataloader """
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)
    train_set = dataset(Config=Config,
                        anno=Config.train_anno,
                        common_aug=transformers["common_aug"],
                        totensor=transformers["train_totensor"],
                        is_train=True)

    test_set = dataset(Config=Config,
                       anno=Config.test_anno,
                       common_aug=transformers["None"],
                       totensor=transformers["test_totensor"],
                       is_train=False)

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

    print('Load imagenet pretrained model ResNet50')
    model = Model(Config)
    model = torch.nn.DataParallel(model).cuda()
    # print(model)
    with open(os.path.join(Config.exp_name, 'log.txt'), 'a') as log_file:
        log_file.write(repr(model) + '\n')

    # optimizer prepare
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    # exp_lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.decay_step,gamma=0.1)

    # for superconvergence with one cycle learning rate
    step_up_size = len(dataloader['train']) * args.epoch / 2
    step_down_size = len(dataloader['train']) * args.epoch / 2
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.base_lr, max_lr=args.base_lr * 10,
                                                  step_size_up=step_up_size, step_size_down=step_down_size,
                                                  cycle_momentum=False)

    # train
    train(Config,
          model,
          epoch_num=args.epoch,
          optimizer=optimizer,
          exp_lr_scheduler=scheduler,
          dataset=train_set,
          data_loader=dataloader,
          data_size=args.crop_resolution,
          save_per_epoch=args.save_per_epoch)
