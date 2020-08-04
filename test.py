import os
import json
import argparse
import numpy as np
from math import ceil
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from Model import MainModel
from dataset import collate_fn, dataset
from config import LoadConfig, load_data_transformers
from utils import cls_base_acc

import pdb


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset', default='CUB', type=str)
    parser.add_argument('--backbone', dest='backbone', default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size', default=128, type=int)
    parser.add_argument('--nw', dest='num_workers', default=4, type=int)
    parser.add_argument('--ver', dest='version', default='test', type=str)
    parser.add_argument('--save', dest='resume', default=None, type=str)
    parser.add_argument('--size', dest='resize_resolution', default=512, type=int)
    parser.add_argument('--crop', dest='crop_resolution', default=448, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        print('For my setting, only use 1 GPU so it should be missed CUDA_VISIBLE_DEVICES=0 or typo' )
        sys.exit()

    args = parse_args()
    print(args)
    Config = LoadConfig(args, args.version)
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution)
    data_set = dataset(Config,
                       anno=Config.test_anno,
                       totensor=transformers['test_totensor'],
                       is_train=False)

    dataloader = torch.utils.data.DataLoader(data_set,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=collate_fn)

    setattr(dataloader, 'total_item_len', len(data_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(args.resume))
    
    model.eval()
    with torch.no_grad():
        test_corrects1 = 0
        test_corrects2 = 0
        test_corrects3 = 0
        test_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        count_bar = tqdm(total=dataloader.__len__())
        for batch_cnt_test, data_test in enumerate(dataloader):
            count_bar.update(1)
            inputs, labels, img_name = data_test
            inputs = inputs.cuda()
            labels = torch.from_numpy(np.array(labels)).long().cuda()

            outputs = model(inputs)
            outputs_pred = outputs

            top3_test, top3_pos = torch.topk(outputs_pred, 3)

            
            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            test_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            test_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            test_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

            for sub_name, sub_cat, sub_test, sub_label in zip(img_name, top3_pos.tolist(), top3_test.tolist(), labels.tolist()):
                result_gather[sub_name] = {'top1_cat': sub_cat[0], 'top2_cat': sub_cat[1], 'top3_cat': sub_cat[2],
                                           'top1_test': sub_test[0], 'top2_test': sub_test[1], 'top3_test': sub_test[2],
                                           'label': sub_label}
    
    folder_name = args.resume.split('/')[-2]
    os.makedirs(f'results/{folder_name}', exist_ok=True)
    file_base_name = args.resume.split('/')[-1]
    torch.save(result_gather, f'results/{folder_name}/result_gather_{file_base_name}.pt')

    count_bar.close()

    test_acc1 = test_corrects1 / len(data_set) * 100
    test_acc2 = test_corrects2 / len(data_set) * 100
    test_acc3 = test_corrects3 / len(data_set) * 100
    print('%sacc1 %.1f%s\n%sacc2 %.1f%s\n%sacc3 %.1f%s\n' %
          (8 * '-', test_acc1, 8 * '-', 8 * '-', test_acc2, 8 * '-', 8 * '-',  test_acc3, 8 * '-'))

    cls_top1, cls_top3, cls_count = cls_base_acc(result_gather)

    acc_report_io = open(f'results/{folder_name}/acc_report_{file_base_name}.json', 'w')
    json.dump({'test_acc1': test_acc1,
               'test_acc2': test_acc2,
               'test_acc3': test_acc3,
               'cls_top1': cls_top1,
               'cls_top3': cls_top3,
               'cls_count': cls_count}, acc_report_io)
    acc_report_io.close()
