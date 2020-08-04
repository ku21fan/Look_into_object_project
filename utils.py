import os
import time
import datetime

import numpy as np
import torch


class LossRecord(object):

    def __init__(self, batch_size):
        self.rec_loss = 0
        self.count = 0
        self.batch_size = batch_size

    def update(self, loss):
        if isinstance(loss, list):
            avg_loss = sum(loss)
            avg_loss /= (len(loss) * self.batch_size)
            self.rec_loss += avg_loss
            self.count += 1
        if isinstance(loss, float):
            self.rec_loss += loss / self.batch_size
            self.count += 1

    def get_val(self, init=False):
        pop_loss = self.rec_loss / self.count
        if init:
            self.rec_loss = 0
            self.count = 0
        return pop_loss


def cls_base_acc(result_gather):
    top1_acc = {}
    top3_acc = {}
    cls_count = {}
    for img_item in result_gather.keys():
        acc_case = result_gather[img_item]

        if acc_case['label'] in cls_count:
            cls_count[acc_case['label']] += 1
            if acc_case['top1_cat'] == acc_case['label']:
                top1_acc[acc_case['label']] += 1
            if acc_case['label'] in [acc_case['top1_cat'], acc_case['top2_cat'], acc_case['top3_cat']]:
                top3_acc[acc_case['label']] += 1
        else:
            cls_count[acc_case['label']] = 1
            if acc_case['top1_cat'] == acc_case['label']:
                top1_acc[acc_case['label']] = 1
            else:
                top1_acc[acc_case['label']] = 0

            if acc_case['label'] in [acc_case['top1_cat'], acc_case['top2_cat'], acc_case['top3_cat']]:
                top3_acc[acc_case['label']] = 1
            else:
                top3_acc[acc_case['label']] = 0

    for label_item in cls_count:
        top1_acc[label_item] /= max(1.0 * cls_count[label_item], 0.001)
        top3_acc[label_item] /= max(1.0 * cls_count[label_item], 0.001)
        top1_acc[label_item] = round(top1_acc[label_item], 1)
        top3_acc[label_item] = round(top3_acc[label_item], 1)

    print('top1_acc:', top1_acc)
    print('top3_acc:', top3_acc)
    print('cls_count', cls_count)

    return top1_acc, top3_acc, cls_count


def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def eval_turn(Config, model, data_loader, val_version, epoch_num):
    val_corrects1 = 0
    val_corrects2 = 0
    val_corrects3 = 0
    val_size = data_loader.__len__()
    item_count = data_loader.total_item_len
    t0 = time.time()
    get_ce_loss = torch.nn.CrossEntropyLoss()

    val_batch_size = data_loader.batch_size
    val_epoch_step = data_loader.__len__()
    num_cls = data_loader.num_cls

    val_loss_recorder = LossRecord(val_batch_size)
    print('evaluating %s ...' % val_version)
    with torch.no_grad():
        for step, data_val in enumerate(data_loader):
            inputs = data_val[0].cuda()
            labels = torch.from_numpy(np.array(data_val[1])).long().cuda()
            outputs = model(inputs, is_train=False)

            loss = get_ce_loss(outputs, labels).item()
            val_loss_recorder.update(loss)

            outputs_pred = outputs
            top3_val, top3_pos = torch.topk(outputs_pred, 3)

            if step % 20 == 0:
                print('{:s} eval_batch: {:-6d} / {:d} loss: {:8.4f}'.format(val_version,
                                                                            step, val_epoch_step, loss))

            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)

        val_acc1 = val_corrects1 / item_count * 100
        val_acc2 = val_corrects2 / item_count * 100
        val_acc3 = val_corrects3 / item_count * 100

        t1 = time.time()
        since = t1 - t0
        print('-' * 80)
        test_log = '% 3d %s %s %s-loss: %.4f ||%s-acc@1: %.1f %s-acc@2: %.1f %s-acc@3: %.1f ||time: %d' % (epoch_num, val_version, dt(), val_version,
                                                                                                           val_loss_recorder.get_val(init=True), val_version, val_acc1, val_version, val_acc2, val_version, val_acc3, since)
        print(test_log)
        with open(os.path.join(Config.exp_name, 'log.txt'), 'a') as log_file:
            log_file.write(test_log + '\n')
        print('-' * 80)

    return val_acc1, val_acc2, val_acc3
