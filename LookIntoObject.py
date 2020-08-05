import random

import numpy as np
import torch
import torch.nn as nn
import pretrainedmodels

from pdb import set_trace as bp


class Model(nn.Module):

    def __init__(self, Config, is_train=True):
        super(Model, self).__init__()
        self.module = Config.module
        self.num_classes = Config.numcls
        self.backbone_arch = Config.backbone
        self.model = pretrainedmodels.__dict__[self.backbone_arch](num_classes=1000, pretrained='imagenet')
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # this is 14x14!!
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(2048, self.num_classes, bias=False)

        self.maxpool_for_featuremap_7x7 = nn.MaxPool2d(2, 2)
        if is_train and (Config.module == 'OEL' or Config.module == 'SCL' or Config.module == 'LIO'):
            # To avoid Nan loss, added ReLu to make <0 value into zero.
            # To stablize the SCL loss, added BN.
            self.OEL_mask_small_m = nn.Sequential(nn.Conv2d(2048, 1, 1, 1), nn.BatchNorm2d(1), nn.ReLU(True))

        if is_train and (Config.module == 'SCL' or Config.module == 'LIO'):
            self.SCL_conv = nn.Sequential(nn.Conv2d(2048, 512, 1, 1), nn.ReLU(True))
            self.SCL_fc = nn.Sequential(nn.Linear(512 * 2, 2), nn.ReLU(True))

    def forward(self, x, is_train=True, get_featuremap_7x7=False):
        if get_featuremap_7x7:
            return self.get_featuremap_7x7(x)

        common_feature = self.model(x)

        # classification backbone
        cls_feature = self.avgpool(common_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        cls_prediction = self.classifier(cls_feature)

        if is_train and (self.module == 'LIO' or self.module == 'SCL'):
            featuremap_7x7 = self.maxpool_for_featuremap_7x7(common_feature)
            oel_mask = self.OEL_mask_small_m(featuremap_7x7)
            scl_polar_coordinate = self.SCL(featuremap_7x7, oel_mask.detach())
            return cls_prediction, featuremap_7x7, oel_mask, scl_polar_coordinate

        elif is_train and self.module == 'OEL':
            featuremap_7x7 = self.maxpool_for_featuremap_7x7(common_feature)
            oel_mask = self.OEL_mask_small_m(featuremap_7x7)
            return cls_prediction, featuremap_7x7, oel_mask

        return cls_prediction

    def get_featuremap_7x7(self, x):
        common_feature = self.model(x)
        featuremap_7x7 = self.maxpool_for_featuremap_7x7(common_feature)
        return featuremap_7x7

    def SCL(self, featuremap, mask):
        featuremap_h = self.SCL_conv(featuremap)  # N, C, H, W

        """ find R0 following to equation (6) then concat channels where R0 index
        1) change 2D coordinates (h, w) -> 1D (h*w) to use torch.max() and find feature where R0 index easily.
        2) find argmax coordinate for each mask.
        3) then change back change 1D coordinates (h*w) -> 2D (h, w).
        """
        # equation (6) in the paper
        N, C, H, W = mask.size()
        mask_1D = mask.view(N, C, H * W)  # C = 1
        _, R0_index = torch.max(mask_1D, dim=2)  # [N]
        # print('R0', R0_index, R0_index.size())

        N, C, H, W = featuremap_h.size()
        featuremap_h_1D = featuremap_h.view(N, C, H * W)  # C = 512
        feature_whereR0 = torch.zeros((N, C)).cuda()
        for batch_idx, R0_idx in enumerate(R0_index):
            feature_whereR0_index = featuremap_h_1D[batch_idx, :, R0_idx]
            feature_whereR0[batch_idx, :] = feature_whereR0_index.squeeze(1)

        feature_whereR0_expand = feature_whereR0.unsqueeze(2).expand(N, C, H * W)
        feature_whereR0_2d = feature_whereR0_expand.view(N, C, H, W)
        h_concat_R0 = torch.cat((featuremap_h, feature_whereR0_2d), dim=1)
        h_concat_R0 = h_concat_R0.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C for calculate FC
        scl_pred = self.SCL_fc(h_concat_R0)
        scl_pred = scl_pred.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        # print(scl_pred.size())

        """ make SCL gt with R0_index """
        scl_gt = torch.zeros((N, 2, H, W)).cuda()
        R0_index = R0_index.float()
        y = R0_index // W
        x = R0_index % W
        y = y.squeeze(1)  # [N, 1] -> [N]
        x = x.squeeze(1)

        for i in range(W):
            for j in range(H):
                # equation (5) in paper.
                scl_gt[:, 0, i, j] = torch.sqrt((x - i)**2 + (y - j)**2) / (2**(1 / 2) * N)
                scl_gt[:, 1, i, j] = (torch.atan2(y - j, x - i) + np.pi) / (2 * np.pi)

        scl_polar_coordinate = {}
        scl_polar_coordinate['gt'] = scl_gt
        scl_polar_coordinate['pred'] = scl_pred

        return scl_polar_coordinate


def OEL_make_pseudo_mask(model, featuremap_7x7, label_list, positive_image_list, positive_image_number=3):

    model.eval()
    # select positive images
    sampled_positive_image_list = []
    for label in label_list:
        sampled_positive_image = torch.stack(random.sample(
            positive_image_list[label.item()], positive_image_number), dim=0)
        sampled_positive_image_list.append(sampled_positive_image)

    # N * [positive_image_number, C, H, W] -> [N*positive_image_number, C, H, W]
    positive_images = torch.cat(sampled_positive_image_list, dim=0)
    # print('pi', positive_images.size())

    with torch.no_grad():
        """ 
        1) change 2D coordinates (h, w) -> 1D (h*w) to use torch.bmm and torch.max easily.
        2) calculate for each coordinates.
        3) then change back change 1D coordinates (h*w) -> 2D (h, w).
        """
        N, C, H, W = featuremap_7x7.size()
        featuremap_7x7_1D = featuremap_7x7.view(N, C, H * W)
        
        # To use for loop, change [N*positive_image_number, C, H, W] -> [positive_image_number, N, C, H, W]
        positive_image_featuremap_7x7 = model(positive_images, get_featuremap_7x7=True)
        positive_image_featuremap_7x7 = positive_image_featuremap_7x7.unsqueeze(
            0).view(positive_image_number, N, C, H, W)
        
        phi = []
        for pos_img_7x7 in positive_image_featuremap_7x7:
            pos_img_7x7_1D = pos_img_7x7.view(N, C, H * W)
            
            # equation (2) in the paper. matmul ([N, H*W, C], [N, C, H*W]) = [N, H*W, H*W] for calculate correlation.
            dot_product = torch.bmm(featuremap_7x7_1D.permute(0, 2, 1), pos_img_7x7_1D)
            phi.append(torch.max(dot_product, dim=2)[0] / C)  # [N, H*W]
            # print('dot, phi', dot_product.size(), torch.max(dot_product, dim=2)[0].size())

        # equation (3) in the paper
        phi_1D = torch.mean(torch.stack(phi, dim=0), dim=0)  # [positive_image_number, N, H*W] -> [N, H*W]
        pseudo_mask = phi_1D.view(N, H, W)  # [N, H, W]
        pseudo_mask = pseudo_mask.unsqueeze(1) # [N, C, H, W] (C=1)
        # print('phi1D, pseudo_mask', phi_1D.size(), pseudo_mask.size())

    model.train()
    return pseudo_mask


# def get_SCL_loss(pred_list, gt_list, mask_list):
def get_SCL_loss(pred, gt, mask):

    pred_dis, pred_angle = pred[:, 0, :, :], pred[:, 1, :, :]
    gt_dis, gt_angle = gt[:, 0, :, :], gt[:, 1, :, :]

    # calculate mean_angle first
    N, _, I, J = mask.size()

    # Seems like if mask elements < 0, the loss gonna be NaN!
    # find hint from here https://discuss.pytorch.org/t/getting-nan-after-first-iteration-with-custom-loss/25929/12
    mask_sum = torch.sum(mask, dim=(2, 3))  # [N, 1, H, W] -> [N, 1] # denominator of equation (7), (8)
    mask_sum = mask_sum.squeeze(1)

    mean_angle_numerate_sum = 0
    angle_for_mean = torch.zeros(N, I, J).cuda()
    for i in range(I):
        for j in range(J):
            sub = pred_angle[:, i, j] - gt_angle[:, i, j]
            sub_morethan_zero = sub >= 0
            sub_lessthan_zero = sub < 0
            angle_for_mean[sub_morethan_zero, i, j] = sub[sub >= 0]
            angle_for_mean[sub_lessthan_zero, i, j] = 1 + sub[sub < 0]
            mean_angle_numerate_sum += mask[:, 0, i, j] * angle_for_mean[:, i, j]

    mean_angle = (mean_angle_numerate_sum + 1e-6) / (mask_sum + 1e-6)

    # calculate loss_dis (relative distance) and loss_angle (polar angle)
    dis_numerator_sum = 0
    angle_numerator_sum = 0
    angle = torch.zeros(N, I, J).cuda()
    for i in range(I):
        for j in range(J):
            # for numerator equation (7) in the paper
            dis_numerator_sum += mask[:, 0, i, j] * (pred_dis[:, i, j] - gt_dis[:, i, j])**2

            # for numerator equation (8) in the paper
            sub = pred_angle[:, i, j] - gt_angle[:, i, j]
            sub_morethan_zero = sub >= 0
            sub_lessthan_zero = sub < 0
            angle[sub_morethan_zero, i, j] = sub[sub >= 0]
            angle[sub_lessthan_zero, i, j] = 1 + sub[sub < 0]
            angle_numerator_sum += mask[:, 0, i, j] * (angle[:, i, j] - mean_angle)**2

    loss_dis = torch.sqrt((dis_numerator_sum + 1e-6) / (mask_sum + 1e-6))
    loss_angle = torch.sqrt((angle_numerator_sum + 1e-6) / (mask_sum + 1e-6))

    loss_dis = torch.sum(loss_dis, dim=0)
    loss_angle = torch.sum(loss_angle, dim=0)

    # equation (9) in the paper
    loss = loss_dis + loss_angle
    # print(loss, loss_dis, loss_angle)

    return loss
