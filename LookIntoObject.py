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
            self.OEL_mask_small_m = nn.Conv2d(2048, 1, 1, 1)

        if is_train and (Config.module == 'SCL' or Config.module == 'LIO'):
            self.SCL_conv = nn.Sequential(nn.Conv2d(2048, 512, 1, 1), nn.ReLU(True))
            self.SCL_fc = nn.Sequential(nn.Linear(512 * 2, 2), nn.ReLU(True))

    def forward(self, x, is_train=True):
        common_feature = self.model(x)

        # classification backbone
        cls_feature = self.avgpool(common_feature)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        cls_prediction = self.classifier(cls_feature)

        if is_train and (self.module == 'LIO' or self.module == 'SCL'):
            featuremap_7x7 = self.maxpool_for_featuremap_7x7(common_feature)
            oel_mask = self.OEL_mask_small_m(featuremap_7x7)
            scl_polar_coordinate = self.SCL(featuremap_7x7, oel_mask.detach())
            return cls_prediction, oel_mask, scl_polar_coordinate

        elif is_train and self.module == 'OEL':
            featuremap_7x7 = self.maxpool_for_featuremap_7x7(common_feature)
            oel_mask = self.OEL_mask_small_m(featuremap_7x7)
            return cls_prediction, oel_mask

        return cls_prediction

    def SCL(self, featuremap, mask):
        featuremap_h = self.SCL_conv(featuremap)  # N, C, H, W

        """ find R0 following to equation (6) then concat channels where R0 index
        1) change 2D coordinates (h, w) -> 1D (h*w).
        2) find argmax coordinate for each mask.
        3) then change back change 1D coordinates (h*w) -> 2D (h, w).
        """
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
        for batch_idx in range(N):
            y = R0_index[batch_idx] // W
            x = R0_index[batch_idx] % W

            for i in range(W):
                for j in range(H):
                    # equation (5) in paper.
                    scl_gt[batch_idx, 0, i, j] = torch.sqrt((x - i)**2 + (y - j)**2) / (2**(1 / 2) * N)
                    scl_gt[batch_idx, 1, i, j] = (torch.atan2(y - j, x - i) + np.pi) / (2 * np.pi)

        scl_polar_coordinate = {}
        scl_polar_coordinate['gt'] = scl_gt
        scl_polar_coordinate['pred'] = scl_pred

        return scl_polar_coordinate


def OEL_make_pseudo_mask(model, image_list, label_list, positive_image_list, positive_image_number=3):

    # label 별 positive image 필요 = 미리셋을 만들어두고, label에 맞는 것을 샘플링 해와야겟네?!
    model.eval()
    pseudo_mask_list = []
    with torch.no_grad():
        for image, label in zip(image_list, label_list):
            image_featuremap_14x14 = model.model(image)
            image_featuremap_7x7 = model.maxpool_for_featuremap_7x7(image_featuremap_14x14)

            positive_images = random.sample(positive_image_list[label], positive_image_number)
            positive_image_featuremap_14x14 = model.model(positive_images)
            positive_image_featuremap_7x7 = model.maxpool_for_featuremap_7x7(positive_image_featuremap_14x14)

            """ 
            1) change 2D coordinates (h, w) -> 1D (h*w).
            2) calculate in each coordinates.
            3) then change back change 1D coordinates (h*w) -> 2D (h, w).
            """
            N, C, H, W = image_featuremap_7x7.size()
            image_featuremap_1D = image_featuremap_7x7.view(N, C, H * W)  # N = 1
            image_featuremap_1D.squeeze(0)

            positive_image_featuremap_1D = positive_image_featuremap_7x7.view(N, C, H * W)  # N = 3

            phi_list = []
            for pos_img_index in range(positive_image_number):
                phi = []
                for image_feat in torch.unbind(image_featuremap_1D, 1):
                    print(positive_image_featuremap_1D[pos_img_index])  # [C, 49]
                    # positive_images_feat_49 = torch.unbind(positive_image_featuremap_1D[pos_img_index], 1) <-- 요거 없어도 될듯?
                    # feature_dot_product = torch.dot(image_feat, positive_images_feat_49) # [C, 1] x [C, 49] = [49]
                    feature_dot_product = torch.dot(image_feat, positive_image_featuremap_1D[
                                                    pos_img_index])  # [C, 1] x [C, 49] = [49]
                    phi.append(torch.max(feature_dot_product)[0] / C)

                phi_1D = torch.cat(phi, dim=0)
                phi_2D = torch.view(phi, (H, W))
                phi_list.append(phi_2D)

            phi_list = torch.stack(phi_list, dim=0)
            pseudo_mask = torch.mean(phi_list, dim=0)
            pseudo_mask_list.append(pseudo_mask)

    pseudo_mask_list = torch.cat(pseudo_mask_list, dim=0)
    model.train()

    return pseudo_mask_list


def get_SCL_loss(pred_list, gt_list, mask_list):

    loss_dis_list = []
    loss_angle_list = []
    for pred, gt, mask in zip(pred_list, gt_list, mask_list):

        pred_dis, pred_angle = pred
        gt_dis, gt_angle = gt

        # calculate mean_angle first
        mean_angle_numerate_sum = 0
        _, I, J = mask.size()

        # Seems like if mask elements < 0, the loss gonna be NaN! 
        # find hint from here https://discuss.pytorch.org/t/getting-nan-after-first-iteration-with-custom-loss/25929/12
        mask_clone = mask.clone() # to solve inplace error
        mask_clone[mask < 0] = 0 
        mask[mask < 0] = 0
        mask = mask_clone.detach()
        mask_sum = torch.sum(mask, dim=(1, 2))  # [1, H, W] -> [1] # denominator of equation (7), (8)

        for i in range(I):
            for j in range(J):
                angle = (pred_angle[i, j] - gt_angle[i, j]) if pred_angle[i, j] - \
                    gt_angle[i, j] >= 0 else (1 + pred_angle[i, j] - gt_angle[i, j])
                mean_angle_numerate_sum += mask[0, i, j] * angle

        mean_angle = mean_angle_numerate_sum / mask_sum

        # calculate loss_dis (relative distance) and loss_angle (polar angle)
        dis_numerator_sum = 0
        angle_numerator_sum = 0
        for i in range(I):
            for j in range(J):
                # for numerator equation (7) in the paper
                dis_numerator_sum += mask[0, i, j] * (pred_dis[i, j] - gt_dis[i, j])**2

                # for numerator equation (8) in the paper
                if pred_angle[i, j] - gt_angle[i, j] >= 0:
                    angle = (pred_angle[i, j] - gt_angle[i, j])
                else:
                    angle = (1 + pred_angle[i, j] - gt_angle[i, j])
                angle_numerator_sum += mask[0, i, j] * (angle - mean_angle)**2

        loss_dis_list.append(torch.sqrt(dis_numerator_sum / mask_sum))
        loss_angle_list.append(torch.sqrt(angle_numerator_sum / mask_sum))
        # print(mask_sum, dis_numerator_sum, angle_numerator_sum)

    loss_dis = torch.sum(torch.stack(loss_dis_list, dim=0), dim=0)
    loss_angle = torch.sum(torch.stack(loss_angle_list, dim=0), dim=0)

    # equation (9) in the paper
    loss = loss_dis + loss_angle
    # print(loss, loss_dis, loss_angle)

    return loss
