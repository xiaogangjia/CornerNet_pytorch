from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
import random
import math

from cornernet.module import convolution, tl_corner_pooling, br_corner_pooling, backbone, out_conv_layer
from cornernet.utils import gaussian2D, draw_gaussian, gaussian_radius


class Net(nn.Module):

    def __init__(self, dim):
        super(Net, self).__init__()

        self.backbone = backbone(n=5, dims=[256, 256, 384, 384, 384, 512], modules=[2, 2, 2, 2, 2, 4])

        self.top_left_pool = tl_corner_pooling(dim)
        self.bottom_right_pool = br_corner_pooling(dim)

        self.tl_heatmaps = out_conv_layer(dim, dim, 20)
        self.tl_embeddings = out_conv_layer(dim, dim, 1)
        self.tl_offsets = out_conv_layer(dim, dim, 2)

        self.br_heatmaps = out_conv_layer(dim, dim, 20)
        self.br_embeddings = out_conv_layer(dim, dim, 1)
        self.br_offsets = out_conv_layer(dim, dim, 2)

        self.standard_conv = convolution(3, dim, dim)

        self.preconv = convolution(7, 3, 128, stride=2)
        self.preresidual = convolution(3, 128, 256, stride=2)

    def forward(self, x):

        outs = []
        x = self.preconv(x)
        x = self.preresidual(x)

        feature_map = self.backbone(x)
        feature_map = self.standard_conv(feature_map)
        #top_left module
        tl_out = self.top_left_pool(feature_map)

        tl_heats = self.tl_heatmaps(tl_out)
        tl_embeds = self.tl_embeddings(tl_out)
        tl_offsets = self.tl_offsets(tl_out)

        #bottom_right module
        br_out = self.bottom_right_pool(feature_map)

        br_heats = self.br_heatmaps(br_out)
        br_embeds = self.br_embeddings(br_out)
        br_offsets = self.br_offsets(br_out)

        #outs += [tl_heats, tl_embeds, tl_offsets, br_heats, br_embeds, br_offsets]

        #return outs
        return tl_heats, tl_embeds, tl_offsets, br_heats, br_embeds, br_offsets


def coco_data_process(detections, img_names, train_num, batch_size, gaussian_flag=True):
    # 最多含正类的数目
    max_pos_len = 128

    images = np.zeros((batch_size, 3, 511, 511), dtype=np.float32)
    tl_heats = np.zeros((batch_size, 80, 128, 128), dtype=np.float32)
    br_heats = np.zeros((batch_size, 80, 128, 128), dtype=np.float32)
    tl_offsets = np.zeros((batch_size, 2, 128, 128), dtype=np.float32)
    br_offsets = np.zeros((batch_size, 2, 128, 128), dtype=np.float32)
    # 储存正类的位置
    tl_pos = np.zeros((batch_size, 128, 128), dtype=np.int64)
    br_pos = np.zeros((batch_size, 128, 128), dtype=np.int64)

    #tl_pos = np.zeros((batch_size, max_pos_len, 2), dtype=np.int64)
    #br_pos = np.zeros((batch_size, max_pos_len, 2), dtype=np.int64)
    # 储存图片中的物体数量
    obj_mask = np.zeros((batch_size, ), dtype=np.int64)

    w_ratio = 128/511
    h_ratio = 128/511

    for batch_num in range(batch_size):
        if train_num==0:
            #img_names.shuffle()
            random.shuffle(img_names)

        img_name = img_names[train_num]

        train_num = (train_num+1) % len(img_names)

        detection = detections[img_name]

        image = cv2.imread('dataset/train2014/' + img_name)
        h,w,c = image.shape

        image = cv2.resize(image, (511, 511))
        detection[:, 0:4:2] *= 511/w
        detection[:, 1:4:2] *= 511/h
        '''
        此处进行数据增强
        '''
        images[batch_num] = image.transpose((2, 0, 1))

        obj_num = 0
        for i, object in enumerate(detection):
            obj_num += 1
            category = int(object[-1]) - 1

            tlx, tly = object[0]-1, object[1]-1
            brx, bry = object[2]-1, object[3]-1
            # 映射在特征图上的位置
            map_tlx = tlx * w_ratio
            map_tly = tly * h_ratio
            map_brx = brx * w_ratio
            map_bry = bry * h_ratio
            # 向下取整
            tlx = int(map_tlx)
            tly = int(map_tly)
            brx = int(map_brx)
            bry = int(map_bry)

            if gaussian_flag:
                width = object[2] - object[0]
                height = object[3] - object[1]

                width = math.ceil(width * w_ratio)
                height = math.ceil(height * h_ratio)

                radius = gaussian_radius((height, width), 0.3)
                radius = max(0, int(radius))

                draw_gaussian(tl_heats[batch_num, category], [tlx, tly], radius)
                draw_gaussian(br_heats[batch_num, category], [brx, bry], radius)
            else:
                tl_heats[batch_num, category, tly, tlx] = 1
                br_heats[batch_num, category, bry, brx] = 1

            tl_offsets[batch_num, :, tly, tlx] = [map_tlx-tlx, map_tly-tly]
            br_offsets[batch_num, :, bry, brx] = [map_brx-brx, map_bry-bry]

            tl_pos[batch_num, tly, tlx] = obj_num
            br_pos[batch_num, bry, brx] = obj_num
            #tl_pos[batch_num, i, :] = [tlx, tly]
            #br_pos[batch_num, i, :] = [brx, bry]

        obj_mask[batch_num] = obj_num

    images     = torch.from_numpy(images)
    tl_heats   = torch.from_numpy(tl_heats)
    br_heats   = torch.from_numpy(br_heats)
    tl_offsets = torch.from_numpy(tl_offsets)
    br_offsets = torch.from_numpy(br_offsets)
    tl_pos     = torch.from_numpy(tl_pos)
    br_pos     = torch.from_numpy(br_pos)
    obj_mask   = torch.from_numpy(obj_mask)

    '''
    return {"xs": [images, tl_pos, br_pos],
            "ys": [tl_heats, br_heats, tl_offsets, br_offsets]}
    '''
    return {"xs": [images],
            "ys": [tl_heats, br_heats, tl_offsets, br_offsets, tl_pos, br_pos, obj_mask]}


def voc_data_process(detections, img_names, train_num, batch_size, gaussian_flag=True):

    images = np.zeros((batch_size, 3, 511, 511), dtype=np.float32)
    tl_heats = np.zeros((batch_size, 20, 128, 128), dtype=np.float32)
    br_heats = np.zeros((batch_size, 20, 128, 128), dtype=np.float32)
    tl_offsets = np.zeros((batch_size, 2, 128, 128), dtype=np.float32)
    br_offsets = np.zeros((batch_size, 2, 128, 128), dtype=np.float32)
    # 储存正类的位置
    tl_pos = np.zeros((batch_size, 128, 128), dtype=np.int64)
    br_pos = np.zeros((batch_size, 128, 128), dtype=np.int64)

    #tl_pos = np.zeros((batch_size, max_pos_len, 2), dtype=np.int64)
    #br_pos = np.zeros((batch_size, max_pos_len, 2), dtype=np.int64)
    # 储存图片中的物体数量
    obj_mask = np.zeros((batch_size, ), dtype=np.int64)

    w_ratio = 128/511
    h_ratio = 128/511

    for batch_num in range(batch_size):
        if train_num==0:
            #img_names.shuffle()
            random.shuffle(img_names)

        img_name = img_names[train_num]

        train_num = (train_num + 1) % len(img_names)

        detection = np.array(detections[img_name])

        image = cv2.imread('dataset/VOCdevkit/VOC2007/JPEGImages/' + img_name + '.jpg')
        h,w,c = image.shape

        image = cv2.resize(image, (511, 511))
        image = image.astype(np.float32) / 255.
        detection[:, 0:4:2] *= 511/w
        detection[:, 1:4:2] *= 511/h
        '''
        此处进行数据增强
        '''
        images[batch_num] = image.transpose((2, 0, 1))

        obj_num = 0
        for i, object in enumerate(detection):
            obj_num += 1
            category = int(object[-1])

            tlx, tly = object[0]-1, object[1]-1
            brx, bry = object[2]-1, object[3]-1
            # 映射在特征图上的位置
            map_tlx = tlx * w_ratio
            map_tly = tly * h_ratio
            map_brx = brx * w_ratio
            map_bry = bry * h_ratio
            # 向下取整
            tlx = int(map_tlx)
            tly = int(map_tly)
            brx = int(map_brx)
            bry = int(map_bry)

            if gaussian_flag:
                width = object[2] - object[0]
                height = object[3] - object[1]

                width = math.ceil(width * w_ratio)
                height = math.ceil(height * h_ratio)

                radius = gaussian_radius((height, width), 0.3)
                radius = max(0, int(radius))

                draw_gaussian(tl_heats[batch_num, category], [tlx, tly], radius)
                draw_gaussian(br_heats[batch_num, category], [brx, bry], radius)
            else:
                tl_heats[batch_num, category, tly, tlx] = 1
                br_heats[batch_num, category, bry, brx] = 1

            tl_offsets[batch_num, :, tly, tlx] = [map_tlx-tlx, map_tly-tly]
            br_offsets[batch_num, :, bry, brx] = [map_brx-brx, map_bry-bry]

            tl_pos[batch_num, tly, tlx] = obj_num
            br_pos[batch_num, bry, brx] = obj_num
            #tl_pos[batch_num, i, :] = [tlx, tly]
            #br_pos[batch_num, i, :] = [brx, bry]

        obj_mask[batch_num] = obj_num

    images     = torch.from_numpy(images)
    tl_heats   = torch.from_numpy(tl_heats)
    br_heats   = torch.from_numpy(br_heats)
    tl_offsets = torch.from_numpy(tl_offsets)
    br_offsets = torch.from_numpy(br_offsets)
    tl_pos     = torch.from_numpy(tl_pos)
    br_pos     = torch.from_numpy(br_pos)
    obj_mask   = torch.from_numpy(obj_mask)

    '''
    return {"xs": [images, tl_pos, br_pos],
            "ys": [tl_heats, br_heats, tl_offsets, br_offsets]}
    '''
    return train_num, \
           {"xs": [images],
            "ys": [tl_heats, br_heats, tl_offsets, br_offsets, tl_pos, br_pos, obj_mask]}

