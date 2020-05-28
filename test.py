import torch
import torch.nn as nn
import cv2
import numpy as np

from data_produce import classes
from cornernet.model import Net
from cornernet.utils import _nms, _topk

device = torch.device('cuda:0')


def inference(file_dir):

    net = Net(256)
    net.load_state_dict(torch.load('models/24999_net_params.pkl'))

    net.to(device)
    net.eval()

    image = cv2.imread(file_dir)
    image = image.astype(np.float32)
    image = cv2.resize(image, (511, 511))
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]

    image = torch.from_numpy(image)
    image = image.to(device)

    tl_heats, tl_embeds, tl_offsets, br_heats, br_embeds, br_offsets = net(image)
    tl_heats = torch.clamp(torch.sigmoid(tl_heats), min=1e-4, max=1 - 1e-4)
    br_heats = torch.clamp(torch.sigmoid(br_heats), min=1e-4, max=1 - 1e-4)

    tl_heats = _nms(tl_heats, kernel=3)
    br_heats = _nms(br_heats, kernel=3)

    tl_scores, tl_clses, tl_ys, tl_xs = _topk(tl_heats, K=100)
    br_scores, br_clses, br_ys, br_xs = _topk(br_heats, K=100)

    tl_ys = tl_ys[0]
    tl_xs = tl_xs[0]
    br_ys = br_ys[0]
    br_xs = br_xs[0]
    tl_embed = []
    br_embed = []
    for i in range(100):
        tl_embed.append(tl_embeds[0, 0, tl_ys[i].int(), tl_xs[i].int()])
        br_embed.append(br_embeds[0, 0, br_ys[i].int(), br_xs[i].int()])

        tl_offset = tl_offsets[0, :, tl_ys[i].int(), tl_xs[i].int()]
        br_offset = br_offsets[0, :, br_ys[i].int(), br_xs[i].int()]
        # 加上预测偏移量
        tl_ys[i] += tl_offset[1]
        tl_xs[i] += tl_offset[0]
        br_ys[i] += br_offset[1]
        br_xs[i] += br_offset[0]

    tl_scores = tl_scores.view(100, 1).expand(100, 100)
    br_scores = br_scores.view(1, 100).expand(100, 100)
    scores    = (tl_scores + br_scores) / 2

    # reject distance > 0.5 or from different calsses
    tl_clses = tl_clses.view(100, 1).expand(100, 100)
    br_clses = br_clses.view(1, 100).expand(100, 100)
    cls_ind  = (tl_clses != br_clses)

    tl_embed = torch.tensor(tl_embed)
    br_embed = torch.tensor(br_embed)
    tl_embed = tl_embed.view(100, 1)
    br_embed = br_embed.view(1, 100)

    dists = torch.abs(tl_embed-br_embed)
    dis_ind = (dists > 0.5)

    scores[cls_ind] = -1
    scores[dis_ind] = -1
    tl, br = torch.where(scores > 0.5)

    boxes = []
    for num in range(len(tl)):
        tlx = tl_xs[tl[num]] * 4
        tly = tl_ys[tl[num]] * 4
        brx = br_xs[br[num]] * 4
        bry = br_ys[br[num]] * 4
        cls = tl_clses[tl[num], br[num]]

        if brx > tlx and bry > tly:
            boxes.append([tlx, tly, brx, bry, cls])

    img = cv2.imread(file_dir)
    img = cv2.resize(img, (511, 511))
    for box in boxes:
        tlx, tly, brx, bry, cls = box
        img = cv2.rectangle(img, (int(tlx), int(tly)), (int(brx), int(bry)), (255, 0, 0), 2)
        cv2.putText(img, classes[cls], (int(tlx), int(tly) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 1)
        cv2.imshow("test", img)
        cv2.waitKey(0)

    '''
    tl_heats = tl_heats.data.cpu().numpy()
    br_heats = br_heats.data.cpu().numpy()
    tl_offsets = tl_offsets.data.cpu().numpy()
    br_offsets = br_offsets.data.cpu().numpy()
    tl_embeds = tl_embeds.data.cpu().numpy()
    br_embeds = br_embeds.data.cpu().numpy()

    tl = np.argwhere(tl_heats > 0.4)
    br = np.argwhere(br_heats > 0.4)

    tly = tl[0, 2]
    tlx = tl[0, 3]
    bry = br[0, 2]
    brx = br[0, 3]

    tl_offset = tl_offsets[0, :, tly, tlx]
    br_offset = br_offsets[0, :, bry, brx]

    img_tlx = (tlx + tl_offset[0]) * 4
    img_tly = (tly + tl_offset[1]) * 4
    img_brx = (brx + br_offset[0]) * 4
    img_bry = (bry + br_offset[1]) * 4

    category = tl[0, 1]

    img = cv2.imread(file_dir)
    img = cv2.resize(img, (511, 511))
    img = cv2.rectangle(img, (int(img_tlx), int(img_tly)), (int(img_brx), int(img_bry)), (255, 0, 0), 2)
    cv2.putText(img, classes[category], (int(img_tlx), int(img_tly) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imshow("test", img)
    cv2.waitKey(0)
    '''


if __name__ == '__main__':
    inference("dataset/VOCdevkit/VOC2007/JPEGImages/009460.jpg")

