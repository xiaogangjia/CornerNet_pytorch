import torch
from tensorboardX import SummaryWriter
import numpy as np
import cv2
import random

from cornernet.model import Net, coco_data_process, voc_data_process
from cornernet.loss import Loss
from data_produce import coco_produce, voc_produce, classes

import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda:0')


def train_coco():
    # 生成数据标签detections={'img_file_name':[bboxes category]}
    detections, img_names, cats = coco_produce()

    '''
    for j in range(5):
        image = cv2.imread('dataset/train2014/' + img_names[j])
        detection = detections[img_names[j]]
        for i in detection:
            frame = cv2.rectangle(image, (int(i[0]), int(i[1])), (int(i[2]), int(i[3])), (255, 0, 0), 2)
            class_name = cats[int(i[4])-1]['name']
            cv2.putText(frame, class_name, (int(i[0]), int(i[1]) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200), 1)
        cv2.imshow('test', frame)
        cv2.waitKey(0)
    '''
    num_image = len(img_names)

    batch = 2
    iterations = num_image * 50

    net = Net(256)
    net.to(device)
    net.train()

    loss = Loss(pull_weight=0.1, push_weight=0.1, off_weight=1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.00025, momentum=0.9, weight_decay=0.0001)

    epoch = 50
    train_num = 0
    for i in range(iterations):
        optimizer.zero_grad()

        inputs_dic = coco_data_process(detections, img_names, train_num, batch)

        xs = inputs_dic["xs"]
        ys = inputs_dic["ys"]

        xs = [x.to(device) for x in xs]
        ys = [y.to(device) for y in ys]

        preds = net(xs[0])
        # preds = [pred.to(device) for pred in preds]
        # print(preds[0].shape)
        # print(preds[3].shape)
        # print(ys[0].shape)
        # print(ys[1].shape)

        all_loss = loss(preds, ys)
        # all_loss = all_loss.mean()
        all_loss.backward()
        optimizer.step()

        print("step {}: ".format(i), all_loss.item())
        '''
        if (i+1) % 100 == 0:
            print("iteration: {}".format(i))
            print("loss: {}".format(all_loss))
            print("save model: ")

        if (i+1) % num_image == 0:
            print("one epoch")
        '''


def train_voc():
    writer = SummaryWriter('runs/cornernet')

    trains, vals = voc_produce('2007')
    img_names = list(trains.keys())

    train_nums = len(trains)
    val_nums   = len(vals)

    batch = 4
    iterations = train_nums * 50
    #iterations = 1

    net = Net(256)

    #net.load_state_dict(torch.load('models/16999_net_params.pkl'))
    start_iter = 0

    net.to(device)
    net.train()

    loss = Loss(pull_weight=0.1, push_weight=0.1, off_weight=1)
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.00025, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
    #                            lr=0.00025, momentum=0.9, weight_decay=0.0001)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                 lr=1e-4)

    train_num = 0
    loss_scalar = 0
    for i in range(start_iter, iterations):
        optimizer.zero_grad()

        train_num, inputs_dic = voc_data_process(trains, img_names, train_num, batch)

        xs = inputs_dic["xs"]
        ys = inputs_dic["ys"]

        xs = [x.to(device) for x in xs]
        ys = [y.to(device) for y in ys]

        writer.add_graph(net, xs[0])

        #preds = net(xs[0])
        tl_heats, tl_embeds, tl_offsets, br_heats, br_embeds, br_offsets = net(xs[0])
        preds = [tl_heats, tl_embeds, tl_offsets, br_heats, br_embeds, br_offsets]

        all_loss = loss(preds, ys)
        all_loss = all_loss.mean()
        all_loss.backward()
        optimizer.step()

        print("step {}: ".format(i), all_loss.item())
        '''
        loss_scalar += all_loss.item()
        if (i+1) % 30 == 0:
            writer.add_scalar('training loss',
                              loss_scalar / 30,
                              i)
            loss_scalar = 0
        '''
        if (i+1) % 500 == 0:
            print("iteration: {}".format(i))
            print("loss: {}".format(all_loss.item()))
            print("save params to : " + "models/{}_net_params.pkl".format(i))

            torch.save(net.state_dict(), 'models/{}_net_params.pkl'.format(i))


if __name__ == '__main__':
    train_voc()
