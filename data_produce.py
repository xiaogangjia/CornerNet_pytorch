from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import random


def coco_produce():

    dataDir = 'dataset'
    dataType = 'train2014'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    image_dir = '{}/{}/'.format(dataDir, dataType)

    # initialize COCO api for instance annotations
    coco = COCO(annFile)

    # return all the category ids
    catIds = coco.getCatIds()
    print(catIds)

    # return all the ids of images
    imgIds = coco.getImgIds()

    # return the filename of the image
    img_names = [coco.loadImgs(img_id)[0]["file_name"] for img_id in imgIds]

    detections = {}

    _cat_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90
    ]
    _classes = {
        ind + 1: cat_id for ind, cat_id in enumerate(_cat_ids)
    }
    _coco_to_class_map = {
        value: key for key, value in _classes.items()
    }

    cats = coco.loadCats(catIds)
    print(len(cats))

    for (imgId, img_name) in zip(imgIds, img_names):
        image = coco.loadImgs(imgId)[0]

        bboxes = []
        categories = []

        for catId in catIds:
            annotationIds = coco.getAnnIds(imgIds=image["id"], catIds=catId)
            annotations = coco.loadAnns(annotationIds)
            category = _coco_to_class_map[catId]
            for annotation in annotations:
                bbox = np.array(annotation["bbox"])
                bbox[[2, 3]] += bbox[[0, 1]]
                bboxes.append(bbox)

                categories.append(category)
        # tlx,tly,brx,bry
        bboxes = np.array(bboxes, dtype=float)
        categories = np.array(categories, dtype=float)

        if bboxes.size==0 or categories.size==0:
            detections[img_name] = np.zeros((0,5), dtype=np.float32)
        else:
            detections[img_name] = np.hstack((bboxes, categories[:, None]))

        #print(detections)

    return detections, img_names, cats


classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']


def get_bbox(file):
    tree = ET.parse(file)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    bbox = []

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        bbox.append([xmin,ymin,xmax,ymax,cls])

    return bbox


def voc_produce(year):
    dataDir   = 'dataset'
    dataType  = 'VOCdevkit'

    ann_dir   = '{}/{}/{}/Annotations/'.format(dataDir, dataType, 'VOC'+year)
    train_txt = '{}/{}/{}/train.txt'.format(dataDir, dataType, 'VOC'+year)
    val_txt   = '{}/{}/{}/val.txt'.format(dataDir, dataType, 'VOC'+year)

    with open(train_txt) as f:
        train_files = f.readlines()
        train_files = [file.strip() for file in train_files]

    with open(val_txt) as f:
        val_files = f.readlines()
        val_files = [file.strip() for file in val_files]

    train_detections = {}
    val_detections   = {}

    for img in train_files:
        bbox = get_bbox(ann_dir + img + '.xml')
        train_detections[img] = bbox

    for img in val_files:
        bbox = get_bbox(ann_dir + img + '.xml')
        val_detections[img] = bbox

    return train_detections, val_detections


if __name__ == '__main__':
    #coco_produce()
    trains, vals = voc_produce('2007')
    print(len(trains))
    print(trains)
    print(vals)

    keys = list(trains.keys())
    random.shuffle(keys)
    print(keys)
