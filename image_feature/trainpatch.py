# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, 'metric_learning')
import time
import paddle
import paddle.fluid as fluid
import train_elem as trainmodule
import myreader
import logging
import resnet18
import l2net
import numpy as np
import cv2
from myreader import update_argv

#替换老的reader


def preprocess(img, mode):
    img = cv2.resize(img, (32, 32))
    return img.reshape((1, img.shape[0], img.shape[1]))


def train(args):
    def train_reader():
        #readerfunc =  myreader.myreader_classify
        readerfunc = myreader.create_multiprocessreader(
            myreader.myreader_classify, 4)

        traindataset = readerfunc(
            args.train_datasetfile,
            args.train_labelfile,
            'train',
            iscolor=0,
            preprocessfunc=preprocess)
        for image, label in traindataset:
            yield image, label

    return train_reader


def val(args):
    def val_reader():
        traindataset = myreader.myreader_classify(
            args.val_datasetfile,
            args.val_labelfile,
            'val',
            doshuffle=False,
            iscolor=0,
            preprocessfunc=preprocess)
        for image, label in traindataset:
            yield image, label

    return val_reader


class Dataset(object):
    def __init__(self, train, val, mean, std):
        self.train = train
        self.val = val
        self.test = val
        self.img_mean = mean
        self.img_std = std


#替换训练主模块里面的reader，并增加ResNet18模型
trainmodule.reader = Dataset(train, val, mean=[127.5], std=[128.0])


class Models(object):
    def __init__(self, modelinfo):
        for modelname in modelinfo:
            setattr(self, modelname, modelinfo[modelname])


trainmodule.models = Models({
    'L2Net': l2net.L2Net,
    'ResNet18': resnet18.ResNet18
})
trainmodule.model_list = ['L2Net', 'ResNet18']


def trainmain():
    defaultargv = [
        'train.py',
        #"--use_gpu=false",
        #"--checkpoint=output/L2Net/12000/",
        #"--pretrained_model=pretrained_model",
        "--input_dtype=uint8",
        #"--model=L2Net",
        "--model=ResNet18",
        "--train_batch_size=512",
        "--test_batch_size=64",
        "--embedding_size=64",
        "--class_dim=500000",
        "--image_shape=1,32,32",
        "--lr=0.1",
        "--lr_strategy=cosine_decay",
        #"--lr_strategy=cosine_decay_with_warmup",
        #"--warmup_iter_num=6000",
        "--display_iter_step=5",
        "--total_iter_num=60000",
        "--test_iter_step=100",
        "--save_iter_step=3000",
        "--loss_name=arcmargin",
        "--arc_scale=80",
        "--arc_margin=0.2",
        "--train_datasetfile=dataset/samepatch_train/samepatch_train.data",
        "--train_labelfile=dataset/samepatch_train/samepatch_train_500000.label",
        "--val_datasetfile=dataset/samepatch_train/samepatch_train.data",
        "--val_labelfile=dataset/samepatch_train/samepatch_test_44803.label",
    ]

    update_argv(defaultargv)
    trainmodule.main()


if __name__ == '__main__':
    trainmain()
    #testforward()
