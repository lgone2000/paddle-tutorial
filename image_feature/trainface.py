# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from myreader import update_argv

#替换老的reader


def train(args):
    def train_reader():
        #readerfunc =  myreader.myreader_classify
        readerfunc = myreader.create_multiprocessreader(
            myreader.myreader_classify, 4)
        traindataset = readerfunc(args.train_datasetfile, args.train_labelfile,
                                  'train')
        for image, label in traindataset:
            yield image, label

    return train_reader


def warmup(args):
    def train_reader():
        warmup_labelfile = 'dataset/face_ms1m_small/warmup.label'
        readerfunc = myreader.myreader_classify
        traindataset = readerfunc(args.train_datasetfile, warmup_labelfile,
                                  'test')
        for image, label in traindataset:
            yield image, label

    return train_reader


def train(args):
    def train_reader():
        #readerfunc =  myreader.myreader_classify
        readerfunc = myreader.create_multiprocessreader(
            myreader.myreader_classify, 4)
        traindataset = readerfunc(args.train_datasetfile, args.train_labelfile,
                                  'train')
        for image, label in traindataset:
            yield image, label

    return train_reader


def val(args):
    def val_reader():
        traindataset = myreader.myreader_classify(
            args.val_datasetfile, args.val_labelfile, 'val', doshuffle=False)
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
        self.warmup = warmup


#替换训练主模块里面的reader，并增加ResNet18模型
trainmodule.reader = Dataset(
    train, val, mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0])


class Models(object):
    def __init__(self, modelinfo):
        for modelname in modelinfo:
            setattr(self, modelname, modelinfo[modelname])


trainmodule.models = Models({'ResNet18': resnet18.ResNet18})
trainmodule.model_list = ['ResNet18']


def trainmain():
    bigargv = [
        'train.py',
        "--model_save_dir=outputface",
        "--input_dtype=uint8",
        "--model=ResNet18",
        "--train_batch_size=512",
        "--test_batch_size=64",
        "--embedding_size=256",
        "--class_dim=80000",
        "--image_shape=3,112,112",
        "--lr=0.1",
        #"--lr_strategy=cosine_decay_with_warmup",
        "--lr_strategy=cosine_decay",
        #"--warmup_iter_num=12000",
        "--display_iter_step=10",
        "--total_iter_num=36000",
        "--test_iter_step=500",
        "--save_iter_step=6000",
        #"--loss_name=softmax",
        "--loss_name=arcmargin",
        "--arc_scale=64",
        "--arc_margin=0.5",
        "--train_datasetfile=dataset/face_ms1m/ms1m_train.data",
        "--train_labelfile=dataset/face_ms1m/ms1m_train_80000.label",
        "--val_datasetfile=dataset/face_ms1m/ms1m_train.data",
        "--val_labelfile=dataset/face_ms1m/ms1m_train_5164.label",
    ]

    bigargvkaibin = [
        'train.py',
        "--model_save_dir=outputface",
        "--input_dtype=uint8",
        "--model=ResNet18",
        "--train_batch_size=512",
        "--test_batch_size=64",
        "--embedding_size=512",
        "--class_dim=85164",
        "--image_shape=3,112,112",
        "--lr=0.1",
        "--lr_strategy=piecewise_decay",
        "--lr_steps=1000,2000,3000,4000,100000,140000,160000, 200000",
        "--lr_steps_values=0.01,0.05,0.1,0.5,1,0.1,0.01,0.001,0.0001",
        "--display_iter_step=10",
        "--total_iter_num=200000",
        "--test_iter_step=500",
        "--save_iter_step=5000",
        "--loss_name=arcmargin",
        "--arc_scale=64",
        "--arc_margin=0.3",
        "--train_datasetfile=dataset/face_ms1m/ms1m_train.data",
        "--train_labelfile=dataset/face_ms1m/ms1m_train.label",
        "--val_datasetfile=dataset/face_ms1m/ms1m_train.data",
        "--val_labelfile=dataset/face_ms1m/ms1m_train_5164.label",
    ]

    smallargv = [
        'train.py',
        #"--checkpoint=2400",
        #"--pretrained_model=softmaxface600",
        "--model_save_dir=outputface",
        "--use_gpu=true",
        "--input_dtype=uint8",
        "--model=ResNet18",
        "--train_batch_size=400",
        "--test_batch_size=64",
        "--embedding_size=256",
        "--class_dim=1000",
        "--image_shape=3,112,112",
        "--lr=0.1",
        #"--lr_strategy=cosine_decay_with_warmup",
        "--lr_strategy=cosine_decay",
        "--display_iter_step=1",
        "--total_iter_num=500",
        "--test_iter_step=100",
        "--save_iter_step=100",
        #"--loss_name=softmax",
        "--loss_name=arcmargin",
        "--arc_scale=64",
        "--arc_margin=0.5",
        "--train_datasetfile=dataset/face_ms1m_small/train.data",
        "--train_labelfile=dataset/face_ms1m_small/train.label",
        "--val_datasetfile=dataset/face_ms1m_small/train.data",
        "--val_labelfile=dataset/face_ms1m_small/train.label",

        #"--val_datasetfile=dataset/face_ms1m_small/test.data",
        #"--val_labelfile=dataset/face_ms1m_small/test.label",
    ]

    update_argv(bigargvkaibin)
    #update_argv(bigargv)
    trainmodule.main()


if __name__ == '__main__':
    trainmain()
