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

#替换老的reader

train_datasetfile = 'dataset/face_ms1m/ms1m_train.data'
train_labelfile = 'dataset/face_ms1m/ms1m_train_80000.label'

val_datasetfile = 'dataset/face_ms1m/ms1m_train.data'
val_labelfile = 'dataset/face_ms1m/ms1m_train_5164.label'


def train(args):
    def train_reader():
        traindataset = myreader.myreader_classify_multiprocess(
            train_datasetfile, train_labelfile, 'train')
        for image, label in traindataset:
            yield image, label

    return train_reader


def val(args):
    def val_reader():
        traindataset = myreader.myreader_classify(
            val_datasetfile, val_labelfile, 'val', doshuffle=False)
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
trainmodule.reader = Dataset(
    train, val, mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0])


class Models(object):
    def __init__(self, modelinfo):
        for modelname in modelinfo:
            setattr(self, modelname, modelinfo[modelname])


trainmodule.models = Models({'ResNet18': resnet18.ResNet18})
trainmodule.model_list = ['ResNet18']


def trainmain():
    sys.argv = [
        'train.py',
        "--input_dtype=uint8",
        "--model=ResNet18",
        "--train_batch_size=512",
        "--test_batch_size=64",
        "--embedding_size=256",
        "--class_dim=80000",
        "--image_shape=3,112,112",
        "--lr=0.1",
        #"--lr_strategy=piecewise_decay",
        #"--lr_steps=100000, 140000, 160000",
        #"--lr_epoch=30, 60, 90",
        #"--l2_decay=5e-4",
        "--lr_strategy=cosine_decay_with_warmup",
        "--warmup_iter_num=12000",
        "--display_iter_step=10",
        "--total_iter_num=36000",
        "--test_iter_step=500",
        "--save_iter_step=6000",
        "--loss_name=arcmargin",
        "--arc_scale=64",
        "--arc_margin=0.5",
    ]
    trainmodule.main()


if __name__ == '__main__':
    trainmain()
