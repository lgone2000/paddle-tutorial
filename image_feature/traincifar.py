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


def train(args):
    def train_reader():
        #readerfunc = myreader.myreader_classify
        readerfunc = myreader.create_multiprocessreader(
            myreader.myreader_classify, 4)
        traindataset = readerfunc(args.train_datasetfile, args.train_labelfile,
                                  'train')
        for image, label in traindataset:
            yield image, label

    return train_reader


def val(args):
    def val_reader():
        traindataset = myreader.myreader_classify(args.val_datasetfile,
                                                  args.val_labelfile, 'val')
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
    train,
    val,
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255])


class Models(object):
    def __init__(self, modelinfo):
        for modelname in modelinfo:
            setattr(self, modelname, modelinfo[modelname])


trainmodule.models = Models({'ResNet18': resnet18.ResNet18})
trainmodule.model_list = ['ResNet18']


def trainmain():
    sys.argv = [
        'train.py',
        "--use_gpu=true",
        "--input_dtype=uint8",
        "--model=ResNet18",
        "--train_batch_size=512",
        "--test_batch_size=64",
        "--embedding_size=256",
        "--class_dim=10",
        "--image_shape=3,32,32",
        "--lr=0.1",
        "--lr_strategy=piecewise_decay",
        "--lr_steps=6000,12000,18000",
        #"--lr_epoch=30, 60, 90",
        #"--l2_decay=5e-4",
        "--display_iter_step=10",
        "--total_iter_num=20000",
        "--test_iter_step=500",
        "--save_iter_step=2000",
        "--loss_name=softmax",
        #"--pretrained_model=cifar_pretrained"
        "--train_datasetfile=dataset/cifar10/cifar10_train.data",
        "--train_labelfile=dataset/cifar10/cifar10_train.label",
        "--val_datasetfile=dataset/cifar10/cifar10_test.data",
        "--val_labelfile=dataset/cifar10/cifar10_test.label",
    ]
    trainmodule.main()


if __name__ == '__main__':
    trainmain()
