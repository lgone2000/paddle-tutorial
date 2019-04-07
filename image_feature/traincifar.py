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

train_datasetfile = 'cifar10/cifar10_train.data'
train_labelfile = 'cifar10/cifar10_train.label'

val_datasetfile = 'cifar10/cifar10_test.data'
val_labelfile = 'cifar10/cifar10_test.label'


def train(args):
    def train_reader():
        traindataset = myreader.myreader_classify(train_datasetfile,
                                                  train_labelfile, 'train')
        for image, label in enumerate(traindataset):
            yield image, label

    return train_reader


def val(args):
    def val_reader():
        traindataset = myreader.myreader_classify(val_datasetfile,
                                                  val_labelfile, 'val')
        for image, label in enumerate(traindataset):
            yield (image, label)

    return val_reader


class Dataset(object):
    def __init__(self, train, val):
        self.train = train
        self.val = val
        self.test = val


#替换训练主模块里面的reader，并增加ResNet18模型
trainmodule.reader = Dataset(train, val)


class Models(object):
    def __init__(self, modelinfo):
        for modelname in modelinfo:
            setattr(self, modelname, modelinfo[modelname])


trainmodule.models = Models({'ResNet18': resnet18.ResNet18})
trainmodule.model_list = ['ResNet18']


def trainmain():
    sys.argv = [
        'train.py',
        "--model=ResNet18",
        "--train_batch_size=100",
        "--test_batch_size=50",
        "--embedding_size=256",
        "--class_dim=10",
        "--image_shape=3,32,32",
        "--lr=0.1",
        "--lr_strategy=piecewise_decay",
        "--lr_steps=6000000,9000000,12000000",
        #"--l2_decay=5e-4",
        "--display_iter_step=100",
        "--total_iter_num=16000000",
        "--test_iter_step=100000",
        "--save_iter_step=1000000",
        "--loss_name=softmax",
        #120,180, 240
    ]
    trainmodule.main()


if __name__ == '__main__':
    trainmain()
