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
#替换老的reader

train_datasetfile = 'dataset/samepatch_train/samepatch_train.data'
train_labelfile = 'dataset/samepatch_train/samepatch_train_500000.label'

val_datasetfile = 'dataset/samepatch_train/samepatch_train.data'
val_labelfile = 'dataset/samepatch_train/samepatch_test_44803.label'


def preprocess(img, mode):
    img = cv2.resize(img, (32, 32))
    return img.reshape((1, img.shape[0], img.shape[1]))


def train(args):
    def train_reader():
        traindataset = myreader.myreader_classify_multiprocess(
            train_datasetfile,
            train_labelfile,
            'train',
            iscolor=0,
            preprocessfunc=preprocess)
        for image, label in traindataset:
            yield image, label

    return train_reader


def val(args):
    def val_reader():
        traindataset = myreader.myreader_classify(
            val_datasetfile,
            val_labelfile,
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

def testforward():
    #debug loss NaN problem
    from losses import SoftmaxLoss
    from losses import ArcMarginLoss
    traindataset = myreader.myreader_classify(
            train_datasetfile,
            train_labelfile,
            'train',
            iscolor=0,
            preprocessfunc=preprocess)
    
    embedding_size = 64
    class_dim = 500000
    batchsize = 512
    model = l2net.L2Net()
    #model = resnet18.ResNet18()
    image = fluid.layers.data(name='image', shape=[1,32,32], dtype='uint8')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    inputdata = trainmodule.preprocessimg(image)
    out = model.net(input=inputdata, embedding_size=embedding_size)
    #metricloss = ArcMarginLoss(class_dim = class_dim,margin = 0.5,scale = 64,easy_margin = False)
    
    metricloss = SoftmaxLoss(class_dim=class_dim)
    
    cost, logit = metricloss.loss(out, label)
    avg_cost = fluid.layers.mean(x=cost)
    
    imgs = []
    labels = []
    for img,label in traindataset:
        imgs.append(img.reshape(1,1,32,32))
        labels.append(label)
        if len(imgs) == batchsize:
            break
    imgs = np.vstack(imgs)
    labels = np.array(labels, np.int64).reshape((-1,1))
    
    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    opts = optimizer.minimize(avg_cost)
   
    #lace = fluid.CPUPlace()
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
        
    print('avg_cost.name', avg_cost.name)
    outputlist = exe.run(
        fluid.default_main_program(),
        feed={'image': imgs,'label':labels},
        fetch_list=[avg_cost.name])
    
    print 'outputlist', outputlist
    

def trainmain():
    sys.argv = [
        'train.py',
        #"--use_gpu=false",
        "--input_dtype=uint8",
        "--model=L2Net",
        "--train_batch_size=512",
        "--test_batch_size=64",
        "--embedding_size=64",
        "--class_dim=500000",
        "--image_shape=1,32,32",
        "--lr=0.1",
        #"--lr_strategy=piecewise_decay",
        #"--lr_steps=100000, 140000, 160000",
        #"--lr_epoch=30, 60, 90",
        #"--l2_decay=5e-4",
        "--lr_strategy=cosine_decay_with_warmup",
        "--warmup_iter_num=6000",
        "--display_iter_step=5",
        "--total_iter_num=18000",
        "--test_iter_step=500",
        "--save_iter_step=3000",
        "--loss_name=arcmargin",
        "--arc_scale=64",
        "--arc_margin=0.5",
    ]
    trainmodule.main()


if __name__ == '__main__':
    trainmain()
    #testforward()
