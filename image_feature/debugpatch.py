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


def initrandom():
    np.random.seed(0)
    fluid.default_startup_program().random_seed = 1000
    fluid.default_main_program().random_seed = 1000

train_datasetfile = 'dataset/samepatch_train/samepatch_train.data'
train_labelfile = 'dataset/samepatch_train/samepatch_train_500000.label'
def preprocess(img, mode):
    img = cv2.resize(img, (32, 32))
    return img.reshape((1, img.shape[0], img.shape[1]))

def test_multiprocess_reader():
    readerfunc =  myreader.myreader_classify
    #readerfunc = myreader.create_multiprocessreader(myreader.myreader_classify, 4)

    traindataset = readerfunc(
        train_datasetfile,
        train_labelfile,
        'train',
        iscolor=0,
        preprocessfunc=preprocess)
    for i in range(10):
        image, label = traindataset.next()
        print image.shape, label
        
def debug_patchloss():
    #debug loss NaN problem
    from losses import SoftmaxLoss
    from losses import ArcMarginLoss
    traindataset = myreader.myreader_classify(
        train_datasetfile,
        train_labelfile,
        'train',
        iscolor=0,
        doshuffle=True,
        preprocessfunc=preprocess)

    embedding_size = 64
    class_dim = 50000
    batchsize = 128
    model = l2net.L2Net()
    #model = resnet18.ResNet18()
    image = fluid.layers.data(name='image', shape=[1, 32, 32], dtype='uint8')
    labelvar = fluid.layers.data(name='label', shape=[1], dtype='int64')
    inputdata = trainmodule.preprocessimg(image)
    out = model.net(input=inputdata, embedding_size=embedding_size)
    metricloss = ArcMarginLoss(
        class_dim=class_dim, margin=0.5, scale=64, easy_margin=False)

    #metricloss = SoftmaxLoss(class_dim=class_dim)

    cost, logit = metricloss.loss(out, labelvar)
    avg_cost = fluid.layers.mean(x=cost)

    imgs = []
    labels = []
    for img, label in traindataset:
        if label < class_dim:
            imgs.append(img.reshape(1, 1, 32, 32))
            labels.append(label)
            if len(imgs) == batchsize:
                break
    imgs = np.vstack(imgs)
    labels = np.array(labels, np.int64).reshape((-1, 1))

    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    opts = optimizer.minimize(avg_cost)

    place = fluid.CPUPlace()
    #place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    loss_value, output_value, label_value = exe.run(
        fluid.default_main_program(),
        feed={
            'image': imgs,
            'label': labels
        },
        fetch_list=[cost.name, out.name, labelvar.name])

    for x, y, z in zip(loss_value, output_value, label_value):
        print list(x.flat), np.sum(y.flat), list(z.flat)


def test_reader():
    import paddle
    from paddle import fluid
    train_batch_size = 10

    def getreader():
        reader = myreader_classify('dataset/cifar10/cifar10_train.data',
                                   'dataset/cifar10/cifar10_train.label',
                                   'train')
        return reader

    reader = paddle.batch(
        getreader, batch_size=train_batch_size, drop_last=True)

    if 0:
        imagevar = fluid.layers.data(
            name='image', shape=[3, 32, 32], dtype='float32')
        labelvar = fluid.layers.data(name='label', shape=[1], dtype='int64')
        #根据变量类型，进行自动转换
        test_feeder = fluid.DataFeeder(
            place=fluid.CPUPlace(), feed_list=[imagevar, labelvar])
        for batch in reader():
            print 'batchsize', len(batch)
            for data, label in batch:
                print('datashape', data.shape, 'label', label)

            feeddata = test_feeder.feed(batch)
            print('feeddata', feeddata)
            break

    pyreader = fluid.layers.py_reader(
        capacity=64,
        shapes=[(-1, 3, 33, 33), (-1, 1)],
        dtypes=['float32', 'int64'])
    pyreader.decorate_paddle_reader(reader)

    img, label = fluid.layers.read_file(pyreader)
    loss = fluid.layers.mean(img)
    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    opts = optimizer.minimize(loss)

    fluid.Executor(fluid.CUDAPlace(0)).run(fluid.default_startup_program())
    exe = fluid.ParallelExecutor(use_cuda=True, loss_name=loss.name)
    pyreader.start()
    for i in range(10):
        try:
            outputlist = exe.run(fetch_list=[loss.name])
            print outputlist
        except fluid.core.EOFException as e:
            reader.reset()


def test_reader1():
    train_datasetfile = 'dataset/face_ms1m/ms1m_train.data'
    train_labelfile = 'dataset/face_ms1m/ms1m_train_80000.label'

    val_datasetfile = 'dataset/face_ms1m/ms1m_train.data'
    val_labelfile = 'dataset/face_ms1m/ms1m_train_5164.label'
    traindataset = myreader_classify(
        train_datasetfile, train_labelfile, 'train', doshuffle=False)
    for img, label in traindataset:
        break
        
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr, 'usage: python train.py test_xx args...'
