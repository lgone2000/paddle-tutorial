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
import random
from myreader import test_sample_label


def initrandom():
    random.seed(0)
    np.random.seed(0)
    fluid.default_startup_program().random_seed = 1000
    fluid.default_main_program().random_seed = 1000


train_datasetfile = 'dataset/samepatch_train/samepatch_train.data'
train_labelfile = 'dataset/samepatch_train/samepatch_train_500000.label'


def preprocess(img, mode):
    img = cv2.resize(img, (32, 32))
    return img.reshape((1, img.shape[0], img.shape[1]))


def test_multiprocess_reader():
    readerfunc = myreader.myreader_classify
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


class SimpleNet():
    def __init__(self):
        pass

    def net(self, input, embedding_size=256):
        conv = fluid.layers.conv2d(
            name='conv',
            input=input,
            num_filters=2,
            filter_size=3,
            stride=2,
            padding=1,
            groups=1,
            act='relu')
        output = fluid.layers.fc(input=conv, size=embedding_size)
        return output


def getvariable_value(prefix):
    variables = list(fluid.default_main_program().list_vars())
    vs = [v for v in variables if v.name.startswith(prefix)]
    for v in vs:
        print '#', v.name
    return vs


def debug_faceloss():
    initrandom()
    #debug loss NaN problem
    from losses import SoftmaxLoss
    from losses import ArcMarginLoss

    train_datasetfile = 'dataset/face_ms1m_small/train.data'
    train_labelfile = 'dataset/face_ms1m_small/train.label'
    traindataset = myreader.myreader_classify(
        train_datasetfile,
        train_labelfile,
        'train',
        iscolor=1,
        doshuffle=False,  #disable shuffle for debug
    )

    embedding_size = 256
    class_dim = 1000
    batchsize = 128
    imgshape = [3, 112, 112]
    model = resnet18.ResNet18()
    #model = SimpleNet()
    image = fluid.layers.data(name='image', shape=imgshape, dtype='uint8')
    labelvar = fluid.layers.data(name='label', shape=[1], dtype='int64')
    inputdata = trainmodule.preprocessimg(
        image, mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0])
    out = model.net(input=inputdata, embedding_size=embedding_size)

    metricloss = ArcMarginLoss(
        class_dim=class_dim, margin=0.5, scale=64, easy_margin=False)

    #metricloss = SoftmaxLoss(class_dim=class_dim)

    cost, logit = metricloss.loss(out, labelvar)
    avg_cost = fluid.layers.mean(x=cost)

    if os.path.exists('debugbatch.npz'):
        loaddata = np.load('debugbatch.npz')
        imgs, labels = loaddata['imgs'], loaddata['labels']
        print('img,label shape', imgs.shape, labels.shape, 'save mat sum',
              np.sum(imgs), np.sum(labels))
    else:
        imgs = []
        labels = []
        for img, label in traindataset:
            if label < class_dim:
                imgs.append(img.reshape([1] + imgshape))
                labels.append(label)
                if len(imgs) == batchsize:
                    break
        imgs = np.vstack(imgs)
        labels = np.array(labels, np.int64).reshape((-1, 1))
        np.savez('debugbatch.npz', imgs=imgs, labels=labels)
        print('save mat sum', np.sum(imgs), np.sum(labels))

    #place = fluid.CPUPlace()
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if not os.path.exists('debugmodel'):
        fluid.io.save_persistables(
            exe, 'debugmodel', main_program=fluid.default_main_program())
    else:
        fluid.io.load_persistables(
            exe, 'debugmodel', main_program=fluid.default_main_program())

    fcvars = getvariable_value('embedded.w_0')
    assert (len(fcvars) == 1)
    fcname = fcvars[0].name

    #optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    #opts = optimizer.minimize(avg_cost)

    fetch_list = [cost.name, inputdata.name, logit.name]
    for i in range(1):
        outputlist = exe.run(
            fluid.default_main_program(),
            feed={
                'image': imgs,
                'label': labels
            },
            fetch_list=fetch_list)

        for row in range(len(outputlist[0])):
            outsum = [(fetch_list[col], list(outputlist[col][row].flat[:3]))
                      for col in range(len(fetch_list))]
            out = outputlist[2][row]
            normout = np.sum(out * out)
            print row, outsum, normout, out.shape


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


def preparefc():

    #使用test_program 预测所有类别
    warm_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    warmup_reader = paddle.batch(
        reader.warmup(args), batch_size=args.test_batch_size, drop_last=False)

    #fc_class.w_0
    fc_class_name = 'fc_class.w_0'
    classfeas = {}
    classfeasnum = {}
    for batch_id, data in enumerate(warmup_reader()):
        test_outputlist = exe.run(
            test_prog, fetch_list=test_fetch_list, feed=warm_feeder.feed(data))
        label_vals = np.asarray([x[1] for x in data])
        fea_vals = test_outputlist[0]
        for label_val, fea_val in zip(label_vals, fea_vals):
            fea_val /= np.sum(fea_val * fea_val)**0.5
            if label_val not in classfeas:
                classfeas[label_val] = fea_val
                classfeasnum[label_val] = 1
            else:
                classfeas[label_val] = (
                    classfeas[label_val] * classfeasnum[label_val] +
                    fea_val) / (classfeasnum[label_val] + 1)
                classfeasnum[label_val] += 1

    assert (len(classfeas) == args.class_dim)

    v = fluid.global_scope().find_var(fc_class_name)
    w = v.get_tensor()
    newfcvalue = np.transpose(
        np.vstack([
            classfeas[i] / np.sum(classfeas[i] * classfeas[i])**0.5
            for i in range(args.class_dim)
        ]))
    print(np.array(w).shape)
    print(newfcvalue.shape, newfcvalue.dtype)
    #np.save('newfc.npy',newfcvalue)
    #np.save('oldfc.npy',np.array(w))
    #newfcvalue = np.load('oldfc.npy')
    w.set(newfcvalue, place)


def testsavevars():
    #print fluid.default_main_program()
    #     if pretrained_model:
    #         def if_exist(var):
    #             return os.path.exists(os.path.join(pretrained_model, var.name))
    #         fluid.io.load_vars(
    #             exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    #fluid.io.save_vars(executor=exe, dirname=outputfolder, main_program=fluid.default_main_program(),
    #               vars=None)
    pass


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr, 'usage: python train.py test_xx args...'
