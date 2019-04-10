# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import mmap
import random
import math
from paddle import fluid
import time


def loadimagefromstr(imagestr, iscolor):
    imgdata = np.fromstring(imagestr, dtype='uint8')
    image = cv2.imdecode(imgdata, iscolor)
    if image is None:
        return None
    return image


def convert2png(imgdata):
    img_ori = Image.open(StringIO(imgdata))
    img = img.convert('BGR')
    dummy, imgdata = cv2.imencode('.png', img)
    return imgdata


class ImageData(object):
    def __init__(self, datafile):
        headerfile = os.path.splitext(datafile)[0] + '.header'
        self.offsetdict = {}
        for line in open(headerfile, 'rb'):
            key, val_pos, val_len = line.split('\t')
            self.offsetdict[key] = (int(val_pos), int(val_len))
        self.fp = open(datafile, 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)

    def getvalue(self, key):
        p = self.offsetdict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    def getkeys(self):
        return self.offsetdict.keys()


def flip(img):
    if random.randint(0, 1) == 1:
        return cv2.flip(img, 1)
    return img


def swapaxis(img):
    img = np.swapaxes(img, 1, 2)
    img = np.swapaxes(img, 1, 0)
    return img


def normalize(img, mean, std):
    img_mean = np.array(mean, np.float32).reshape((3, 1, 1))
    img_std = np.array(std, np.float32).reshape((3, 1, 1))
    return (img.astype(np.float32, False) - img_mean) / img_std


def convert2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(img, operators):
    for op in operators:
        func, args = op[0], op[1:]
        img = func(img, *args)
    return img


def process_image(img, mode):

    if mode == 'train':
        preprocess_ops = [
            [flip],
            [convert2rgb],
            [swapaxis],
            #[normalize, mean, std],
        ]
    else:
        preprocess_ops = [
            [convert2rgb],
            [swapaxis],
            #[normalize, mean, std],
        ]
    return preprocess(img, preprocess_ops)


def loadlabeldata(labelfile):
    labeldatas = []
    #读入标注
    labelset = set()
    for line in open(labelfile, 'rb'):
        key, label = line.split('\t')
        label = int(label)
        labeldatas.append((key, label))
        if label not in labelset:
            labelset.add(label)
    return labeldatas, labelset


def myreader_classify(datasetfile,
                      labelfile,
                      mode,
                      doshuffle=True,
                      iscolor=1,
                      preprocessfunc=None):
    allimagedata = ImageData(datasetfile)
    labeldatas, labelset = loadlabeldata(labelfile)
    if mode == 'train':
        #保证所有标签都有样本
        labels = sorted(list(labelset))
        assert (len(labels) == labels[-1] + 1)
    if preprocessfunc is None:
        preprocessfunc = process_image
    while True:
        if doshuffle:
            random.shuffle(labeldatas)
        for key, label in labeldatas:
            imgdata = allimagedata.getvalue(key)
            img = loadimagefromstr(imgdata, iscolor)
            assert (img is not None)
            img = preprocessfunc(img, mode)
            yield img, label
        #如果是训练就循环读取，测试只读取一遍
        if mode != 'train':
            break


#仅用于训练
def myreader_classify_multiprocess(datasetfile,
                                   labelfile,
                                   mode,
                                   doshuffle=True,
                                   iscolor=1,
                                   preprocessfunc=None,
                                   threadnum=4):
    import multiprocessing as mp
    import Queue
    import threading

    def mp_writer(classname, configfile, option, pipe):
        #子进程 读入数据，并发到管道中
        output_p, input_p = pipe
        output_p.close()
        datareader = myreader_classify(datasetfile, labelfile, mode, doshuffle,
                                       iscolor, preprocessfunc)
        for x in datareader:
            input_p.send(x)

    def mp_reader(queue):
        #启动子进程，并从子进程读入数据，再写入队列queue
        output_p, input_p = pipe = mp.Pipe()
        #start subprocess for read
        writer_p = mp.Process(
            target=mp_writer, args=(datasetfile, labelfile, mode, pipe))
        writer_p.start()

        input_p.close()
        while True:
            try:
                bloblist = output_p.recv()
                queue.put(bloblist)
            except EOFError:
                break

    queue = Queue.Queue(60)
    threads = []
    for i in range(threadnum):
        thread = threading.Thread(target=mp_reader, args=(queue, ))
        thread.setDaemon(True)
        thread.start()

    while True:
        yield queue.get()


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
