# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import sys
import os
import mmap
import random
import math
from paddle import fluid
import time
import signal


def loadimagefromstr(imagestr, iscolor):
    imgdata = np.fromstring(imagestr, dtype='uint8')
    image = cv2.imdecode(imgdata, iscolor)
    if image is None:
        return None
    return image


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


def grouplabels(labeldata, labelset):
    datadict = {label: [] for label in labelset}
    for data in labeldata:
        key, label = data
        datadict[label].append(data)

    return datadict


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


def myreader_metric(datasetfile,
                    labelfile,
                    mode,
                    doshuffle=True,
                    iscolor=1,
                    preprocessfunc=None):
    allimagedata = ImageData(datasetfile)
    labeldatas, labelset = loadlabeldata(labelfile)

    #grouped_labeldata = #()

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


#仅用于训练, 使用多个子进程读取数据, 不能退出
def create_multiprocessreader(reader_createor, threadnum):
    import multiprocessing as mp
    import Queue
    import threading

    def term(sig_num, addtion):
        print(
            'term reader child process %s, parent:%s!!!!!!!!!!!!!' %
            (os.getpid(), os.getppid()),
            file=sys.stderr)
        #force to quit
        os._exit(-1)

    def subprocess_reader(pipe, args, argk):
        #make sure when main process killed , subprocess receive term signal
        signal.signal(signal.SIGTERM, term)
        try:
            import ctypes
            libc = ctypes.CDLL('libc.so.6')
            libc.prctl(1, 15)
        except:
            pass

        #子进程 读入数据，并发到管道中
        output_p, input_p = pipe
        output_p.close()
        datareader = reader_createor(*args, **argk)
        for x in datareader:
            input_p.send(x)

    def fetch_thread_main(queue, args, argk):
        #启动子进程，并从子进程读入数据，再写入队列queue
        output_p, input_p = pipe = mp.Pipe()
        #start subprocess for read
        writer_p = mp.Process(
            target=subprocess_reader, args=(pipe, args, argk))
        #子进程设为守护进程，父进程退出，子进程也退出， https://blog.csdn.net/wenzhou1219/article/details/81320622
        writer_p.daemon = True
        writer_p.start()
        input_p.close()
        while True:
            try:
                queue.put(output_p.recv())
            except EOFError:
                break

    def reader(*args, **argk):
        queue = Queue.Queue(60)
        threads = []
        for i in range(threadnum):
            thread = threading.Thread(
                target=fetch_thread_main, args=(queue, args, argk))
            thread.setDaemon(True)
            thread.start()

        while True:
            yield queue.get()

    return reader


def test_sample_label(labelfile, sampleperlabel):
    sampleperlabel = int(sampleperlabel)
    labeldatas, labelset = loadlabeldata(labelfile)
    datadict = grouplabels(labeldatas, labelset)
    for label in sorted(list(labelset)):
        dd = datadict[label]
        random.shuffle(dd)
        datas = dd[:sampleperlabel]
        for key, label in datas:
            print(key + '\t' + str(label))
