# -*- coding: utf-8 -*-
import six
import tarfile
import cPickle as pickle
import numpy as np
import cv2


#数据集，可以从 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz 下载
def loadcifar10(filename, sub_name):
    alldatas = []
    with tarfile.open(filename, mode='r') as f:
        names = (each_item.name for each_item in f
                 if sub_name in each_item.name)
        for name in names:
            batch = pickle.load(f.extractfile(name))
            data = batch[six.b('data')]
            labels = batch.get(
                six.b('labels'), batch.get(six.b('fine_labels'), None))
            assert labels is not None
            for sample, label in six.moves.zip(data, labels):
                image = np.transpose(sample.reshape([3, 32, 32]), [1, 2, 0])
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                alldatas.append((image, int(label)))
    return alldatas


def cifareader(filename, sub_name, cycle=False):
    '''修改自 lib/python2.7/site-packages/paddle/dataset/cifar.py
       返回 [32,32,3] bgr image ,与 cv2.imread 读取一个图片格式相同
    '''
    #读入全部数据后，进行迭代
    print 'loading cifer10 ==='
    loadcifar10(filename, sub_name)
    print 'finish loading cifer10'
    index = list(range(len(alldatas)))
    while True:
        for i in index:
            yield alldatas[i]
        if not cycle:
            break
        random.shuffle(index)
