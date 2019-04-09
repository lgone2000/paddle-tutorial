# -*- coding: utf-8 -*-
import paddle
import numpy as np
import cv2
import random
import struct
import os
import cPickle as pickle
#1.1 定义cifa reader，读取数据
from cifar_reader import loadcifar10

#各种数据集数据格式不同，有必要统一训练格式，简化读取, 另外大数据读取也比从文件读取快些
#数据集打包 1个数据文件 + 文件header + label
#数据文件打包所有数据： 格式为 keylen(4tyes)+key + valuelen(4types)+ value
#header为文本文件，记录文件名到图片文件偏移量的映射，格式为 key+tab+valueoffset+valuelen
#label为文本文件 key+tab+labeltext (分类格式)， metriclearning key1 \t key2 .. label
#cifar-10-python.tar.gz


class DataSetWriter(object):
    def __init__(self, prefix):
        self.datafile = open(prefix + '.data', 'wb')
        self.headerfile = open(prefix + '.header', 'wb')
        self.labelfile = open(prefix + '.label', 'wb')
        self.offset = 0

    def adddata(self, key, value):
        self.datafile.write(struct.pack('I', len(key)))
        self.datafile.write(key)
        self.datafile.write(struct.pack('I', len(value)))
        self.datafile.write(value)
        self.offset += 4 + len(key) + 4
        self.headerfile.write(key + '\t' + str(self.offset) + '\t' +
                              str(len(value)) + '\n')
        self.offset += len(value)

    def addlabel(self, label):
        self.labelfile.write(label + '\n')


#python formatconverter.py test_convert_cifar cifar-10-python.tar.gz data_batch cifar10/cifar10_train
#python formatconverter.py test_convert_cifar cifar-10-python.tar.gz test_batch cifar10/cifar10_test


def test_convert_cifar(filename, subname, outputprefix):
    writer = DataSetWriter(outputprefix)
    for i, record in enumerate(loadcifar10(filename, subname)):
        key = subname + '_' + str(i)
        img, label = record
        dummy, imgdata = cv2.imencode('.png', img)
        writer.adddata(key, imgdata)
        writer.addlabel(key + '\t' + str(label))


#
def test_convert_facems1m_train(datafolder, outputprefix):
    DATA_DIM = 112
    TRAIN_LIST = os.path.join(datafolder, 'label.txt')
    train_list = open(TRAIN_LIST, "r").readlines()
    train_image_list = []
    for i, item in enumerate(train_list):
        path, label = item.strip().split()
        label = int(label)
        train_image_list.append((path, label))
    print "train_data size:", len(train_image_list)
    writer = DataSetWriter(outputprefix)
    for i, record in enumerate(train_image_list):
        imgpath, label = record
        key = os.path.splitext(os.path.basename(imgpath))[0]
        value = open(os.path.join(datafolder, imgpath), 'rb').read()
        writer.adddata(key, value)
        writer.addlabel(key + '\t' + str(label))


def load_bin(path, image_size):
    bins, issame_list = pickle.load(open(path, 'rb'))
    data_list = []

    def datatostring(d):
        return d.tostring() if not isinstance(d, basestring) else d

    for i in xrange(len(issame_list)):
        img1 = datatostring(bins[i * 2])
        img2 = datatostring(bins[i * 2 + 1])
        yield i, img1, img2, issame_list[i]


#python formatconverter.py test_convert_face_test MS1M face_test
def test_convert_face_test(datafolder, outputprefix):
    writer = DataSetWriter(outputprefix)

    TEST_LIST = 'lfw,cfp_fp,agedb_30'
    imgsize = 112

    for datasetname in TEST_LIST.split(','):
        path = os.path.join(datafolder, datasetname + ".bin")
        if os.path.exists(path):
            data_set = load_bin(path, (imgsize, imgsize))
            count = 0
            for i, img1, img2, flag in data_set:
                key1 = datasetname + '_' + str(2 * i)
                key2 = datasetname + '_' + str(2 * i + 1)
                writer.adddata(key1, img1)
                writer.adddata(key2, img2)
                label = '\t'.join([key1, key2, str(int(flag))])
                writer.addlabel(label)
                count += 1
            print('finish', datasetname, count)


def convert_patchfolder(datafolder, writer, idstart, labelstart):
    def read_patch_file(fname, patch_w, patch_h):
        img = cv2.imread(fname, 0)
        height, width = img.shape[:2]
        assert (height == 1024 and width == 1024)
        assert ((height % patch_h == 0) and (width % patch_w == 0))
        for y in range(0, height, patch_h):
            for x in range(0, width, patch_w):
                patch = img[y:y + patch_h, x:x + patch_w].copy()
                if (patch.mean() != 0) and (patch.astype(np.float32).std() >
                                            1e-2):
                    yield patch

    labels = np.fromfile(
        os.path.join(datafolder, 'info.txt'), np.int32, sep=' ').reshape((-1,
                                                                          2))
    offset = 0

    for i in range(10000):
        bmpfile = os.path.join(datafolder, 'patches%4.4d.bmp' % (i))
        #print bmpfile
        if not os.path.exists(bmpfile):
            print('total image number is %s' % i)
            break

        for patch in read_patch_file(bmpfile, 64, 64):
            label = int(labels[offset][0])
            offset += 1
            key = str(idstart + offset)
            dummy, imgdata = cv2.imencode('.png', patch)
            writer.adddata(key, imgdata)
            writer.addlabel(key + '\t' + str(labelstart + label))
        if i % 100 == 0:
            print '#',
    print(datafolder, 'totalimg', offset, 'totallabel', label + 1)
    assert (offset == len(labels))
    maxlabel = labelstart + label + 1
    return idstart + offset, maxlabel


def test_convert_patchfolders(outputprefix, *folders):
    writer = DataSetWriter(outputprefix)
    idstart, labelstart = 0, 0
    for folder in folders:
        idstart, labelstart = convert_patchfolder(folder, writer, idstart,
                                                  labelstart)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr, 'usage: python train.py test_xx args...'
