# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0, 'metric_learning')
import paddle
import numpy as np
import cv2
import random
import paddle.fluid as fluid
import resnet18
#import resnet_arcface

from fluidpreprocess import preprocessimg
import face_verification
from myreader import ImageData, flip, convert2rgb, swapaxis, loadimagefromstr, preprocess
import face_verification


#python inferface.py test_convertsnap2inference facepretrainmodel/36000/ faceinference
def test_convertsnap2inference(loadmodel,
                               outputfolder,
                               inputshape='3,112,112',
                               embedding_size=512):
    embedding_size = int(embedding_size)
    inputshape = [int(x) for x in inputshape.split(',')]
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    image = fluid.layers.data(name='image', shape=inputshape, dtype='float32')
    inputdata = preprocessimg(
        image, mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0])
    #out = resnet_arcface.ResNet_ARCFACE().net(input=inputdata, embedding_size=embedding_size)

    out = resnet18.ResNet().net(input=inputdata, embedding_size=embedding_size)
    #载入预训练模型
    fluid.io.load_persistables(
        exe, loadmodel, main_program=fluid.default_main_program())
    print('outputname:', out.name)

    test_program = fluid.default_main_program().clone(for_test=True)

    fluid.io.save_inference_model(
        dirname=outputfolder,
        feeded_var_names=[image.name],
        target_vars=[out],
        executor=exe,
        main_program=test_program,
        model_filename='model',
        params_filename='params')


def test_forward():
    modelfolder = 'faceinference'
    input_shape = [1, 3, 112, 112]

    img = cv2.imread('testdata/infer_face.jpg')
    inputdata = np.swapaxes(img, 1, 2)
    inputdata = np.swapaxes(inputdata, 1, 0).reshape(input_shape)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_names, fetch_targets = fluid.io.load_inference_model(
        modelfolder, exe, 'model', 'params')

    feed_shapes = [input_shape]
    fetch_list_name = [a.name for a in fetch_targets]
    print('feedname:', feed_names, 'feed_shapes:', feed_shapes,
          'fetch_list_name:', fetch_list_name)
    result = exe.run(
        program, fetch_list=fetch_targets, feed={'image': inputdata})
    print(result)


def load_faceeval_label(labelfile):
    data_list = []
    issame_list = []

    for line in open(labelfile, 'rb'):
        key1, key2, label = line.split('\t')
        label = int(label)
        data_list.append((key1, key2))
        issame_list.append(label)
    return data_list, issame_list


def myreader_evalface(datasetfile, labeldata, batchsize):
    allimagedata = ImageData(datasetfile)
    batches = []

    for key1, key2 in labeldata:
        for flip in range(2):
            img1 = loadimagefromstr(allimagedata.getvalue(key1), 1)
            img2 = loadimagefromstr(allimagedata.getvalue(key2), 1)
            ops = [[convert2rgb],
                   [swapaxis]] if flip == 0 else [[flip], [convert2rgb],
                                                  [swapaxis]]
            data1 = preprocess(img1, ops)
            data2 = preprocess(img2, ops)
            batches.append(data1)
            batches.append(data2)
            if len(batches) == batchsize:
                yield np.vstack(batches).reshape(
                    (len(batches), data1.shape[0], data1.shape[1],
                     data1.shape[2]))
                batches = []
            break
    if batches:
        np.vstack(batches).reshape((len(batches), data1.shape[0],
                                    data1.shape[1], data1.shape[2]))


def eval_face(facemodelpath, testnames='cfp_fp,agedb_30,lfw'):
    input_shape = [1, 3, 112, 112]
    place = fluid.CUDAPlace(0)
    #place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    test_program, feed_names, fetch_list_test = fluid.io.load_inference_model(
        facemodelpath, exe, 'model', 'params')
    test_list = testnames.split(',')
    batchsize = 128
    for testname in test_list:
        test_datasetfile = 'dataset/face_ms1m/face_test.data'
        test_labelfile = 'dataset/face_ms1m/face_test_%s.label' % (testname)
        data_list, issame_list = load_faceeval_label(test_labelfile)
        all_embeddings = []
        for batch in myreader_evalface(test_datasetfile, data_list, batchsize):
            [embeddings] = exe.run(
                test_program,
                fetch_list=fetch_list_test,
                feed={'image': batch})
            #print('embeddings', embeddings.shape, (embeddings * embeddings).sum(axis=1).shape)
            embeddings /= np.sqrt(
                (embeddings * embeddings).sum(axis=1)).reshape((-1, 1))
            all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        _, _, accuracy, val, val_std, far = face_verification.evaluate(
            all_embeddings, issame_list, nrof_folds=10)
        acc, std = np.mean(accuracy), np.std(accuracy)
        print('[%s]Accuracy-Flip: %1.5f+-%1.5f' % (testname, acc, std))
        sys.stdout.flush()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr, 'usage: python train.py test_xx args...'
