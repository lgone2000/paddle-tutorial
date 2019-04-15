# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, 'metric_learning')
import paddle
import numpy as np
import cv2
import random
import paddle.fluid as fluid
import resnet18
from fluidpreprocess import preprocessimg

#python inferface.py test_convertsnap2inference facepretrainmodel/36000/ faceinference
def test_convertsnap2inference(loadmodel, outputfolder,
                               inputshape='3,112,112'):
    embedding_size = 256
    inputshape = [int(x) for x in inputshape.split(',')]
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    image = fluid.layers.data(name='image', shape=inputshape, dtype='float32')
    inputdata = preprocessimg(
        image, meanvalue=[127.5, 127.5, 127.5], stdvalue=[128.0, 128.0, 128.0])
    model = resnet18.ResNet()
    out = model.net(input=inputdata, embedding_size=embedding_size)
    #载入预训练模型
    fluid.io.load_persistables(
        exe, loadmodel, main_program=fluid.default_main_program())
    print('outputname:', out.name)
    #print fluid.default_main_program()
    #     if pretrained_model:
    #         def if_exist(var):
    #             return os.path.exists(os.path.join(pretrained_model, var.name))
    #         fluid.io.load_vars(
    #             exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    #fluid.io.save_vars(executor=exe, dirname=outputfolder, main_program=fluid.default_main_program(),
    #               vars=None)

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
    print 'feedname:', feed_names, 'feed_shapes:', feed_shapes, 'fetch_list_name:', fetch_list_name
    result = exe.run(
        program, fetch_list=fetch_targets, feed={'image': inputdata})
    print result


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr, 'usage: python train.py test_xx args...'
