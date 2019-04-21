# -*- coding: utf-8 -*-
import paddle
import numpy as np
import cv2
import random
import paddle.fluid as fluid

#1.1 定义cifa reader，读取数据
from cifar_reader import cifareader


#1.2 测试读取10个图片和label 并保存
def test_reader(readnum):
    #测试读取程序，验证
    traindataset = cifareader('dataset/cifar-10-python.tar.gz', 'data_batch')
    #testdataset = reader('cifar-10-python.tar.gz', 'test_batch')
    readnum = int(readnum)
    for i, data in enumerate(traindataset):
        image, label = data
        #这里 image是 shape = [32,32,3] , dtype='int8'
        filename = 'temp/%d.png' % i
        cv2.imwrite(filename, image)
        if i >= readnum:
            break


class ResNet():
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
        output = fluid.layers.fc(input=conv, size=classnum)


#1.3 构建一个简单的分类网络 input->conv->relu->fc
def classifynetwork():
    classnum = 10
    input = fluid.layers.data(name='input', shape=[3, 32, 32], dtype='float32')
    #参考文档 http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#conv2d
    conv = fluid.layers.conv2d(
        name='conv',
        input=input,
        num_filters=2,
        filter_size=3,
        stride=2,
        padding=1,
        groups=1,
        act='relu')
    output = fluid.layers.fc(input=conv, size=classnum)
    print 'conv output name', conv.name, 'fc output name', output.name

    return output


#1.4 查看网络中的变量名称和类型
def print_variable():
    variables = list(fluid.default_main_program().list_vars())
    #变量类型请参考
    #http://paddlepaddle.org/documentation/docs/zh/1.3/user_guides/howto/training/save_load_variables.html#permalink-1--
    for var in variables:
        print 'var', var.name, var.persistable

    all_params = fluid.default_main_program().global_block().all_parameters()
    for var in all_params:
        print 'param', var.name, var.persistable


#1.5 建立运行环境executer 并进行一次forward预测
def test_forward():
    """测试网络forward预测"""
    #组网
    output = classifynetwork()

    #创建一个和input变量一样大小的 numpy array, batch大小为1
    data = np.zeros((1, 3, 32, 32), np.float32)
    #创建网络运行环境（在CPU上运行）
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    #执行网络参数随机初始化操作，需要首先调用
    exe.run(fluid.default_startup_program())

    #打印网络中的所有变量
    print_variable()
    #会输出这些变量
    #input : 输入变量
    # 模型变量（网络参数）
    #conv.w_0 :卷积系数
    #conv.b_0 :卷积bias
    #fc_0.w_0 :fc层系数
    #fc_0.b_0 :fc层bias

    #每层输出临时变量(临时变量在保存网络参数时不会被保存)
    #conv.tmp_0 ：卷积层输出变量
    #conv.tmp_1 ：卷积层输出变量（加bias后+relu后）, 定义conv层可以通过 bias_attr=False 去掉bias
    #fc_0.tmp_0 ：fc层输出变量
    #fc_0.tmp_1 ：fc层输出变量（加bias后），也就是output

    #执行网络预测操作， 执行后，返回fetch_list 里面指定变量的值（转换为numpy array）
    outputlist = exe.run(
        fluid.default_main_program(),
        #传入图像数据
        feed={'input': data},
        #指定输出那个变量
        fetch_list=['fc_0.tmp_1', 'conv.w_0'])
    fc_output_tensor, conv_weight_tensor = outputlist
    print 'output', fc_output_tensor
    #conv_weight_tensor shape is [filternum, inputfilterchannels, filtersize, filtersize]
    print 'conv weight', conv_weight_tensor


#1.6 图像预处理，将图像处理为网络输入的格式
def preprocess(image):
    data = np.transpose(image, [2, 0, 1]).reshape([1, 3, 32, 32])
    data = data.astype(np.float32) / 255.0
    return data


#1.7 batch reader, 将输入转换为batch
def createbatchreader(subname, batchsize):
    images, labels = [], []
    traindataset = cifareader('dataset/cifar-10-python.tar.gz', subname, True)
    while True:
        for i in range(batchsize):
            image, label = traindataset.next()
            images.append(preprocess(image))
            labels.append(label)

        labelinput = np.array(labels, np.int64).reshape((-1, 1))
        imageinput = np.vstack(images)
        yield imageinput, labelinput


#1.8 定义损失函数，并执行一次训练迭代（单batch随机梯度下降优化)


def test_backward():
    #组正向网络预测
    output = classifynetwork()
    #定义label变量
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    #通过比较output 和label 计算损失
    #参考 http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/layers_cn.html#softmax-with-cross-entropy

    loss = fluid.layers.softmax_with_cross_entropy(logits=output, label=label)
    #对batch内损失求和， 得到1个值
    lossmean = fluid.layers.mean(loss)

    #识别结果top1
    resultscore, resultlabel = fluid.layers.topk(
        name='outputlabel', input=output, k=1)
    #识别准确率
    acc_top1 = fluid.layers.accuracy(input=output, label=label, k=1)

    #选择优化算法，这里使用SGD优化器， 定义了如何使用梯度更新参数
    optimizer = fluid.optimizer.SGD(learning_rate=0.1)

    #自动添加反向传播op
    opts = optimizer.minimize(lossmean)
    print_variable()
    #会增加不少以@GRAD结尾的梯度变量conv.w_0@GRAD

    #读入一个minibatch， 大小是[10, 3, 32, 32]
    batchreader = createbatchreader(subname='data_batch', batchsize=20)
    imageinput, labelinput = batchreader.next()

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())

    # 取出损失和一些内部变量，比如conv.w_0的梯度
    fetch_list = [
        lossmean.name, loss.name, 'conv.w_0@GRAD', acc_top1.name,
        resultlabel.name
    ]
    #运行网络：对这个minibatch执行多次次优化迭代。
    #对同一个batch进行优化，应该看到损失下降和准确率上升, 可以换成 for imageinput, labelinput in batchreader:  训练全部样本
    print '==== training'
    for i in range(100):
        outputlist = exe.run(
            fluid.default_main_program(),
            feed={
                'input': imageinput,
                'label': labelinput
            },
            fetch_list=fetch_list)

        lossmean_value, loss_value, convwgrad_value, accuracy_value, resultlabel_value = outputlist
        #print 'lossmean', lossmean_value, 'loss', loss_value, 'convwgrad', convwgrad_value, 'accuracy',accuracy_value, 'resultlabel', resultlabel_value
        print 'train iter', i, 'loss', lossmean_value, 'accuray', accuracy_value
    #打印一下每个图像的预测标签和损失
    print '==== batch predict result'
    for i in range(len(loss_value)):
        print i, 'loss', loss_value[i], 'label', labelinput[i][
            0], 'predictlabel', resultlabel_value[i][0]


#1.9 保存训练模型并预测
def test_predict():
    test_backward()
    #clone一份网络用于预测（其中会去除反向部分，并fix BN层参数）
    test_program = fluid.default_main_program().clone(for_test=True)
    modelfolder = 'model'

    #得到输出的变量，保存模型时需要指定输出变量（不输出的变量保存时会被优化）
    outputvarname = 'outputlabel.tmp_1'
    outputvar = [
        var for var in fluid.default_main_program().list_vars()
        if var.name == outputvarname
    ][0]
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    #将网络结构和参数保存下来，用于预测,
    fluid.io.save_inference_model(
        dirname=modelfolder,
        feeded_var_names=['input'],
        target_vars=[outputvar],
        executor=exe,
        main_program=test_program,
        model_filename='model',
        params_filename='params')
    #save_persistables, load_persistables, load_vars

    #载入网络，并预测
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    program, feed_names, fetch_targets = fluid.io.load_inference_model(
        modelfolder, exe, 'model', 'params')

    fetch_list_name = [a.name for a in fetch_targets]
    print 'finish load model', 'feedname:', feed_names, 'fetch_list_name:', fetch_list_name

    print '==== inference'
    imagefilename = 'testdata/predict_cifar.png'
    image = cv2.imread(imagefilename, 1)
    inputdata = preprocess(image)
    outputlist = exe.run(
        program, fetch_list=fetch_targets, feed={'input': inputdata})
    label_value = outputlist[0]
    print imagefilename, 'predictlabel', label_value[0][0]


def test_train():
    """
        实际图像分类训练还需要解决的问题， 其中1-4 可以参考后面的特征学习示例1
        1. 训练调优：训练中评估、不同模型结构、学习率调整策略、参数初始化方法、损失优化策略。
        2. 怎么使用预训练模型在新任务中finetune
        3. 数据读取和样本增广及加速
        4. 调试网络、中断训练恢复，显存优化
        5. 大规模图像分类训练：集群数据(hdfs)读取，GPU 多卡训练，多机训练策略
        6. 预测服务：c++预测使用，tensorrt,anakin 加速, batch预测提高吞吐
        7. 模型量化压缩: int8模型量化训练、模型裁剪，int8模型预测
        
        解决实际问题还需要，可以参考后面的特征学习示例2
        1. 分类体系及标注数据
        2. 检测+多模型
        3. 数据迭代
    """
    print test_train.__doc__


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print >> sys.stderr, 'usage: python train.py test_xx args...'
