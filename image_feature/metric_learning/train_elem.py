# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import time
import logging
import argparse
import functools
import threading
import subprocess
import numpy as np
import paddle
import paddle.fluid as fluid
import models
import reader
from losses import SoftmaxLoss
from losses import ArcMarginLoss
from utility import add_arguments, print_arguments
from utility import fmt_time, recall_topk, get_gpu_num, get_cpu_num
from learning_rate import cosine_decay_v2, cosine_decay_v2_with_warmup
from fluidpreprocess import preprocessimg
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('input_dtype', str, 'float32', "input blob's dtype uint8|float32")
add_arg('model', str, "ResNet50", "Set the network to use.")
add_arg('embedding_size', int, 0, "Embedding size.")
add_arg('train_batch_size', int, 256, "Minibatch size.")
add_arg('test_batch_size', int, 50, "Minibatch size.")
add_arg('image_shape', str, "3,224,224", "input image size")
add_arg('class_dim', int, 11318 , "Class number.")
add_arg('lr', float, 0.01, "set learning rate.")
add_arg('lr_strategy', str, "piecewise_decay", "Set the learning rate decay strategy.")
add_arg('lr_steps', str, "15000,25000", "step of lr")

add_arg('warmup_iter_num', int, 0, "warmup_iter_num")
add_arg('total_iter_num', int, 30000, "total_iter_num")
add_arg('display_iter_step', int, 10, "display_iter_step.")
add_arg('test_iter_step', int, 1000, "test_iter_step.")
add_arg('save_iter_step', int, 1000, "save_iter_step.")
add_arg('use_gpu', bool, True, "Whether to use GPU or not.")
add_arg('with_mem_opt', bool, True, "Whether to use memory optimization or not.")
add_arg('pretrained_model', str, None, "Whether to use pretrained model.")
add_arg('checkpoint', str, None, "Whether to resume checkpoint.")
add_arg('model_save_dir', str, "output", "model save directory")
add_arg('loss_name', str, "softmax", "Set the loss type to use.")
add_arg('arc_scale', float, 80.0, "arc scale.")
add_arg('arc_margin', float, 0.15, "arc margin.")
add_arg('arc_easy_margin', bool, False, "arc easy margin.")

add_arg('samples_each_class', int, 2, "samples_each_class.")
add_arg('margin', float, 0.1, "margin.")
add_arg('npairs_reg_lambda', float, 0.01, "npairs reg lambda.")

add_arg('enable_ce', bool, False, "If set True, enable continuous evaluation job.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]

#optimizer:分类优化器，一般选择Momentum(momentum=0.9)优化器， 学习率一般采用分段下降的调整（每次调整到原来1/10)，或者cosine 下降调整策略，
#learning-rate:基础学习率 如果从随机初始化训练开始，设置为0.1，如果finetune则会降低到0.01~0.001（前面加上warm-up)
#weightdecay:正则化 缺省一般用1e-4, 很大网络比如se-resnext152 可以适当放大到5.0e-4，小网络比如mobilenet训练小数据集 可以缩小到1e-5
#epochnum:训练轮数 目前统一到200， 会比90 高1%， finetune可以降低到10~20 epoch
#Momentum优化器文档  http://paddlepaddle.org/documentation/docs/zh/1.3/api_cn/optimizer_cn.html#permalink-15-momentum


def optimizer_setting(params, args):
    ls = params["learning_strategy"]
    assert ls["name"] in [
        "piecewise_decay", "cosine_decay", "cosine_decay_with_warmup"
    ]
    base_lr = params["lr"]

    if ls['name'] == "piecewise_decay":
        bd = [int(e) for e in ls["lr_steps"].split(',')]
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        lrs = fluid.layers.piecewise_decay(boundaries=bd, values=lr)
    elif ls['name'] == "cosine_decay":
        lrs = cosine_decay_v2(base_lr, args.total_iter_num)
    elif ls['name'] == "cosine_decay_with_warmup":
        lrs = cosine_decay_v2_with_warmup(base_lr, args.warmup_iter_num,
                                          args.total_iter_num)

    #相对于SGD ，使用Momentum 加快收敛速度而不影响收敛效果。如果用Adam，或者RMScrop收敛可以更快，但在imagenet上收敛有损失
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lrs,
        momentum=0.9,
        regularization=fluid.regularizer.L2Decay(1e-4))
    return optimizer


def createmodel(image, model, args):

    if args.input_dtype == 'uint8':
        assert (str(image.dtype) == 'VarType.UINT8')
        inputdata = preprocessimg(image, reader.img_mean, reader.img_std)
    else:
        inputdata = image

    out = model.net(input=inputdata, embedding_size=args.embedding_size)
    return out


def net_config_test(image, label, model, args):
    out = createmodel(image, model, args)
    return out, image, label


#基于deepid思路的，基于分类损失进行特征学习，对比 SoftmaxLoss, ArcMarginLoss。
#原始网络+降维的fc 做为特征输出层， embedding_size 控制特征层维度。
def net_config_metric(image, label, model, args):
    out = createmodel(image, model, args)

    if args.loss_name == "triplet":
        metricloss = TripletLoss(margin=args.margin, )
    elif args.loss_name == "quadruplet":
        metricloss = QuadrupletLoss(
            train_batch_size=args.train_batch_size,
            samples_each_class=args.samples_each_class,
            margin=args.margin,
        )
    elif args.loss_name == "eml":
        metricloss = EmlLoss(
            train_batch_size=args.train_batch_size,
            samples_each_class=args.samples_each_class,
        )
    elif args.loss_name == "npairs":
        metricloss = NpairsLoss(
            train_batch_size=args.train_batch_size,
            samples_each_class=args.samples_each_class,
            reg_lambda=args.npairs_reg_lambda,
        )
    cost, logit = metricloss.loss(out, label)
    return [avg_cost, out, label]


def net_config_classify(image, label, model, args):
    out = createmodel(image, model, args)

    if args.loss_name == "softmax":
        metricloss = SoftmaxLoss(class_dim=args.class_dim, )
    elif args.loss_name == "arcmargin":
        metricloss = ArcMarginLoss(
            class_dim=args.class_dim,
            margin=args.arc_margin,
            scale=args.arc_scale,
            easy_margin=args.arc_easy_margin,
        )
    cost, logit = metricloss.loss(out, label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=logit, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=logit, label=label, k=5)
    return [avg_cost, acc_top1, acc_top5]


#标准的feed数据方式，通过executer.run 的feed参数同步传入数据， 并获得fetch变量的值。这样会导致 feed数据和训练是串行的两步，
#采用py_reader可以使得feed数据和训练变成异步操作，加速训练过程
def build_program(is_train, net_config, main_prog, startup_prog, args):
    image_shape = [int(m) for m in args.image_shape.split(",")]
    model = models.__dict__[args.model]()
    assert args.model in model_list, "{} is not in lists: {}".format(
        args.model, model_list)

    #在 指定的main_prog 和start_prog 下组网（而不是用缺省的那个）
    with fluid.program_guard(main_prog, startup_prog):
        if is_train:
            queue_capacity = 64
            py_reader = fluid.layers.py_reader(
                capacity=queue_capacity,
                shapes=[[-1] + image_shape, [-1, 1]],
                lod_levels=[0, 0],
                dtypes=[args.input_dtype, "int64"],
                use_double_buffer=True)
            image, label = fluid.layers.read_file(py_reader)

        else:
            py_reader = None
            image = fluid.layers.data(
                name='image', shape=image_shape, dtype=args.input_dtype)
            label = fluid.layers.data(name='label', shape=[1], dtype='int64')

        #fluid.unique_name.guard()函数是为了初始化参数名称的时候统一名称，同样名字的参数在不同网络中属于一个参数，则在使用这个网络的时候保证用的同一套参数，而且这个函数的传参可以规定网络中所有参数开头的名称
        with fluid.unique_name.guard():
            #构建网络结构
            outputvars = net_config(image, label, model, args)
            if is_train:
                avg_cost = outputvars[0]
                params = model.params
                params["lr"] = args.lr
                params["learning_strategy"]["lr_steps"] = args.lr_steps
                params["learning_strategy"]["name"] = args.lr_strategy
                #根据配置创建优化器
                optimizer = optimizer_setting(params, args)
                optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()
                outputvars.append(global_lr)
    """            
    if not is_train:
        main_prog = main_prog.clone(for_test=True)
    """
    return py_reader, outputvars


class EvalTrain_Classify(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_info = [0, 0, 0, 0]

    def pushdata(self, outputlist):
        lr, loss, acc1, acc5 = outputlist
        self.lr = np.mean(np.array(lr))
        self.train_info[0] += np.mean(np.array(loss))
        self.train_info[1] += np.mean(np.array(acc1))
        self.train_info[2] += np.mean(np.array(acc5))
        self.train_info[3] += 1

    def getaccuracy(self):
        avg_loss = self.train_info[0] / self.train_info[3]
        avg_acc1 = self.train_info[1] / self.train_info[3]
        avg_acc5 = self.train_info[2] / self.train_info[3]
        return {
            'avg_loss': avg_loss,
            'avg_acc1': avg_acc1,
            'avg_acc5': avg_acc5,
            'lr': self.lr
        }


class EvalTrain_Metric(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_info = [0, 0, 0]

    def pushdata(self, outputlist):
        lr, loss, acc1, acc5 = outputlist
        self.lr = np.mean(np.array(lr))
        self.train_info[0] += np.mean(np.array(loss))
        self.train_info[1] += recall_topk(feas, label, k=1)
        self.train_info[2] += 1

    def getaccuracy(self):
        avg_loss = self.train_info[0] / self.train_info[2]
        avg_recall = self.train_info[1] / self.train_info[2]
        return {'avg_loss': avg_loss, 'avg_recall': avg_recall, 'lr': lr}


class EvalTest(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.f, self.l = [], []

    def pushdata(self, outputlist):
        feas, label = outputlist
        self.f.append(feas)
        self.l.append(label)

    def getaccuracy(self):
        f = np.vstack(self.f)
        l = np.hstack(self.l)
        recall = recall_topk(f, l, k=1)
        return {'recall': recall}


def train_async(args):
    # parameters from arguments

    logging.debug('enter train')
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    model_save_dir = args.model_save_dir

    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    tmp_prog = fluid.Program()

    #测试使用，固定随机参数种子
    if args.enable_ce:
        assert args.model == "ResNet50"
        assert args.loss_name == "arcmargin"
        np.random.seed(0)
        startup_prog.random_seed = 1000
        train_prog.random_seed = 1000
        tmp_prog.random_seed = 1000

    trainclassify = args.loss_name in ["softmax", "arcmargin"]
    train_py_reader, outputvars = build_program(
        is_train=True,
        net_config=net_config_classify,
        main_prog=train_prog,
        startup_prog=startup_prog,
        args=args)

    if trainclassify:
        train_cost, train_acc1, train_acc5, global_lr = outputvars
        train_fetch_list = [
            global_lr.name, train_cost.name, train_acc1.name, train_acc5.name
        ]
        evaltrain = EvalTrain_Classify()

    else:
        train_cost, train_feas, train_label, global_lr = outputvars
        train_fetch_list = [
            global_lr.name, train_cost.name, train_feas.name, train_label.name
        ]
        evaltrain = EvalTrain_Metric()

    _, outputvars = build_program(
        is_train=False,
        net_config=net_config_test,
        main_prog=tmp_prog,
        startup_prog=startup_prog,
        args=args)
    test_feas, image, label = outputvars
    test_prog = tmp_prog.clone(for_test=True)

    test_fetch_list = [test_feas.name]

    #打开内存优化，可以节省显存使用(注意，取出的变量要使用skip_opt_set设置一下，否则有可能被优化覆写)
    if args.with_mem_opt:
        fluid.memory_optimize(train_prog, skip_opt_set=set(train_fetch_list))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    #初始化变量
    exe.run(startup_prog)

    logging.debug('after run startup program')

    #从断点中恢复
    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint, main_program=train_prog)

    #加载预训练模型的参数到网络。如果使用预训练模型，最后一层fc需要改一下名字，或者删掉预训练模型的fc对应的权值文件
    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(
            exe, pretrained_model, main_program=train_prog, predicate=if_exist)

    #得到机器gpu卡数。
    #
    if args.use_gpu:
        devicenum = get_gpu_num()
        assert (args.train_batch_size % devicenum) == 0
    else:
        devicenum = get_cpu_num()
        assert (args.train_batch_size % devicenum) == 0
    #注意： 使用py_reader 的输入的batch大小，是单卡的batch大小，所以要除一下
    train_batch_size = args.train_batch_size // devicenum
    test_batch_size = args.test_batch_size

    logging.debug('device number is %d, batch on each card:%d', devicenum,
                  train_batch_size)

    #创建新的train_reader 将输入的reader读入的数据组成batch 。另外将train_reader 连接到 pyreader,由pyreader创建的线程主动读取，不在主线程调用。
    train_reader = paddle.batch(
        reader.train(args), batch_size=train_batch_size, drop_last=True)
    test_reader = paddle.batch(
        reader.test(args), batch_size=test_batch_size, drop_last=False)
    test_feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_py_reader.decorate_paddle_reader(train_reader)

    #使用ParallelExecutor 实现多卡训练
    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=args.use_gpu,
        loss_name=train_cost.name)

    totalruntime = 0
    #启动pyreader的读取线程
    train_py_reader.start()
    iter_no = 0
    while iter_no <= args.total_iter_num:
        t1 = time.time()
        #注意对于pyreader异步读取，不需要传入feed 参数了
        outputlist = train_exe.run(fetch_list=train_fetch_list)
        t2 = time.time()
        period = t2 - t1

        evaltrain.pushdata(outputlist)

        #计算多个batch的平均准确率
        if iter_no % args.display_iter_step == 0:
            avgruntime = totalruntime / args.display_iter_step
            train_accuracy = evaltrain.getaccuracy()

            print("[%s] trainbatch %d, "\
                    "accuracy[%s], time %2.2f sec" % \
                    (fmt_time(), iter_no, train_accuracy, avgruntime))
            sys.stdout.flush()
            totalruntime = 0
        if iter_no % 1000 == 0:
            evaltrain.reset()

        totalruntime += period

        if iter_no % args.test_iter_step == 0 and (pretrained_model or
                                                   checkpoint or iter_no != 0):
            #保持多个batch的feature 和 label 分别到 f, l
            evaltest = EvalTest()
            max_test_count = 100
            for batch_id, data in enumerate(test_reader()):
                t1 = time.time()
                test_outputlist = exe.run(
                    test_prog,
                    fetch_list=test_fetch_list,
                    feed=test_feeder.feed(data))

                label = np.asarray([x[1] for x in data])

                evaltest.pushdata((test_outputlist[0], label))
                t2 = time.time()
                period = t2 - t1
                if batch_id % 20 == 0:
                    print("[%s] testbatch %d, time %2.2f sec" % \
                            (fmt_time(), batch_id, period))
                if batch_id > max_test_count:
                    break
            #测试检索的准确率，当query和检索结果类别一致，检索正确。（这里测试数据集类别与训练数据集类别不重叠，因此网络输出的类别没有意义）
            test_recall = evaltest.getaccuracy()
            print("[%s] test_img_num %d, trainbatch %d, testaccarcy %s" % \
                    (fmt_time(), max_test_count * args.test_batch_size, iter_no, test_recall))
            sys.stdout.flush()

        if iter_no % args.save_iter_step == 0 and iter_no != 0:
            model_path = os.path.join(model_save_dir + '/' + model_name,
                                      str(iter_no))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            #保存模型， 可用于训练断点恢复
            fluid.io.save_persistables(
                exe, model_path, main_program=train_prog)

        iter_no += 1

    # This is for continuous evaluation only
    if args.enable_ce:
        # Use the mean cost/acc for training
        print("kpis train_cost      %s" % (avg_loss))
        print("kpis test_recall     %s" % (recall))


def initlogging():
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    loglevel = logging.DEBUG
    logging.basicConfig(
        level=loglevel,
        # logger.BASIC_FORMAT,
        format=
        "%(levelname)s:%(filename)s[%(lineno)s] %(name)s:%(funcName)s->%(message)s",
        datefmt='%a, %d %b %Y %H:%M:%S')


def main():
    initlogging()
    args = parser.parse_args()
    print_arguments(args)
    train_async(args)


if __name__ == '__main__':
    main()
