# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from utility import get_gpu_num
from .commonfunc import calculate_order_dist_matrix

#一个batch输入包含多类， 每类包含samples_each_class个样本， train_batch_size = num_gpus * batch_classnum * samples_each_class
#在batch内的样本  loss = relu (max(类内距离) - min(类间距离) - magin)
# 这个比 三元组损失要求的更严格, 训练集合包含噪声， 或者batch 较大时，都不易收敛
# 可以放松一些条件，比如 loss =  sum (relu(max(每行类内距离) - min(没行类间距离) - magin))
# 也可以松弛条件   loss = relu (max(类内距离) - topN(类间距离) - magin)

class QuadrupletLoss():
    def __init__(self, 
                 train_batch_size = 80, 
                 samples_each_class = 2,
                 margin = 0.1):
        self.margin = margin
        self.samples_each_class = samples_each_class
        self.train_batch_size = train_batch_size
        num_gpus = get_gpu_num()
        assert(train_batch_size % num_gpus == 0)
        self.cal_loss_batch_size = train_batch_size // num_gpus
        assert(self.cal_loss_batch_size % samples_each_class == 0)

    def loss(self, input, label=None):
        
        #特征层在计算距离前会被L2归一化。使用l2_normalize 应该和后面两句等价
        #input = fluid.layers.l2_normalize(input, axis=1)
        input_norm = fluid.layers.sqrt(fluid.layers.reduce_sum(fluid.layers.square(input), dim=1))
        input = fluid.layers.elementwise_div(input, input_norm, axis=0)

        samples_each_class = self.samples_each_class
        batch_size = self.cal_loss_batch_size
        margin = self.margin
        
        #计算距离矩阵，并重新排序，并将正样本距离调整到前面samples_each_class 列（原来在对角线附近）。自己到自己的距离为0，调整到第0列
        d = calculate_order_dist_matrix(input, self.cal_loss_batch_size, self.samples_each_class)
        ignore, pos, neg = fluid.layers.split(d, num_or_sections= [1, 
            samples_each_class-1, batch_size-samples_each_class], dim=1)
        
        #矩阵切分，去掉第一列，将矩阵分为正样本距离和负样本距离两部分
        ignore.stop_gradient = True
        pos_max = fluid.layers.reduce_max(pos)
        neg_min = fluid.layers.reduce_min(neg)
        #pos_max = fluid.layers.sqrt(pos_max + 1e-6)
        #neg_min = fluid.layers.sqrt(neg_min + 1e-6)
        
        #计算损失
        loss = fluid.layers.relu(pos_max - neg_min + margin)
        return loss
    
