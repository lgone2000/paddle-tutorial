# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import paddle.fluid as fluid


class ArcMarginLoss():
    def __init__(self, class_dim, margin=0.15, scale=80.0, easy_margin=False):
        self.class_dim = class_dim
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

    def loss(self, input, label):
        out = self.arc_margin_product(input, label, self.class_dim,
                                      self.margin, self.scale,
                                      self.easy_margin)
        out = fluid.layers.softmax(input=out)
        loss = fluid.layers.cross_entropy(input=out, label=label)
        return loss, out

    def arc_margin_product(self,
                           input,
                           label,
                           out_dim,
                           m,
                           s,
                           easy_margin=False):
        epsilon = 0.0
        #对输出特征做L2 norm
        input = fluid.layers.l2_normalize(input, axis=1, epsilon=epsilon)
        weight = fluid.layers.create_parameter(
            shape=[input.shape[1], out_dim],
            dtype='float32',
            name='fc_class',
            attr=fluid.param_attr.ParamAttr(
                regularizer=fluid.regularizer.L2Decay(0.0),
                initializer=fluid.initializer.Xavier()),
        )

        #对分类的FC层参数也做L2norm
        weight = fluid.layers.l2_normalize(weight, axis=0, epsilon=epsilon)
        #做内积（FC）, 输出结果相当于和每个fc行的consine相似度
        cosine = fluid.layers.mul(input, weight)

        #m(marge)是为了促进同一类样本更好的聚合，并扩大类别间的距离。
        # 增加marge，使得正样本 cos value更小，因此分类界面会向正样本方向进一步压缩
        sine = fluid.layers.sqrt(
            fluid.layers.relu(1.0 - fluid.layers.square(cosine)))

        cos_m = math.cos(m)
        sin_m = math.sin(m)
        phi = cosine * cos_m - sine * sin_m

        #th = math.cos(math.pi - m)
        #mm = math.sin(math.pi - m) * m
        #if easy_margin:
        #    phi = self.paddle_where_more_than(cosine, 0, phi, cosine)
        #else:
        #    phi = self.paddle_where_more_than(cosine, th, phi, cosine-mm)

        one_hot = fluid.layers.one_hot(input=label, depth=out_dim)
        output = one_hot * phi + (1.0 - one_hot) * cosine
        #output = fluid.layers.elementwise_mul(one_hot, phi) + fluid.layers.elementwise_mul((1.0 - one_hot), cosine)
        output = output * s
        return output

    def paddle_where_more_than(self, target, limit, x, y):
        mask = fluid.layers.cast(x=(target > limit), dtype='float32')
        output = fluid.layers.elementwise_mul(
            mask, x) + fluid.layers.elementwise_mul((1.0 - mask), y)
        return output
