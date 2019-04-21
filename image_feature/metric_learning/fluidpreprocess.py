# -*- coding: utf-8 -*-
from paddle import fluid
import numpy as np


#在网络里面减均值除方差
def preprocessimg(image, mean, std):
    data_ori = fluid.layers.cast(x=image, dtype='float32')
    mean_values_numpy = np.array(mean, np.float32).reshape(
        -1, 1, 1).astype(np.float32)
    mean_values = fluid.layers.create_tensor(dtype="float32")
    fluid.layers.assign(input=mean_values_numpy, output=mean_values)
    mean_values.stop_gradient = True

    std_values_numpy = np.array(std, np.float32).reshape(-1, 1, 1).astype(
        np.float32)
    std_values = fluid.layers.create_tensor(dtype="float32")
    fluid.layers.assign(input=std_values_numpy, output=std_values)
    std_values.stop_gradient = True

    datasubmean = fluid.layers.elementwise_sub(data_ori, mean_values)
    datasubmean.stop_gradient = True
    inputdata = fluid.layers.elementwise_div(datasubmean, std_values)
    inputdata.stop_gradient = True
    return inputdata
