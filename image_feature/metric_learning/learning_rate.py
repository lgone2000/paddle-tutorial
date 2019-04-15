# -*- coding: utf-8 -*-
import paddle
from paddle import fluid
import math
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
from paddle.fluid.initializer import init_on_cpu
import paddle.fluid.layers.ops as ops
from paddle.fluid.layers import control_flow

def cosine_decay_v2(learning_rate, totalsteps):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(global_step * (math.pi / totalsteps)) + 1)
    decrease lr for every mini-batch.
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        decayed_lr = learning_rate * \
                     (ops.cos(global_step * (math.pi / float(totalsteps))) + 1)/2
    return decayed_lr


def cosine_decay_v2_with_warmup(learning_rate, warmupsteps, totalsteps):
    """Applies cosine decay to the learning rate.
    lr = 0.05 * (math.cos(epoch * (math.pi / 120)) + 1)
    decrease lr for every mini-batch and start with warmup.
    """
    global_step = _decay_step_counter()
    lr = fluid.layers.tensor.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    with init_on_cpu():
        with control_flow.Switch() as switch:
            with switch.case(global_step < warmupsteps):
                decayed_lr = learning_rate * (global_step / float(warmupsteps))
                fluid.layers.tensor.assign(input=decayed_lr, output=lr)
            with switch.default():
                decayed_lr = learning_rate * \
                     (ops.cos((global_step - warmupsteps) * (math.pi / (totalsteps))) + 1)/2
                fluid.layers.tensor.assign(input=decayed_lr, output=lr)
    return lr
