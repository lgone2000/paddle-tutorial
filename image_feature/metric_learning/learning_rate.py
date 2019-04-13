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


