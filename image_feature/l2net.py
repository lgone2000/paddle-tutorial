import paddle
import paddle.fluid as fluid

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}


class HardNet():
    def __init__(self):
        self.params = train_parameters

    def net(self, input, embedding_size):

        conv1 = self.conv_bn_layer(
            input, num_filters=32, filter_size=3, stride=1, padding=1)
        conv2 = self.conv_bn_layer(
            conv1, num_filters=32, filter_size=3, stride=1, padding=1)
        conv3 = self.conv_bn_layer(
            conv2, num_filters=64, filter_size=3, stride=2, padding=1)
        conv4 = self.conv_bn_layer(
            conv3, num_filters=64, filter_size=3, stride=1, padding=1)
        conv5 = self.conv_bn_layer(
            conv4, num_filters=128, filter_size=3, stride=2, padding=1)
        conv6 = self.conv_bn_layer(
            conv5, num_filters=128, filter_size=3, stride=1, padding=1)
        conv6 = fluid.layers.dropout(x=conv6, dropout_prob=0.3)
        out = self.conv_bn_layer(
            conv6,
            num_filters=embedding_size,
            filter_size=8,
            stride=1,
            padding=0)
        #print('#'*20, input.shape, out.shape)
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride,
                      padding,
                      groups=1,
                      act='relu'):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=act,
            bias_attr=False)
        return fluid.layers.batch_norm(input=conv, act=act)


def L2Net():
    model = HardNet()
    return model
