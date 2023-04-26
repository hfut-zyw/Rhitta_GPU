from ..layers import *


class SCN(Module):
    def __init__(self, in_channels=1, num_classes=10):
        Module.__init__(self)
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=3, kernel_size=5)
        self.layernorm1 = LayerNorm()
        self.relu1 = ReLU(in_channels=3)
        self.pool1 = Pooling(in_channels=3, window_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=3, out_channels=5, kernel_size=3, stride=1)
        self.layernorm2 = LayerNorm()
        self.relu2 = ReLU(in_channels=5)
        self.pool2 = Pooling(in_channels=5, window_size=2, stride=2, mode="AveragePooling")

        self.conv3 = Conv2D(in_channels=5, out_channels=20, kernel_size=5, stride=1)

        self.linear = Linear(20, num_classes)

    # 构造计算图
    def __call__(self, inputs):
        output = self.conv1(inputs)
        output = self.layernorm1(output)
        output = self.relu1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.layernorm2(output)
        output = self.relu2(output)
        output = self.pool2(output)

        output = self.conv3(output)
        output = Concat(*output)
        output = self.linear(output)
        return output


class SCN2(Module):
    def __init__(self, in_channels=1, num_classes=10):
        Module.__init__(self)
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=3, kernel_size=5)
        self.layernorm1 = LayerNorm2()
        self.relu1 = ReLU(in_channels=3)
        self.pool1 = Pooling(in_channels=3, window_size=2, stride=2)

        self.conv2 = Conv2D(in_channels=3, out_channels=5, kernel_size=3, stride=1)
        self.layernorm2 = LayerNorm2()
        self.relu2 = ReLU(in_channels=5)
        self.pool2 = Pooling(in_channels=5, window_size=2, stride=2, mode="AveragePooling")

        self.conv3 = Conv2D(in_channels=5, out_channels=20, kernel_size=5, stride=1)
        self.list2tensor = List2Tensor()
        self.linear = Linear(20, num_classes)

    # 构造计算图
    def __call__(self, inputs):
        output = self.conv1(inputs)
        output = self.layernorm1(output)
        output = self.relu1(output)
        output = self.pool1(output)

        output = self.conv2(output)
        output = self.layernorm2(output)
        output = self.relu2(output)
        output = self.pool2(output)

        output = self.conv3(output)
        output = self.list2tensor(output)
        output = self.linear(output)
        return output


