from .convolution import *
from .normalization import LayerNorm

class ResAdd(Module):
    def __init__(self):
        Module.__init__(self)
        self.container.add_layer(self)

    def __call__(self, input1, input2):
        assert isinstance(input1, list) and isinstance(input2, list)
        assert len(input1) == len(input2)
        N = len(input1)
        out_feature_map = [input1[i] + input2[i] for i in range(N)]
        return out_feature_map

    def __str__(self):
        return "ResAdd()"

# （4，4）输入输入通道都是4的等宽残差卷积神经网络
class ResBlock4x4(Module):
    def __init__(self):
        Module.__init__(self)
        self.conv1 = Conv2D(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.layernorm1 = LayerNorm()
        self.relu = ReLU(in_channels=4)

        self.conv2 = Conv2D(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.layernorm2 = LayerNorm()

        self.add = ResAdd()

    # 构造计算图
    def __call__(self, inputs):
        output1 = self.conv1(inputs)
        output2 = self.layernorm1(output1)
        output3 = self.relu(output2)
        output4 = self.conv2(output3)
        output5 = self.layernorm2(output4)
        # 把输入的4通道和output5相加并ReLU后输出
        output6 = self.add(inputs, output5)
        output = self.relu(output6)
        return output


# （4，8）输入通道为4输出通道为8的等宽残差卷积神经网络
class ResBlock4x8(Module):
    def __init__(self):
        Module.__init__(self)
        self.conv1 = Conv2D(in_channels=4, out_channels=4, kernel_size=3, padding=1)
        self.layernorm1 = LayerNorm()
        self.relu1 = ReLU(in_channels=4)

        self.conv2 = Conv2D(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.layernorm2 = LayerNorm()
        self.relu2 = ReLU(in_channels=8)
        # 用于把输入的4通道翻倍
        self.conv1x1 = Conv2D(in_channels=4, out_channels=8, kernel_size=1)

        self.add = ResAdd()

    # 构造计算图
    def __call__(self, inputs):
        output1 = self.conv1(inputs)
        output2 = self.layernorm1(output1)
        output3 = self.relu1(output2)
        output4 = self.conv2(output3)
        output5 = self.layernorm2(output4)
        # 把输入的4通道通过1x1卷积将通道翻倍后与output5相加并ReLU后输出
        output6 = self.add(self.conv1x1(inputs), output5)
        output = self.relu2(output6)
        return output


# （8，8）输入输入通道都是8的等宽残差卷积神经网络
class ResBlock8x8(Module):
    def __init__(self):
        Module.__init__(self)
        self.conv1 = Conv2D(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.layernorm1 = LayerNorm()
        self.relu = ReLU(in_channels=8)

        self.conv2 = Conv2D(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.layernorm2 = LayerNorm()

        self.add = ResAdd()

    # 构造计算图
    def __call__(self, inputs):
        output1 = self.conv1(inputs)
        output2 = self.layernorm1(output1)
        output3 = self.relu(output2)
        output4 = self.conv2(output3)
        output5 = self.layernorm2(output4)
        # 把输入的4通道和output5相加并ReLU后输出
        output6 = self.add(inputs, output5)
        output = self.relu(output6)
        return output
