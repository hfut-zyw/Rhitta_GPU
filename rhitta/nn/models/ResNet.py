from ..layers import *




class ResNet_simple(Module):
    def __init__(self, in_channels=1, num_classes=10):
        Module.__init__(self)
        # 卷积单元，包含1个卷积层
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=4, kernel_size=5)  # 24x24
        self.layernorm1 = LayerNorm()
        self.relu1 = ReLU(in_channels=4)
        self.pool1 = Pooling(in_channels=4, window_size=2, stride=2)  # 12x12
        # 残差单元，包含2个卷积层
        self.resblock = ResBlock4x4()  # 12x12

        # 卷积单元，包含一个全局平均汇聚层,另外加一个卷积降维
        self.conv2 = Conv2D(in_channels=4, out_channels=4, kernel_size=5)  # 8x8
        self.pool2 = Pooling(in_channels=4, window_size=4, stride=4, mode="AveragePooling")  # 2x2

        # 全连接层
        self.linear = Linear(2 * 2 * 4, num_classes)

    # 构造计算图
    def __call__(self, inputs):
        # 第1个单元
        output1 = self.conv1(inputs)
        output1 = self.layernorm1(output1)
        output1 = self.relu1(output1)
        output1 = self.pool1(output1)

        # 第2个单元
        output2 = self.resblock(output1)

        # 第3个单元
        output3 = self.conv2(output2)
        output4 = self.pool2(output3)
        vector = Concat(*output4)
        output = self.linear(vector)
        return output

class ResNet_untest(Module):
    def __init__(self, in_channels=1, num_classes=10):
        Module.__init__(self)
        # 第一个卷积单元，包含1个卷积层
        self.conv1 = Conv2D(in_channels=in_channels, out_channels=4, kernel_size=5)  # 24x24
        self.layernorm1 = LayerNorm()
        self.relu1 = ReLU(in_channels=4)
        self.pool1 = Pooling(in_channels=4, window_size=2, stride=2)  # 12x12
        # 第二个残差单元，包含2个卷积层
        self.resblock1 = ResBlock4x4()  # 12x12
        # 第三个残差单元，包含2个卷积层
        self.resblock2 = ResBlock4x8()  # 12x12
        # 第四个卷积单元，包含一个全局平均汇聚层,另外加一个卷积降维
        self.pool2 = Pooling(in_channels=8, window_size=4, stride=4, mode="AveragePooling")  # 3x3
        self.conv2 = Conv2D(in_channels=8, out_channels=16, kernel_size=3)  # 1x1
        # 第五个单元，全连接层
        self.linear = Linear(16, num_classes)

    # 构造计算图
    def __call__(self, inputs):
        # 第一个单元
        output1 = self.conv1(inputs)
        output1 = self.layernorm1(output1)
        output1 = self.relu1(output1)
        output1 = self.pool1(output1)

        # 第2、3个单元
        output2 = self.resblock1(output1)
        output3 = self.resblock2(output2)

        # 第4、5个单元
        output4 = self.pool2(output3)
        output5 = self.conv2(output4)
        vector = Concat(*output5)
        output = self.linear(vector)
        return output
