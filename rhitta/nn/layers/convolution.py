from ...tensor import *
from ...operator import *
from ..module import *





class Conv2D(Module):
    """
    一些说明：
    1.这里的卷积计算操作是通过conv2d算子实现的
    2.conv2d的卷积核尺寸是正方形，横向步长与纵向步长也是取相同的
    3.每个输出通道的偏置是不一样的
    """

    def __init__(self, in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0):
        """
        可选参数，默认值看上面提示
        :param in_channels: 输入通道数,默认值为1
        :param out_channels: 输出通道数，默认值为1
        :param kernel_size: 卷积核大小，默认值为2
        :param stride: 卷积步长，默认值为1
        :param padding: 填充，默认值为0
        """
        Module.__init__(self)
        self.container.add_layer(self)

        # 基本属性
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initializer = Normal_initializer(0, 1)

        # 辅助属性，用于输出卷积核的数量
        self.num_kernels = None  # 需要的卷积核数量

        # 单通道到单通道
        if self.in_channels == 1 and self.out_channels == 1:
            self.kernel = to_tensor(size=(self.kernel_size, self.kernel_size), require_gradient=True,
                                    initializer=self.initializer)
            self.params["kernel"] = self.kernel
            self.bias = to_tensor(size=(1, 1), require_gradient=True, initializer=self.initializer)
            self.params["bias"] = self.bias
            self.num_kernels = 1
            self.num_params = kernel_size * kernel_size + 1  # 一个卷积核与一个偏置

        # 单通道到多通道
        elif self.in_channels == 1:
            self.kernel = []
            for i in range(self.out_channels):
                # 每个输出通道i仅需要一个卷积核
                self.kernel.append(to_tensor(size=(self.kernel_size, self.kernel_size), require_gradient=True,
                                             initializer=self.initializer))
            for i in range(self.out_channels):
                self.params["kernel[{}]".format(i)] = self.kernel[i]

            self.bias = []
            for i in range(self.out_channels):
                # 每个输出通道i都有一个偏置
                self.bias.append(to_tensor(size=(1, 1), require_gradient=True, initializer=self.initializer))
            for i in range(self.out_channels):
                self.params["bias[{}]".format(i)] = self.bias[i]
            self.num_kernels = self.out_channels
            self.num_params = (kernel_size * kernel_size + 1) * out_channels  # 每个输出通道需要一个卷积核与一个偏置

        # 多通道到单通道
        elif self.out_channels == 1:
            self.kernel = []
            for i in range(self.in_channels):
                self.kernel.append(to_tensor(size=(self.kernel_size, self.kernel_size), require_gradient=True,
                                             initializer=self.initializer))
            for i in range(self.in_channels):
                self.params["kernel[{}]".format(i)] = self.kernel[i]

            self.bias = to_tensor(size=(1, 1), require_gradient=True, initializer=self.initializer)
            self.params["bias"] = self.bias
            self.num_kernels = self.in_channels
            self.num_params = (kernel_size * kernel_size + 1) * in_channels  # in_channel个卷积核与一个偏置

        # 多通道到多通道
        else:
            self.kernel = []
            for i in range(self.out_channels):
                # 每个输出通道i仅需要一个卷积核
                self.kernel.append(to_tensor(size=(self.kernel_size, self.kernel_size), require_gradient=True,
                                             initializer=self.initializer))
            for i in range(self.out_channels):
                self.params["kernel[{}]".format(i)] = self.kernel[i]

            self.bias = []
            for i in range(self.out_channels):
                # 每个输出通道i都有一个偏置
                self.bias.append(to_tensor(size=(1, 1), require_gradient=True, initializer=self.initializer))
            for i in range(self.out_channels):
                self.params["bias[{}]".format(i)] = self.bias[i]
            self.num_kernels = self.out_channels
            self.num_params = (kernel_size * kernel_size + 1) * out_channels  # 每个输出通道需要一个卷积核与一个偏置

    # 计算图构造

    def __call__(self, inputs):
        output = None

        # 单通道到单通道
        if self.in_channels == 1 and self.out_channels == 1:
            conv = conv2d(inputs, self.kernel, stride=self.stride, padding=self.padding)
            output = conv + self.bias
            return output

        # 单通道到多通道
        elif self.in_channels == 1:
            out_feature_map = []
            for i in range(self.out_channels):
                # 每个通道的卷积核与输入图片的卷积加偏置
                temp = conv2d(inputs, self.kernel[i], stride=self.stride, padding=self.padding) + self.bias[i]
                out_feature_map.append(temp)
            return out_feature_map

        # 多通道到单通道
        elif self.out_channels == 1:
            convs = []
            for i in range(self.in_channels):
                convs.append(conv2d(inputs[i], self.kernel[i], stride=self.stride, padding=self.padding))
            for conv in convs:
                if output is None:
                    output = conv
                else:
                    output += conv
            output += self.bias
            return output

        # 多通道到多通道
        else:
            out_feature_map = []
            for i in range(self.out_channels):
                # 先计算第i个输出通道的feature map
                convs = []
                for j in range(self.in_channels):
                    # 卷积核与每个feature map的卷积
                    convs.append(conv2d(inputs[j], self.kernel[i], stride=self.stride, padding=self.padding))
                for conv in convs:
                    if output is None:
                        output = conv
                    else:
                        output += conv
                output += self.bias[i]
                out_feature_map.append(output)
                output = None
            return out_feature_map

    def __str__(self):
        return " Conv2D(in_channels={},out_channels={},kernel_size={},stride={},padding={})" \
            .format(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding) \
            + " kernel : [{},{},{}]  num_params：{} " \
                .format(self.num_kernels, self.kernel_size, self.kernel_size,
                        self.num_params)


class Pooling(Module):
    def __init__(self, in_channels=1, window_size=2, stride=1, mode="MaxPooling"):
        """
        可选参数，默认值看上面提示
        :param in_channels: 输入通道数
        :param window_size: 滑动窗口大小
        :param stride: 滑动步长
        :param mode: 池化模式 “MaxPooling" or "AveragePooling"
        """
        Module.__init__(self)
        self.container.add_layer(self)

        self.in_channels = in_channels
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.num_params = 0

    # 因为没有什么参数，所以直接开始计算图构造

    def __call__(self, inputs):

        # 单通道到单通道
        if self.in_channels == 1:
            if self.mode == "MaxPooling":
                return MaxPooling(inputs, stride=self.stride, window_size=self.window_size)
            elif self.mode == "AveragePooling":
                return AveragePooling(inputs, stride=self.stride, window_size=self.window_size)
            else:
                raise "mode ERROR!"

        # 多通道到多通道
        else:
            assert self.in_channels == len(inputs)
            feature_map = []
            if self.mode == "MaxPooling":
                for i in range(self.in_channels):
                    feature_map.append(MaxPooling(inputs[i], stride=self.stride, window_size=self.window_size))
                return feature_map
            elif self.mode == "AveragePooling":
                for i in range(self.in_channels):
                    feature_map.append(AveragePooling(inputs[i], stride=self.stride, window_size=self.window_size))
                return feature_map
            else:
                raise "mode ERROR!"

    def __str__(self):
        return " Pooling(in_channels={},window_size={},stride={},mode={})   ".format(self.in_channels, self.window_size,
                                                                                     self.stride, self.mode)


class ReLU(Module):
    def __init__(self, in_channels=1):
        Module.__init__(self)
        self.container.add_layer(self)
        self.in_channels = in_channels
        self.num_params = 0

    def __call__(self, inputs):
        if self.in_channels == 1:
            return Relu(inputs)
        else:
            feature_map = []
            for x in inputs:
                feature_map.append(Relu(x))
            return feature_map

    def __str__(self):
        return " ReLU() "
