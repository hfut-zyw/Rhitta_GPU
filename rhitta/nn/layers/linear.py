from ...tensor import *
from ...operator import *
from ..module import *


class Linear(Module):
    def __init__(self, input_size, output_size, activation=None):
        """
        :param input_size: 输入向量的维度
        :param output_size: 神经元个数，即输出个数（输出向量的维度）
        :param activation: 激活函数类型
        """
        Module.__init__(self)
        self.container.add_layer(self)

        # 基本属性
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.num_params = input_size * output_size + self.output_size

        # 初始化模型参数
        self.initializer = Normal_initializer(0, 1)
        self.params["W"] = to_tensor(size=(self.input_size, self.output_size), require_gradient=True,
                                     initializer=self.initializer)
        self.params["b"] = to_tensor(size=(1, self.output_size), require_gradient=True, initializer=self.initializer)

    # 计算图构造，以及节点的进出口，只在第一次召唤生效
    def __call__(self, inputs):
        self.x = inputs
        self.affine = Add(Matmul(self.x, self.params["W"]), self.params["b"])
        if self.activation == "relu":
            self.output = Relu(self.affine)
        elif self.activation == "Logistic":
            self.output = Logistic(self.affine)
        else:
            self.output = self.affine
        return self.output

    def __str__(self):
        return " Linear({},{})   num_params：{} ".format(self.input_size, self.output_size, self.num_params)
