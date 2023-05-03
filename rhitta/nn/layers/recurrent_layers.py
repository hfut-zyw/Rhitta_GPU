from ...tensor import *
from ...operator import *
from ..module import *


class RNNCell(Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: 输入向量的维度
        :param hidden_size: 记忆神经元的维度
        """
        Module.__init__(self)
        self.container.add_layer(self)

        # 基本属性
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_params = hidden_size * (input_size + hidden_size + 1)

        # 初始化模型参数
        self.initializer = Normal_initializer(0, 1)
        self.params["W_x"] = to_tensor(size=(self.input_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["W_h"] = to_tensor(size=(self.hidden_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["b"] = to_tensor(size=(1, self.hidden_size), require_gradient=True, initializer=self.initializer)

    # 计算图构造，以及节点的进出口，只在第一次召唤生效
    def __call__(self, inputs, state):
        self.x = inputs
        self.h = state
        self.h = Tanh(Matmul(self.x, self.params["W_x"]) + Matmul(self.h, self.params["W_h"]) + self.params["b"])
        return self.h, self.h

    def __str__(self):
        return " RNNCell(input_size={},hidden_size={})   num_params：{} ".format(self.input_size, self.hidden_size,
                                                                                self.num_params)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size):
        """
        :param input_size: 输入向量的维度
        :param hidden_size: 记忆神经元的维度
        """
        Module.__init__(self)
        self.container.add_layer(self)
        # 基本属性
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_params = 4 * hidden_size * (input_size + hidden_size + 1)

        # 初始化模型参数
        self.initializer = Normal_initializer(0, 1)
        self.params["W_i"] = to_tensor(size=(self.input_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["U_i"] = to_tensor(size=(self.hidden_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["b_i"] = to_tensor(size=(1, self.hidden_size), require_gradient=True, initializer=self.initializer)
        self.params["W_f"] = to_tensor(size=(self.input_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["U_f"] = to_tensor(size=(self.hidden_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["b_f"] = to_tensor(size=(1, self.hidden_size), require_gradient=True, initializer=self.initializer)
        self.params["W_o"] = to_tensor(size=(self.input_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["U_o"] = to_tensor(size=(self.hidden_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["b_o"] = to_tensor(size=(1, self.hidden_size), require_gradient=True, initializer=self.initializer)
        self.params["W_c"] = to_tensor(size=(self.input_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["U_c"] = to_tensor(size=(self.hidden_size, self.hidden_size), require_gradient=True,
                                       initializer=self.initializer)
        self.params["b_c"] = to_tensor(size=(1, self.hidden_size), require_gradient=True, initializer=self.initializer)

        # 计算图构造，以及节点的进出口，只在第一次召唤生效

    def __call__(self, inputs, hidden_state, cell_state):
        # 输入
        self.x = inputs
        self.h = hidden_state
        self.c = cell_state

        # 门和候选内部状态
        self.i = Tanh(Matmul(self.x, self.params["W_i"]) + Matmul(self.h, self.params["U_i"]) + self.params["b_i"])
        self.f = Tanh(Matmul(self.x, self.params["W_f"]) + Matmul(self.h, self.params["U_f"]) + self.params["b_f"])
        self.o = Tanh(Matmul(self.x, self.params["W_o"]) + Matmul(self.h, self.params["U_o"]) + self.params["b_o"])
        self.c_tilde = Tanh(
            Matmul(self.x, self.params["W_c"]) + Matmul(self.h, self.params["U_c"]) + self.params["b_c"])

        # 计算输出状态
        self.c = Multiply(self.f, self.c) + Multiply(self.i, self.c_tilde)
        self.h = Multiply(self.o, Tanh(self.c))
        return self.h, self.c

    def __str__(self):
        return " LSTMCell(input_size={},hidden_size={})   num_params：{} ".format(self.input_size, self.hidden_size,
                                                                                 self.num_params)









