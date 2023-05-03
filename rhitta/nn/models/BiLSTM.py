from ..layers import *


class BiLSTM(Module):
    def __init__(self, input_size, hidden_size, time_dimension, mode=1):
        """
        :param input_size: 输入向量的维度
        :param hidden_size: 记忆神经元的维度
        :param time_dimension: 时间序列长度
        """
        Module.__init__(self)

        # 基本属性
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_dimension = time_dimension
        self.num_params = 4 * hidden_size * (input_size + hidden_size + 1) * 2  # 就是正向和反向两个cell的参数量

        # 初始化模型参数
        self.lstmcell_1 = LSTMCell(self.input_size, self.hidden_size)
        self.lstmcell_2 = LSTMCell(self.input_size, self.hidden_size)

        # 这个是自定义的模型，不要加入container容器中
        # self.container.add_layer(self)

    # 计算图构造
    def __call__(self, inputs, hidden_state_1, cell_state_1, hidden_state_2, cell_state_2):
        """
        接收一个时间序列节点的列表，和4个初始状态节点
        """
        N = self.time_dimension - 1
        # 正向计算
        y_1 = []
        hidden_state, cell_state = self.lstmcell_1(inputs[0], hidden_state_1, cell_state_1)  # 先送进初始状态获取第一层隐藏状态
        y_1.append(hidden_state)
        for i in range(N):
            hidden_state, cell_state = self.lstmcell_1(inputs[i + 1], hidden_state, cell_state)
            y_1.append(hidden_state)
        # 反向计算
        y_2 = []
        hidden_state, cell_state = self.lstmcell_2(inputs[N], hidden_state_2, cell_state_2)  # 时间序列反向输入
        y_2.append(hidden_state)
        for i in range(N):
            hidden_state, cell_state = self.lstmcell_2(inputs[N - i - 1], hidden_state, cell_state)
            y_2.append(hidden_state)
        # 拼接
        y = []
        for i in range(N + 1):
            y.append(Concat(y_1[i], y_2[N - i]))

        # 模式0，输出n个隐藏状态
        if self.mode == 0:
            return y  # n进n出
        # 模式1，输出n个隐藏状态的平均
        elif self.mode == 1:
            result = None
            for i in range(N + 1):
                if i == 0:
                    result = y[i]
                else:
                    result = result + y[i]
            return result  # n进单出
        else:
            raise "mode只能为0或者1"
