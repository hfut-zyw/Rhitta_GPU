from ..layers import *


class LSTM(Module):
    def __init__(self, input_size, hidden_size, time_dimension):
        """
        :param input_size: 输入向量的维度
        :param hidden_size: 记忆神经元的维度
        :param time_dimension: 时间序列长度
        """
        Module.__init__(self)

        # 基本属性
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_dimension = time_dimension
        self.num_params = 4 * hidden_size * (input_size + hidden_size + 1)  # 就是一个cell的参数量

        # 初始化模型参数
        self.lstmcell = LSTMCell(self.input_size, self.hidden_size)

        # 这个是自定义的模型，不要加入container容器中
        # self.container.add_layer(self)

    # 计算图构造
    def __call__(self, inputs, hidden_state, cell_state):
        """
        接收一个时间序列节点的列表，和两个初始状态节点
        """
        hidden_state, cell_state = self.lstmcell(inputs[0], hidden_state, cell_state)  # 先送进初始状态获取第一层隐藏状态
        N = self.time_dimension - 1
        for i in range(N):
            hidden_state, cell_state = self.lstmcell(inputs[i + 1], hidden_state, cell_state)
        return hidden_state  # 输出最后一个隐藏状态的节点作为输出
