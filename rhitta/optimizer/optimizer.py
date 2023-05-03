# -*- coding: utf-8 -*-
"""
Created on 2023/4/2
@author: hfut-zyw
"""
import abc
import numpy as np
from .. import Tensor, to_tensor
from ..graph import Graph


class Optimizer:
    """
    优化器的基类

    基本属性：
    初始化必填参数
    graph：用于获取计算图的节点，以此来操作计算图
    loss：接收lossfunction，对这个目标损失函数进行梯度计算
    learning_rate：设定梯度下降法的步长
    内置属性（用于记录计算结果）
    acc_gradient：空字典，用于累积1个batch中每个样本跑计算图算出来的参数梯度，最后用于算平均梯度
    acc_n：计数变量，每轮batch中，从0开始计数，每跑一个样本，计数器加一，一轮跑完后计数器归零

    基本方法：
    forward_backward()：对loss执行前向传播，需要求梯度的参数执行反向传播，并把梯度值传给acc_gradient累加器
    one_step()：执行forward_backward(),计数器加一
    get_gradient():获取节点在当前批次的平均梯度
    前面一步执行计算，获取梯度
    _update()：由各种梯度下降法子类实现
    update()：执行_update()，清空优化器内置属性
    后面一步根据梯度和步长，更新参数
    zero_grad()：优化器初始化，也就是清空累积梯度
    """

    def __init__(self, graph, loss, learning_rate=0.01):
        self.graph = graph
        self.loss = loss
        self.learning_rate = learning_rate
        self.acc_gradient = dict()
        self.acc_n = 0

    def forward_backward(self):
        self.graph.clear_grad()
        self.loss.forward()
        for node in self.graph.nodes:
            if isinstance(node, to_tensor) and node.require_gradient == True:
                node.backward(self.loss)
                gradient = node.grad.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient

    def one_step(self):
        self.forward_backward()
        self.acc_n += 1

    def get_gradient(self, node):
        return self.acc_gradient[node] / self.acc_n

    @abc.abstractmethod
    def _update(self):
        """
        抽象方法，由子类实现，各种变种的梯度下降法
        """

    def update(self):
        self._update()
        self.acc_gradient.clear()  # 清空字典
        self.acc_n = 0

    def zero_grad(self):
        self.acc_gradient.clear()  # 梯度已经在上面字典中清空了，这里的方法是多余的,写在这里方便与其他框架进行对比


class SGD(Optimizer):
    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, to_tensor) and node.require_gradient == True:
                g = self.get_gradient(node)
                node.set_value(node.value - self.learning_rate * g)


class Momentum(Optimizer):
    def __init__(self, graph, loss, learning_rate, momentum):
        Optimizer.__init__(self, graph, loss, learning_rate)
        self.momentum = momentum
        self.v = dict()
        # 动量存储器在优化器一创建就会生成，根据以往的梯度累积能量，一般会设置衰减系数遗忘前面的梯度

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, to_tensor) and node.require_gradient == True:
                g = self.get_gradient(node)
                if node not in self.v:
                    self.v[node] = g
                else:
                    self.v[node] = self.momentum * self.v[node] - self.learning_rate * g
                node.set_value(node.value + self.v[node])


class AdaGrad(Optimizer):
    def __init__(self, graph, loss, learning_rate):
        Optimizer.__init__(self, graph, loss, learning_rate)
        self.v = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, to_tensor) and node.require_gradient == True:
                g = self.get_gradient(node)
                if node not in self.v:
                    self.v[node] = np.power(g, 2)
                else:
                    self.v[node] = self.v[node] + np.power(g, 2)
                node.set_value(node.value - self.learning_rate * g / np.sqrt(self.v[node] + 1e-10))


class RMSProp(Optimizer):
    """
    RMSProp优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta=0.9):

        Optimizer.__init__(self, graph, target)

        self.learning_rate = learning_rate

        # 衰减系数
        assert 0.0 < beta < 1.0
        self.beta = beta

        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, to_tensor) and node.require_gradient == True:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                # 滑动加权累积梯度各分量的平方和
                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + \
                                   (1 - self.beta) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))


class Adam(Optimizer):
    """
    Adam优化器
    """

    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):

        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

        # 历史梯度衰减系数
        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1

        # 历史梯度各分量平方衰减系数
        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2

        # 历史梯度累积
        self.v = dict()

        # 历史梯度各分量平方累积
        self.s = dict()

    def _update(self):

        for node in self.graph.nodes:
            if isinstance(node, to_tensor) and node.require_gradient == True:

                # 取得该节点在当前批的平均梯度
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.v[node] = gradient
                    self.s[node] = np.power(gradient, 2)
                else:
                    # 梯度累积
                    self.v[node] = self.beta_1 * self.v[node] + \
                                   (1 - self.beta_1) * gradient

                    # 各分量平方累积
                    self.s[node] = self.beta_2 * self.s[node] + \
                                   (1 - self.beta_2) * np.power(gradient, 2)

                # 更新变量节点的值
                node.set_value(node.value - self.learning_rate *
                               self.v[node] / np.sqrt(self.s[node] + 1e-10))
