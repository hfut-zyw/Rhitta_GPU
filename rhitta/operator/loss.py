# -*- coding: utf-8 -*-
"""
Created on 2023/4/1
@author: hfut-zyw
"""

import cupy as cp
from .. import Tensor
from . import matrix_calculus


class LossFunction(Tensor):
    '''
    损失函数基类
    '''
    pass


class BinaryClassLoss(LossFunction):
    """
        predicts: （1，1） 可能是0，1，2维的数值  (t_1=parents[0])
        labels:   （1，1）                     (t_0=parents[1])
     """

    def compute(self):
        t_0 = self.parents[0].value
        t_1 = self.parents[1].value
        if t_1 == cp.array(1):
            self.value = -cp.log(t_0)
        if t_1 == cp.array(0):
            self.value = -cp.log(1 - t_0)

    def get_jacobi(self, parent):
        """
        仅返回对predicts的梯度即可
        """
        t_0 = self.parents[0].value
        t_1 = self.parents[1].value
        if t_1 == cp.array(1):
            gradient = -1 / t_0
        if t_1 == cp.array(0):
            gradient = 1 / (1 - t_0)
        return cp.array(gradient,ndmin=2)


class CrossEntropyLoss(LossFunction):
    """
    features: 1xc , c为类别数         (parents[0])
    label：   1xc ，one-hot vector   (parents[1])
    注意，接收的向量不是概率值向量，也就是普通特征向量
    这里实现的是CrossEntropyLossWithSoftmax
    """

    def compute(self):
        features = self.parents[0].value
        label = self.parents[1].value
        prob, _ = matrix_calculus.softmax(features)  # 这个地方特别容易错，因为Softmax函数返回的是两个东西
        self.value = -cp.sum(cp.multiply(label, cp.log(prob + 1e-10)))  # 1e-10防止出现log0

    def get_jacobi(self, parent):
        """
        仅对features求梯度，不对label求
        """
        features = self.parents[0].value
        label = self.parents[1].value
        prob, _ = matrix_calculus.softmax(features)
        jacobi = prob - label
        return cp.array(jacobi,ndmin=2)


class MSELoss(LossFunction):
    """
    features: 1xd   (parents[0])
    label：   1xd   (parents[1])
    """

    def compute(self):
        features = self.parents[0].value
        label = self.parents[1].value
        self.value = cp.sum(cp.power((features - label), 2))

    def get_jacobi(self, parent):
        """
        仅对features求梯度，不对label求
        """
        features = self.parents[0].value
        label = self.parents[1].value
        jacobi = 2 * (features.flatten() - label.flatten())
        return cp.array(jacobi,ndmin=2)
