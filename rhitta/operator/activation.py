import cupy as cp
from . import Operator
from . import matrix_calculus


class Logistic(Operator):

    def compute(self):
        functionValue, gradient = matrix_calculus.logistic(self.parents[0].value)
        self.value = functionValue

    def get_jacobi(self, parent):
        functionValue, gradient = matrix_calculus.logistic(self.parents[0].value)
        return gradient


class Tanh(Operator):

    def compute(self):
        functionValue, gradient = matrix_calculus.tanh(self.parents[0].value)
        self.value = functionValue

    def get_jacobi(self, parent):
        functionValue, gradient = matrix_calculus.tanh(self.parents[0].value)
        return gradient


class Softmax(Operator):
    """
    对于一维向量，就是一行进行softmax
    对于二维矩阵，就是每一行单独进行softmax
    """

    def compute(self):
        if len(self.parents[0].value.shape) == 1:
            functionValue, gradient = matrix_calculus.softmax(self.parents[0].value)
            self.value = functionValue
        if len(self.parents[0].value.shape) == 2:
            functionValue, gradient = matrix_calculus.matrix_softmax(self.parents[0].value)
            self.value = functionValue

    def get_jacobi(self, parent):
        if len(self.parents[0].value.shape) == 1:
            functionValue, gradient = matrix_calculus.softmax(self.parents[0].value)
            return gradient
        if len(self.parents[0].value.shape) == 2:
            functionValue, gradient = matrix_calculus.matrix_softmax(self.parents[0].value)
            return gradient


class Relu(Operator):  # 注意后面卷积里面还有个ReLU，大小写稍有区别
    """
    实现的是LeakyReLU哦
    """
    negative_slope = cp.array(0.1)  # 负半轴的斜率,写死在这里，不需要调整

    def compute(self):
        self.value = cp.where(self.parents[0].value > 0, self.parents[0].value,
                              self.negative_slope * self.parents[0].value)

    def get_jacobi(self, parent):
        return cp.diag(cp.where(self.parents[0].value.flatten() > 0.0, cp.array(1.0), self.negative_slope))
