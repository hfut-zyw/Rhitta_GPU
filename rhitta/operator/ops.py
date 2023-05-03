# -*- coding: utf-8 -*-
"""
Created on 2023/4/1
@author: hfut-zyw
"""
import cupy as cp
from .. import Tensor


class Operator(Tensor):
    '''
    定义操作符抽象类
    '''
    pass


class Add(Operator):
    """
    矩阵加法:
    1.两个形状相同的矩阵相加
    2.一个矩阵，一个数值节点相加（仅仅为了支持Normalization）
    """

    def compute(self):
        self.value = self.parents[0].value + self.parents[1].value

    def get_jacobi(self, parent):
        # 对数值求导
        if parent.dimension() == 1:
            jacobi = cp.ones((self.dimension(), 1))
            return jacobi
        # 对矩阵求导
        return cp.eye(self.dimension())


class Sub(Operator):
    """
    矩阵减法：
    1.两个相同形状矩阵相减
    2.矩阵节点和数值节点相减
    """

    def compute(self):
        # 利用广播机制直接相减
        self.value = self.parents[0].value - self.parents[1].value

    def get_jacobi(self, parent):
        # 矩阵减矩阵，对第一个矩阵求导
        if self.parents[0].shape() == self.parents[1].shape() and parent is self.parents[0]:
            return cp.eye(self.dimension())
        # 矩阵减矩阵，对第二个矩阵求导
        elif self.parents[0].shape() == self.parents[1].shape() and parent is self.parents[1]:
            return -cp.eye(self.dimension())

        # 矩阵减数值，对矩阵求导
        elif parent is self.parents[0] and parent.dimension() != 1:
            return cp.eye(self.dimension())
        # 矩阵减数值，对数值求导
        elif parent is self.parents[1] and parent.dimension() == 1:
            return - cp.ones((self.dimension(), 1))

        # 数值减矩阵，对数值求导
        elif parent is self.parents[0] and parent.dimension() == 1:
            return cp.ones((self.dimension(), 1))
            # 数值减矩阵，对矩阵求导
        elif parent is self.parents[1] and parent.dimension() != 1:
            return -cp.eye(self.dimension())


class Matmul(Operator):
    """
    矩阵乘法，仅允许
    1.矩阵乘矩阵
    2.矩阵和数值相乘

    本框架中约定以下运算规则：（与cupy运算保持一致）
    1维向量和1维向量（其转置等于自己）相乘：计算内积，结果为0维，这种情况要极力避免，避免1维乘以1维的情况出现
    1维向量和2维矩阵相乘：矩阵乘法，结果为1维
    0维数值，1维数值，2维数值和矩阵相乘：点乘，结果为2维
    2维和2维相乘：矩阵乘法，结果为2维

    主要歧义的地方（认为的矩阵乘法，但是实际上是点乘）：
    -- 1维向量和1维向量转置，还是点乘，由于不符合shape[1]=shape[0]，所以调用本算子会出错，所以排除了可能会引发的逻辑错误
    -- (d,1)和(1,1)相乘，(d,1)shape属性与重置前保持一致，(1,1)可能是数值，1维数值，2维数值，但是无论哪种数值，这两个的
    乘法只能调用cupy的点乘，因而本算子必须要让这种情况进入到点乘运算的分支，只需要判断其一的dimension属性是否维1即可

    关于矩阵乘法中可能发生的矩阵维数变化：
    -- 1维向量使用矩阵乘法一路向前，还是1维的
    -- 2维向量使用矩阵乘法一路向前，还是2维的
    -- mean，sum，flatten,1维与1维矩阵相乘，squeeze等可能引发维数变化
    """

    def compute(self):
        assert len(self.parents) == 2
        if self.parents[0].dimension() == 1 or self.parents[1].dimension() == 1:
            self.value = self.parents[0].value * self.parents[1].value
        elif self.parents[0].shape()[1] == self.parents[1].shape()[0]:
            # 根据重置后的shape属性，上面这个条件可能会引发歧义的地方如下：
            # 如果输入是1维向量和1维向量相乘，由于shape重置了，不符合这里的条件，故而不会执行乘法操作导致实际是点乘的错误
            # 如果是数值乘以行向量的情况，满足重置后shape的条件，但是这种情况会进入上面第一个if语句
            self.value = cp.matmul(self.parents[0].value, self.parents[1].value)
        else:
            # 输入的是一维行向量与一维行向量或者矩阵乘法的两个矩阵形状不匹配的情况下会进入这个分支
            assert "你的矩阵乘法有误"


    def get_jacobi(self, parent):
        """
        将矩阵乘法视作映射，求映射对参与计算的矩阵的雅克比矩阵。
        """
        # 向量乘矩阵，矩阵乘矩阵
        if self.parents[0].shape()[1] == self.parents[1].shape()[0]:
            if parent is self.parents[0]:
                m = parent.shape()[0]
                I = cp.eye(m)
                return cp.kron(I, cp.array(self.parents[1].value).T)
            else:
                k = parent.shape()[1]
                I = cp.eye(k)
                return cp.kron(cp.array(self.parents[0].value), I)

        else:
            # 数值乘以矩阵,对数值求导
            if self.parents[0].dimension() == 1 and parent is self.parents[0]:
                return self.parents[1].value.reshape(-1, 1)
            # 数值乘以矩阵,对矩阵求导
            if self.parents[0].dimension() == 1 and parent is self.parents[1]:
                dim = parent.dimension()
                I = cp.eye(dim)
                y = self.parents[0].value
                return y * I
            # 矩阵乘以数值,对数值求导
            if self.parents[1].dimension() == 1 and parent is self.parents[1]:
                return self.parents[0].value.reshape(-1, 1)
            # 矩阵乘以数值,对矩阵求导
            if self.parents[1].dimension() == 1 and parent is self.parents[0]:
                dim = parent.dimension()
                I = cp.eye(dim)
                y = self.parents[1].value
                return y * I


class Multiply(Operator):
    """
    两个父节点的值是相同形状的矩阵，将它们对应位置的值相乘
    """

    def compute(self):
        assert self.parents[0].shape() == self.parents[1].shape()
        self.value = cp.multiply(self.parents[0].value, self.parents[1].value)

    def get_jacobi(self, parent):
        if parent is self.parents[0]:
            return cp.diag(self.parents[1].value.flatten())
        else:
            return cp.diag(self.parents[0].value.flatten())


class Div(Operator):
    """
    矩阵除法：
    矩阵除以数值，并且数值不能为0，如果矩阵除以矩阵，很容易出现分母为0，所以仅实现矩阵除以非0数值
    """

    def compute(self):
        # 直接利用广播机制相除
        assert self.parents[1].dimension() == 1 and self.parents[1].value != 0
        self.value = self.parents[0].value / self.parents[1].value

    def get_jacobi(self, parent):
        # 对矩阵求导
        if parent is self.parents[0]:
            jacobi = cp.eye(self.dimension()) / self.parents[1].value
            return cp.array(jacobi, ndmin=2)
        # 对数值求导
        else:
            temp = self.parents[0].value.reshape(-1, 1)  # 先把矩阵变成一个列向量
            jacobi = -temp / (self.parents[1].value ** 2)
            return cp.array(jacobi, ndmin=2)


class sum(Operator):
    """
    只允许对损失矩阵求和，也就是每一个元素都是一个样本的损失值，根据雅可比算子对求和的线性性，
    对损失Ji求和后求梯度，就是平均梯度，达到了批处理的效果。一般情况不要将不同样本的输出结果进行运算，会导致互相干扰
    也可以使用for对每个样本的运算求梯度，再求平均。这种方法概念清晰，时间复杂度较高
    """

    def compute(self):
        # assert len(self.parents) == 1 and self.parents[0].shape()[1] == 1
        self.value = cp.sum(self.parents[0].value)

    def get_jacobi(self, parent):
        jacobi = cp.ones(parent.dimension())
        return cp.array(jacobi, ndmin=2)


class mean(Operator):
    def compute(self):
        assert len(self.parents) == 1
        self.value = cp.mean(self.parents[0].value)

    def get_jacobi(self, parent):
        jacobi = cp.ones(parent.dimension()) / parent.dimension()
        return cp.array(jacobi, ndmin=2)


class mean_constant(Operator):
    def compute(self):
        assert len(self.parents) == 1
        self.value = cp.sum(self.parents[0].value) / self.parents[0].dimension()
        # 执行完计算后，断开x的child线路，但是mean的父节点不能断，否则无法计算mean的值
        # 第一个样本执行完后，计算图已经修改，后面的样本执行时，remove不起作用
        self.parents[0].children.remove(self)


class power(Operator):
    """
    矩阵逐元素乘方
    """

    def __init__(self, *parents, **kargs):
        Operator.__init__(self, *parents)
        self.pow = kargs.get("pow")

    def compute(self):
        # 直接利用广播机制乘方
        self.value = cp.power(self.parents[0].value, self.pow)

    def get_jacobi(self, parent):
        jacobi = self.pow * cp.diag((cp.power(parent.value, self.pow - 1)).flatten())
        return cp.array(jacobi, ndmin=2)


class sqrt(Operator):
    """
    数值开根号
    """

    def compute(self):
        self.value = cp.sqrt(self.parents[0].value)

    def get_jacobi(self, parent):
        t = 2 * self.value
        jacobi = 1 / t
        return cp.array(jacobi, ndmin=2)


class Concat(Operator):
    """
    将多个父节点的值连接成向量
    """

    def compute(self):
        assert len(self.parents) > 0

        # 将所有父节点矩阵展平并连接成一个向量
        self.value = cp.concatenate([p.value.flatten() for p in self.parents])

    def get_jacobi(self, parent):
        assert parent in self.parents

        dimensions = [p.dimension() for p in self.parents]  # 各个父节点的元素数量
        pos = self.parents.index(parent)  # 当前是第几个父节点
        dimension = parent.dimension()  # 当前父节点的元素数量

        assert dimension == dimensions[pos]

        jacobi = cp.zeros((self.dimension(), dimension))
        start_row = int(cp.sum(cp.array(dimensions[:pos])))
        jacobi[start_row:start_row + dimension, 0:dimension] = cp.eye(dimension)
        return jacobi

# class list2tensor(Operator):
#     def compute(self):
#         data = []
#         # assert self.parents is None
#         for parent in self.parents:
#             data.append(parent.value.getA()[0][0])
#         self.value = cp.mat(data)  # 返回一个行向量节点
#
#     def get_jacobi(self, parent):
#         dim = len(self.parents)
#         jacobi = cp.zeros((dim, 1))
#         for i in range(dim):
#             if parent is self.parents[i]:
#                 jacobi[i][0] = 1
#                 return cp.mat(jacobi)
