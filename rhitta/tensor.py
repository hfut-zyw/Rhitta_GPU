import abc
import cupy as cp
from .graph import Graph, default_graph


class Tensor:
    """
    变量和算子的基类
    to_tensor:数据向量，参数向量
    Operator:各类算子的小基类
    """

    # 算子Operator自动调用来创建节点，同时进行双向连接,并把节点加入图容器当中
    def __init__(self, *parents, **kargs):
        self.parents = list(parents)
        self.children = []
        self.value = None
        self.grad = None
        self.graph = kargs.get("graph", default_graph)
        self.graph.nodes.append(self)
        for parent in self.parents:
            parent.children.append(self)

    # 获取基本属性的方法
    def get_parents(self):
        return self.parents

    def get_children(self):
        return self.children

    # 重置ndarray的shape属性，对后面判断矩阵运算的分类很重要
    def shape(self):
        # 如果是数值
        if len(self.value.shape) == 0:
            return 1, 1
        # 如果是一维向量
        elif len(self.value.shape) == 1:
            return 1, self.value.shape[0]
        # 如果是二维矩阵
        elif len(self.value.shape) == 2:
            return self.value.shape
        raise "矩阵形状有误"

    def dimension(self):
        if len(self.value.shape) == 0:
            return 1
        elif len(self.value.shape) == 1:
            return self.value.shape[0]
        elif len(self.value.shape) == 2:
            return self.value.shape[0] * self.value.shape[1]
        raise "矩阵形状有误"

    def __str__(self):
        return " value={}, \n grad={}, \n type={} \n".format(self.value, self.grad, type(self))

    def reset_value(self, recursive=True):
        self.value = None
        if recursive:
            for child in self.children:
                child.reset_value()

    def clear_grad(self):
        self.grad = None

    # 计算图构建完毕后，forward遍历所有节点，并通过compute方法计算出节点的value值，compute方法由各种各样多态的算子完成
    # 由于compute还没定义这里就用到了，所以后面需要使用抽象方法声明一下
    def forward(self):
        for parent in self.parents:
            if parent.value is None:
                parent.forward()
        self.compute()

    # 在每个节点的value值被计算出后，backward遍历所有节点，算出result对每个节点的梯度，存在grad值当中
    def backward(self, result):
        if self.grad is None:
            # 已抵达终点，并且就是我要的终点
            if self is result:
                self.grad = cp.eye(self.dimension())
                return cp.array(self.grad, ndmin=2)
            else:
                # 如果抵达非loss终点，终点对自己的jacobi会是0
                self.grad = cp.array(cp.zeros((result.dimension(), self.dimension())), ndmin=2)
                for child in self.children:
                    # 检查孩子节点的值，排除没有forward的路线
                    if child.value is not None:
                        self.grad += cp.matmul(child.backward(result), child.get_jacobi(self))
                return self.grad
        return self.grad

    @abc.abstractmethod
    def compute(self):
        """抽象方法，后面重写"""

    @abc.abstractmethod
    def get_jacobi(self, parent):
        """抽象方法，后面重写"""

    def __add__(self, other):
        """
        这部分需要用ops中的算子实现，为了避免循环import，把ops中的算子赋值一份到ops_copy中
        """
        return Add_(self, other)

    def __sub__(self, other):
        return Sub_(self, other)

    def __mul__(self, other):
        return Matmul_(self, other)

    def __truediv__(self, other):
        return Div_(self, other)

    def __pow__(self, power, modulo=None):
        return power_(self, pow=power)

    # def __eq__(self, other):  不要重载等号，否则会在你不知道的地方引发错误
    #     self.reset_value()
    #     self.value = np.array(other.value)


class to_tensor(Tensor):

    def __init__(self, size, require_gradient=False, initializer=None, **kargs):
        """
        必要参数：
        dim：矩阵的长和宽
        缺省参数：
        require_gradient：是否需要求导，默认False
        initializer：指定参数初始化的方式
        **kargs：有待进一步开发
        """
        Tensor.__init__(self, **kargs)
        self.size = size
        self.require_gradient = require_gradient
        if initializer is not None:
            self.value = initializer(self.size)

    def set_value(self, value):
        """
        给样本赋值以及参数更新，并重置下游节点的value
        """
        self.reset_value()
        if isinstance(value, to_tensor):
            self.value = cp.array(value.value)
        else:
            self.value = cp.array(value)
        assert self.shape() == self.size


"""
以下是参数初始化工具
"""


class Initializer:
    """
    所有初始化器的基类
    """

    def __init__(self):
        pass


class Normal_initializer(Initializer):
    def __init__(self, mean, var):
        Initializer.__init__(self)
        self.mean = mean
        self.var = var

    def __call__(self, dim):
        return cp.random.normal(self.mean, self.var, dim)


class Uniform_initializer(Initializer):
    def __init__(self, low, high):
        Initializer.__init__(self)
        self.low = low
        self.high = high

    def __call__(self, size):
        return cp.random.uniform(self.low, self.high, size)


"""
以下是辅助算子，专门为了重载运算符的，代码从ops中copy过来的。
由于基类运算符调用了子类，子类定义使用了基类，所以只能写在一个文件里，不然会引起循环import
"""


class Operator_(Tensor):
    '''
    辅助算子，为了实现Tensor的运算符重载
    '''
    pass


class Add_(Operator_):
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


class Sub_(Operator_):
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


class Matmul_(Operator_):
    """
    矩阵乘法，仅允许
    1.矩阵乘矩阵
    2.矩阵和数值相乘

    本框架中约定以下运算规则：（与cupy运算保持一致）
    1维向量和1维向量（其转置等于自己）相乘：点乘，结果为1维
    1维向量和2维矩阵相乘：矩阵乘法，结果为1维
    0维数值，1维数值，2维数值和矩阵相乘：点乘，结果为2维
    2维和2维相乘：矩阵乘法，结果为2维

    主要歧义的地方（认为的矩阵乘法，但是实际上是点乘）：
    -- 1维向量和1维向量转置，还是点乘，由于不符合shape[1]=shape[0]，所以调用本算子会出错，所以排除了可能会引发的逻辑错误
    -- (d,1)和(1,1)相乘，(d,1)shape属性与重置前保持一致，(1,1)可能是数值，1维数值，2维数值，但是无论哪种数值，这两个的
    乘法只能调用cupy的点乘，因而本算子必须要让这种情况进入到点乘运算的分支，只需要判断其一的dimension属性是否维1即可
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
                m = parent.value.shape[0]
                I = cp.eye(m)
                return cp.kron(I, self.parents[1].value.T)
            else:
                k = parent.value.shape[1]
                I = cp.eye(k)
                return cp.kron(self.parents[0].value, I)

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


class Div_(Operator_):
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


class power_(Operator_):
    """
    矩阵逐元素乘方
    """

    def __init__(self, *parents, **kargs):
        Operator_.__init__(self, *parents)
        self.pow = kargs.get("pow")

    def compute(self):
        # 直接利用广播机制乘方
        self.value = cp.power(self.parents[0].value, self.pow)

    def get_jacobi(self, parent):
        jacobi = self.pow * cp.diag((cp.power(parent.value, self.pow - 1)).flatten())
        return cp.array(jacobi, ndmin=2)
