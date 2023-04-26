# -*- coding: utf-8 -*-
"""
Created on 2023/3/29
@author: hfut-zyw
"""

import cupy as cp

"""
Commutation Matrix
"""


def comm_mat(m, n):
    # determine permutation applied by K
    w = cp.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return cp.eye(m * n)[w, :]


# 测试
# if __name__=="__main__":
#     print("\n","换算矩阵测试","\n")
#     k1=comm_mat(3,2)
#     k2=comm_mat(2,3)
#     print(k1)
#     print(k2)
#     print(cp.arraymul(k1,k2))
#     print(cp.arraymul(k1,k1.T))


"""
logistic()
"""


def logistic(x):
    t_0 = cp.exp(-x)
    t_1 = 1 + t_0
    functionValue = 1 / t_1
    gradient = cp.diag((t_0 / cp.multiply(t_1, t_1)).flatten())

    return functionValue, gradient


# 测试
# if __name__=="__main__":
#     print("\n","Logistic()测试","\n")
#     x=cp.array([[1],[2],[3]])
#     a,b=Logistic(x)
#     k13=comm_mat(1,3)
#     k31=comm_mat(3,1)
#     print(b)
#     print(cp.arraymul(k31,b,k13))


"""
tanh():
"""


def tanh(x):
    t_0 = cp.exp(x)
    t_1 = cp.exp(-x)
    functionValue = (t_0 - t_1) / (t_0 + t_1)
    gradient = cp.diag((1 - cp.power(functionValue, 2)).flatten())

    return functionValue, gradient


# 测试
# if __name__ == "__main__":
#     import torch
#     import torch.nn.functional as F
#
#     print("\n", "Tanh():测试", "\n")
#     x = cp.array([[1], [2], [3]])
#     a, b = Tanh(x)
#     print("首先输出自己写的tanh（x）的值，和雅可比")
#     print(a)
#     print(b)
#     y = torch.tensor([[1.], [2.], [3.]], requires_grad=True)
#     u = torch.sum(F.tanh(y))
#     u.backward()
#     print("下面这个是torch的tanh和sum复合后的值和求导结果，只要把上面的梯度左乘[1,1,1],"
#           "就能得到这里的梯度，注意元素排列方式不同，但结果准确")
#     print(u)
#     print(y.grad)


"""
softmax()
输入必须为：行向量或者列向量矩阵
cupy中列一维向量就是一维行向量，不存在列一维向量
"""


def softmax(x):
    assert len(x.shape) == 1 or len(x.shape) == 2
    # 当x超过100或者小于-100时，数据截断取100
    x = cp.array(x)
    x = cp.where(x > 1e2, 1e2, x)
    x = -cp.where(-x > 1e2, 1e2, -x)
    # 处理一维向量
    if len(x.shape) == 1:
        t_0 = cp.array(cp.exp(x), ndmin=2)  # 必须转换为二维矩阵，行向量矩阵，后面需要它的转置乘自己得到一个矩阵
        t_1 = cp.sum(t_0)
        t_2 = 1 / t_1
        functionValue = t_2 * t_0
        gradient = ((t_2 * cp.diag(t_0.flatten())) - ((1 / (t_1 ** 2)) * cp.matmul(t_0.T, t_0)))
        return functionValue, gradient
    # 处理二维行向量
    elif len(x.shape) == 2 and x.shape[0]==1:
        t_0 = cp.exp(x)  # 已经是二维行向量矩阵了
        t_1 = cp.sum(t_0)
        t_2 = 1 / t_1
        functionValue = t_2 * t_0
        gradient = ((t_2 * cp.diag(t_0.flatten())) - ((1 / (t_1 ** 2)) * cp.matmul(t_0.T, t_0)))
        return functionValue, gradient
    # 处理二维列向量，实际上本框架的softmax只对行操作，不对列操作，下面的算法是对于的
    elif len(x.shape) == 2 and x.shape[1] == 1:
        t_0 = cp.exp(x)  # 就是一个二维矩阵，列向量矩阵
        t_1 = cp.sum(t_0)
        t_2 = 1 / t_1
        functionValue = t_2 * t_0
        gradient = ((t_2 * cp.diag(t_0.flatten())) - ((1 / (t_1 ** 2)) * cp.matmul(t_0, t_0.T)))
        return functionValue, gradient
    raise "softmax()输入有误"

    # 测试


# 1.测试截断方式的逻辑是否正确
# 2.输入分别是行向量，列向量时，看输出是否是行向量（多分类概率分布），梯度是否相等
# 3.把数值改大，测试新老函数是否会数值溢出


# if __name__ == "__main__":
#     print("\n", "===数值截断测试===", "\n")
#     y = cp.array([[-4354], [1], [1054]])
#     z_ = cp.where(y > 1e2, 100, y)
#     z = -cp.where(-z_ > 1e2, 100, -z_)
#     print(y)
#     print(z_)
#     print(z)
#     print("\n", "~~~数值截断测试ok~~~", "\n")
#
#     print("\n", "===softmax()测试===", "\n")
#     x1 = cp.array([[1, 2, 3]])  # 行向量
#     x2 = cp.array([[1], [2], [3]])  # 列向量
#     a, b = softmax(x1)  # 未改动的函数
#     c, d = softmax(x2)  # 改进后的函数
#     print("函数值：",a,"\n","梯度：",b)
#     print("函数值：",c,"\n","梯度：",d)
#     print("\n", "~~~softmax()测试ok~~~", "\n")


"""
matrix_softmax()
"""


def matrix_softmax(x):
    m = x.shape[0]
    n = x.shape[1]
    dim = m * n
    value = []
    grad = []
    for i in range(m):  # 遍历每一行
        value_i, grad_i = softmax(x[i])
        value.append(value_i)
        grad.append(grad_i)
    functionValue = cp.concatenate(value, axis=0)  # 每一行处理好后，拼接起来
    gradient = cp.zeros((dim, dim))
    for i in range(m):
        gradient[n * i:n * i + n, n * i:n * i + n] = grad[i]
    return functionValue, gradient


if __name__ == "__main__":
    print("\n", "===矩阵matrix_softmax测试===", "\n")

    x = cp.array([[1, 2, 3], [4, 4, 6]])
    a, b = matrix_softmax(x)
    print("值为：", "\n", a)
    print("梯度为", "\n", b)

    # 用softmax分别处理每一行
    x1 = cp.array([1, 2, 3])
    x2 = cp.array([4, 4, 6])
    a1, b1 = softmax(x1)
    a2, b2 = softmax(x2)
    print("每一行求值的对比结果")
    print(a1)
    print(a2)
    print("每一行求梯度的对比结果")
    print(b1)
    print(b2)



"""
matrix_softmax()与掩码求梯度
"""
# if __name__ == "__main__":
#     print("\n", "===掩码求梯度实验===", "\n")
#     # 希望能得到在掩码的位置的梯度为0
#     x = cp.array([[1, 2, 3], [4, 4, 6]])
#     mask = cp.array([[0, 0, -1e9], [0, 0, -1e9]])
#     a, b = matrix_softmax(x + mask)  # a是值，b是梯度
#     print("softmax()梯度为", "\n", b.round(5))
