import cupy as cp
import numpy as np

"""
测试1：矩阵乘法
"""
# print("行向量乘以矩阵——>>行向量")
# x = cp.array([1, 4])
# w = cp.array([[1, 2], [3, 4]])
# c = cp.matmul(x,w)
# print(x.shape,w.shape,c,c.shape)
#
# print("行向量乘以矩阵——>>行向量")
# x = cp.array([1, 4])
# w = cp.array([[1], [2]])
# c = cp.matmul(x,w)
# print(x.shape,w.shape,c,c.shape)
#
# print("一行的矩阵乘以矩阵——>>矩阵")
# x = cp.array([[1, 4]])
# w = cp.array([[1, 2], [3, 4]])
# c = cp.matmul(x,w)
# print(x.shape,w.shape,c,c.shape)
#
# print("向量乘以向量——>>内积，获得0维数值")
# x = cp.array([1, 4])
# w = cp.array([1, 2])
# c = cp.matmul(x,w)
# print(x.shape,w.shape,c,c.shape)
#
# print("行向量矩阵乘以列向量矩阵——>>获得二维的数值，再使用squeeze降维成0维")
# a = cp.array([[1, 4, 3]])
# b = cp.array([[4, 5, 6]])
# c = cp.matmul(a, b.T)
# print(c,c.squeeze())

"""
测试2：拉平，对角化
"""
# a = cp.array([[1, 2], [3, 4]])
# b = a.flatten()
# print(b)
# print(cp.diag(b))
#


"""
测试3：矩阵维数强制转换
"""
# x = cp.array(1, ndmin=2)
# y = cp.array(cp.ones(4), ndmin=2)
# print(x, y)
# z = cp.matmul(x, y)
# print(z)
# print(cp.array([4, 1, 1], ndmin=2).shape)


"""
测试4：矩阵拼接
"""
# a = cp.array([[1, 2], [3, 4]])
# b = cp.array([[5, 6], [7, 8]])
# c0 = cp.concatenate([a,b],axis=0)
# c1 = cp.concatenate([a,b],axis=1)
# c2 = cp.concatenate([a.flatten(),b.flatten()])
# print(c0)
# print(c1)
# print(c2)

"""
测试5：数值，数值向量，数值矩阵分别和矩阵点乘
"""
# a=cp.array(2)
# b=cp.array([2])
# c=cp.array([[2]])
# x = cp.array([[1, 2], [3, 4]])
# y1=a*x
# y2=b*x
# y3=c*x
# print(y1)
# print(y2)
# print(y3)

"""
测试6：不同维度的数值之间的运算
"""
#
# a=cp.array([1])
# b=cp.array(2)
# c=a*b
# print(a,b,c)
# print(a==cp.array(1))
# x=cp.array([1,2,3])
# y=cp.array([[1],[2],[3]])
# print(x==y)

"""
测试7：multiply
"""
# a=cp.array([[1,2,3]])
# b=cp.array([1,2,3])
# c=cp.array([4,5,6])
# print(a.device,b.device)
# print(cp.multiply(a,c))
# print(cp.multiply(b,c))

"""
测试8：数据转换
"""

# # cupy->numpy
# x=np.array(5)
# x = cp.asnumpy(x)
# print(type(x))
# # numpy->cupy
# x = cp.array(x)
# print(type(x))


"""
测试9：argmax
"""
x=cp.array([[1,2],[3,4]])
print(x.argmax(axis=1))