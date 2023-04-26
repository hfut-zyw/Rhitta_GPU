"""
激活函数功能测试
激活函数功能测试
激活函数功能测试
"""
import rhitta as rt
import cupy as cp

# 由于采用静态图机制，没有销毁计算图功能，需要使用 Ctrl+/ 把其他部分注释掉，只运行其中一个测试




"""
测试1：Logistic()

激活函数说明
对于二分类问题：输出为单数值，将单数值变成0~1之间的概率值
对于中间层激活：支持任何形式的向量，矩阵（因为它是逐元素函数，jacobi简单）
"""

# x = rt.to_tensor(size=(2, 2))
# x.set_value([[1, 2], [3, 4]])
# y = rt.Logistic(x)
# y.forward()
# x.backward(y)
# print(x)
# print(y)


"""
测试2：Tanh()

激活函数说明
对于二分类问题：输出为单数值，将单数值变成0~1之间的概率值
对于中间层激活：支持任何形式的向量，矩阵（因为它是逐元素函数，jacobi简单）
"""

# x = rt.to_tensor(size=(2, 2))
# x.set_value([[1, 2], [3, 4]])
# y = rt.Tanh(x)
# y.forward()
# x.backward(y)
# print(x)
# print(y)

"""
测试3：Softmax()

激活函数说明：
矩阵的每一行各自使用Softmax
"""

# x = rt.to_tensor(size=(1, 4))  # 单个样本
# x.set_value([1, 2])
# y = rt.Softmax(x)
# y.forward()
# x.backward(y)
# print(x)
# print(y)
#
# x = rt.to_tensor(size=(2, 2))
# x.set_value([[1, 2], [3, 4]])
# y = rt.Softmax(x)
# y.forward()
# x.backward(y)
# print(x)
# print(y)

"""
测试3：Relu()
"""

x = rt.to_tensor(size=(2, 2))
x.set_value([[1, 2], [-3, -4]])
y = rt.Relu(x)
y.forward()
x.backward(y)
print(x)
print(y)