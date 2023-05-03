"""
归一化测试
归一化测试
归一化测试
"""

import cupy as cp
import rhitta.nn as nn

"""
路线1：x到mean再到var再到y
"""

# eps = nn.to_tensor(size=(1, 1))
# eps.set_value(1e-7)
# x = nn.to_tensor(size=(1, 4))
# x.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))
#
# x_copy1 = nn.to_tensor(size=(1, 4))
# x_copy1.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))  # 计算方差的x节点，与x无关
#
# x_copy2 = nn.to_tensor(size=(1, 4))
# x_copy2.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))  # 分子的x节点，与x无关
#
# x_mean = nn.mean(x)                          # 均值节点,由x计算而来
#
# x_mean_copy = nn.to_tensor(size=(1, 1))         # 用来计算y的常量均值，与x无关，
# x_mean_copy.set_value(0.75)
#
# x_var = nn.mean((x_copy1-x_mean)**2)           # 方差节点，与x无关，由均值算来
#
# y = (x_copy2 - x_mean_copy) / nn.sqrt(x_var + eps)  # 归一化节点,从x经过mean，再经过var到达
#
# y.forward()
# x.backward(y)
# print("输入的值和梯度为：")
# print(x)
# print("输出的值和梯度为：")
# print(y)
# print("检查一下其他梯度")
# print("y对var的雅可比为：")
# x_var.backward(y)
# print(x_var)
# print("var对mean的雅可比为：")
# nn.default_graph.clear_grad()
# x_mean.backward(x_var)
# print(x_mean)
# print("故累乘结果为0，反向传播无误")


"""
路线2：x到u再到y
"""

# eps = nn.to_tensor(size=(1, 1))
# eps.set_value(1e-7)
# x = nn.to_tensor(size=(1, 4))
# x.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))
#
# x_copy = nn.to_tensor(size=(1, 4))
# x_copy.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))  # 分子的x节点，与x无关
#
# x_mean = nn.mean(x)  # 均值节点,由x计算而来
#
# x_var = nn.to_tensor(size=(1, 1))             # 方差节点，与x无关
# x_var.set_value(0.0125)
#
# y = (x_copy - x_mean) / nn.sqrt(x_var + eps)  # 归一化节点,从x到y只经过mean，没有其他路线
#
# y.forward()
# x.backward(y)
# print("输入的值和梯度为：")
# print(x)
# print("输出的值和梯度为：")
# print(y)


"""
路线3：x到var再到y
"""

# eps = nn.to_tensor(size=(1, 1))
# eps.set_value(1e-7)
# x = nn.to_tensor(size=(1, 4))
# x.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))
#
# x_copy = nn.to_tensor(size=(1, 4))
# x_copy.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))  # 分子的x节点，与x无关
#
# x_mean = nn.to_tensor(size=(1, 1))  # 均值节点,常量，与x无关
# x_mean.set_value(0.75)
#
# x_var = nn.mean((x-x_mean)**2)   # 方差节点，由x计算而来
#
# mean_copy = nn.to_tensor(size=(1, 1))  # 均值节点,常量，与x无关
# mean_copy.set_value(0.75)
#
# y = (x_copy - mean_copy) / nn.sqrt(x_var + eps)  # 归一化节点,从x到y只经过var，没有其他路线
#
# y.forward()
# x.backward(y)
# print("输入的值和梯度为：")
# print(x)
# print("输出的值和梯度为：")
# print(y)

"""
路线4：x直接到y
"""

# eps=nn.to_tensor(size=(1, 1))
# eps.set_value(1e-7)
# x = nn.to_tensor(size=(1, 4))
# x.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))
# x_mean = nn.to_tensor(size=(1, 1))            # 均值节点,与x无关
# x_mean.set_value(0.75)
# x_var = nn.to_tensor(size=(1, 1))             # 方差节点，与x无关
# x_var.set_value(0.0125)
# y = (x-x_mean)/nn.sqrt(x_var+eps)            # 归一化节点，由x与常量直接得到
#
# y.forward()
# x.backward(y)
# print("输入的值和梯度为：")
# print(x)
# print("输出的值和梯度为：")
# print(y)


"""
综合所有路线
"""
# print("对上面4条路线汇总，其中第1条路线，因为一维var对一维mean的导数为0，所以那条路线的梯度为0")
#
# a2 = cp.array([[-2.23605903, - 2.23605903, - 2.23605903, - 2.23605903],
#              [-2.23605903, - 2.23605903, - 2.23605903, - 2.23605903],
#              [-2.23605903, - 2.23605903, - 2.23605903, - 2.23605903],
#              [-2.23605903, - 2.23605903, - 2.23605903, - 2.23605903]])
#
# a3 = cp.array([[-4.02487406, 1.34162469, 4.02487406, -1.34162469],
#              [1.34162469, -0.44720823, -1.34162469, 0.44720823],
#              [4.02487406, -1.34162469, -4.02487406, 1.34162469],
#              [-1.34162469, 0.44720823, 1.34162469, -0.44720823]])
#
# a4 = cp.array([[8.94423613, 0., 0., 0.],
#              [0., 8.94423613, 0., 0.],
#              [0., 0., 8.94423613, 0.],
#              [0., 0., 0., 8.94423613]])
#
# # print(a1+a2+a3)
#
# print("直接构建整个计算图")
# eps=nn.to_tensor(size=(1, 1))
# eps.set_value(1e-7)
# x = nn.to_tensor(size=(1, 4))
# x.set_value(cp.array([[0.9, 0.7, 0.6, 0.8]]))
# x_mean = nn.mean(x)            # 均值节点
# x_var = nn.mean((x-x_mean)**2)             # 方差节点
# y = (x-x_mean)/nn.sqrt(x_var+eps)            # 归一化节点
#
# y.forward()
# x.backward(y)
# print("输入的值和梯度为：")
# print(x)
# print("输出的值和梯度为：")
# print(y)
#
# # cupy对照计算结果，仅用于核算前向传播是否正确
# print("Numpy对照输出结果", "\n")
# a = cp.array([[0.9, 0.7, 0.6, 0.8]])
# mean = cp.mean(a)
# var = cp.mean(cp.power(a - mean, 2))
# print("均值：", mean, "方差：", var, "\n")
# print("归一化向量", (a - mean) / cp.sqrt(var + 1e-7))
#
# model = nn.LayerNorm()
# output = model(x)
# output.forward()
# x.backward(output)
# print(output)
# print(x)


"""
多通道输入测试
"""

print("3输入--->>>3输出")
channel1 = nn.to_tensor(size=(4, 4))
channel1.set_value(cp.array([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2]]))
channel2 = nn.to_tensor(size=(4, 4))
channel2.set_value(cp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))
channel3 = nn.to_tensor(size=(4, 4))
channel3.set_value(cp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))
image = [channel1, channel2, channel3]

model = nn.LayerNorm()
output = model(image)
output[1].forward()
print(output[1])
print(output[1].get_jacobi(channel2))

a=cp.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
mean=cp.mean(a)
var = cp.mean(cp.power(a-mean,2))
print(mean,var,"\n")
print((a-mean)/cp.sqrt(var+1e-7))
