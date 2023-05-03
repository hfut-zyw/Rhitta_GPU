# ————————————使用LSTM构建自回归模型之气温预测————————————#
import sys
sys.path.append(r"C:\Users\Administrator\Desktop\Rhitta")
from math import sqrt
import numpy as np
import pandas as pd
import rhitta.nn as nn
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



"""
数据准备
"""
data = pd.read_csv("../data/sinx.csv", header=0, index_col=0)
print(data.head())
data_train = data.iloc[0:600, :]
data_test = data.iloc[600:, :]
print(data_train.values.shape, data_test.values.shape)

x_train = data_train[["x_i", "x_i+1", "x_i+2", "x_i+3"]].values
y_train = data_train["label=x_i+4"].values
x_test = data_test[["x_i", "x_i+1", "x_i+2", "x_i+3"]].values
y_test = data_test["label=x_i+4"].values
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

"""
选择模型,并初始化（SRN模型已经构造好了，用RNNCell实现的,这里直接使用即可）
"""
initializer = nn.Normal_initializer(0, 1)
encoder = nn.LSTM(input_size=1, hidden_size=3, time_dimension=4)
# W,b是用于将输出经过一个线形层，将维度变成标签维度
W = nn.to_tensor(size=(3, 1), require_gradient=True, initializer=initializer)
b = nn.to_tensor(size=(1, 1), require_gradient=True, initializer=initializer)

"""
挖坑：
h_0:一个to_tensor节点，用于接收初始状态
c_0:一个to_tensor节点，用于接收初始状态
inputs:一个列表，存放时间序列节点，每个节点都要提前初始化占好坑位
y:一个to_tensor节点，用于接收标签
"""

h_0 = nn.to_tensor(size=(1, 3))
c_0 = nn.to_tensor(size=(1, 3))
inputs = [nn.to_tensor(size=(1, 1)) for i in range(4)]
y = nn.to_tensor(size=(1, 1))

"""
构造计算图,选择损失函数
"""

h_out = encoder(inputs, h_0,c_0)
output = h_out * W + b
loss = nn.MSELoss(output, y)  # 把y和刚刚的输出节点丢进来，构造完整的计算图

"""
选择并初始化优化器
"""
learning_rate = 0.010531415926
optimizer = nn.Adam(nn.default_graph, loss, learning_rate=learning_rate)

"""
开始训练、评估
"""

batch_size = 16
epochs = 30
for epoch in range(epochs):
    count = 0
    N_train = 600

    # 填坑并训练
    for i in range(N_train):
        for j in range(4):
            inputs[j].set_value(x_train[i, j])
        h_0.set_value(np.zeros((1, 3)))
        c_0.set_value(np.zeros((1, 3)))
        y.set_value(y_train[i])
        optimizer.one_step()  # 执行一次，优化器中就累积了当前样本的梯度
        count += 1
        if count >= batch_size:  # 对于最后一批，数量不够16，是不能执行update的，一直到外层for循环结束
            optimizer.update()  # 执行16次后，利用16个样本loss的平均梯度更新W和b
            count = 0

    # 每个epoch后评估模型的平均平方损失
    acc_loss = 0
    N_test = 396
    for i in range(N_test):
        for j in range(4):
            inputs[j].set_value(x_test[i, j])
        h_0.set_value(np.zeros((1, 3)))
        y.set_value(y_test[i])
        loss.forward()
        acc_loss += loss.value.getA()[0][0]
    average_loss = acc_loss / N_test
    print("epoch:{} , average_loss:{:0.5f}".format(epoch + 1, sqrt(average_loss)))

"""
绘制图像
"""

T = 1000
time = np.linspace(0, 10, 1000)
fig = plt.figure(figsize=(60, 6))
axes = fig.subplots(nrows=2,ncols=1)


## 第一张图分别绘制真实值、带噪音的值和预测值：前20个点，方便观察
x_time = time[4:24]  # 时间轴
y_data = y_train[0:20] # 最后一列真实值
y_predict = []  # 预测值

for i in range(20):
    for j in range(4):
        inputs[j].set_value(x_train[i, j])
    h_0.set_value(np.zeros((1, 3)))
    y.set_value(y_train[i])
    output.forward()
    y_predict.append(output.value.getA()[0][0])

ax1 = axes[0]
ax1.plot(x_time, np.sin(x_time), "o", label='sin(x)')
ax1.plot(x_time, y_data, "b--", label='sin(x)+noise')
ax1.plot(x_time, y_predict, "g-", label='predict')
ax1.legend()

# 第二张图绘制测试集的最后100个点,在test中对应296~395
# 注意，test中0~395对应的真实时间轴是604~999；所以最后100个点真实时间轴900~999对应test中的296~395
x_time2 = time[900:1000]
y_data2 = y_test[296:]
y_predict2 = []
for i in range(100):
    for j in range(4):
        inputs[j].set_value(x_test[i+296, j])
    h_0.set_value(np.zeros((1, 3)))
    y.set_value(y_test[i+296])
    output.forward()
    y_predict2.append(output.value.getA()[0][0])

ax2 = axes[1]
ax2.plot(x_time2, np.sin(x_time2), "o", label='sin(x)')
ax2.plot(x_time2, y_data2, "b--", label='sin(x)+noise')
ax2.plot(x_time2, y_predict2, "g-", label='predict')
ax2.legend()
plt.show()
