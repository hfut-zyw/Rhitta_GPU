# ————————————nn.Linear模块测试————————————#
import sys

sys.path.append(r"D:\Rhitta")
import time
import cupy as cp
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import rhitta.nn as nn


"""
数据准备
"""
data = pd.read_csv("../data/dataset/Iris.csv", header=0, index_col="Id")
# data = data.sample(len(data), replace=False,random_state=0)
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(sparse_output=False)
number_label = labelencoder.fit_transform(data["Species"].values)
Y = cp.array(onehotencoder.fit_transform(number_label.reshape(-1, 1)))
X = cp.array(data.iloc[:, 0:4].values)
number_label = cp.array(number_label)
print("前3个样本的X为：")
print(X[0:3, :])
print("前3个样本的Y为：")
print(Y[0:3, :])

"""
构造模型
"""


class mymodel(nn.Module):

    # 实例化，初始化模型参数
    def __init__(self):
        super(mymodel, self).__init__()
        self.fc1 = nn.Linear(input_size=4, output_size=600)
        self.fc2 = nn.Linear(input_size=600, output_size=3)

    # 调用模型，创建计算图;实际上在Module基类中已经实现了，这里不需要再写了，
    # 下面写出来就是为了让你明白 init 中只是创建底层节点，而构建计算图的部分在call中实现，
    # 需要接收一个或多个节点，输出一个或多个节点
    # 但是这样写了后，就调用mymodel的call方法，不能再使用add_layer来增加网络单元了
    def __call__(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return x


"""
选择损失函数，构造计算图
"""
zyw = mymodel()  # 实例化，初始化模型参数
print(zyw)
x = nn.to_tensor(size=(1, 4))
y = nn.to_tensor(size=(1, 3))
output = zyw(x)  # 把x丢进来，构造计算图
predict = nn.Softmax(output)
loss = nn.CrossEntropyLoss(output, y)  # 把y和刚刚的输出节点丢进来，构造完整的计算图

"""
选择并初始化优化器
"""
learning_rate = 0.01
optimizer = nn.Adam(nn.default_graph, loss, learning_rate=learning_rate)

"""
开始训练、评估
"""

batch_size = 16
epochs = 100
start=time.time()
for epoch in range(epochs):
    count = 0
    N = len(X)

    # 训练
    for i in range(N):
        x.set_value(X[i, :])
        y.set_value(Y[i, :])
        optimizer.one_step()  # 执行一次，优化器中就累积了当前样本的梯度
        count += 1
        if count >= batch_size:  # 对于最后一批，数量不够16，是不能执行update的，一直到外层for循环结束
            optimizer.update()  # 执行16次后，利用16个样本loss的平均梯度更新W和b
            count = 0

    # 每个epoch后评估模型的准确率
    pred = []
    for i in range(N):
        x.set_value(X[i, :])
        y.set_value(Y[i, :])
        predict.forward()
        pred.append(predict.value.flatten())
    pred = cp.array(pred).argmax(axis=1)
    accuracy = (number_label == pred).sum() / N
    print("epoch:{} , accuracy:{}".format(epoch + 1, accuracy))
end=time.time()
print(end-start)