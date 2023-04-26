# ————————————实例一：鸢尾花分类————————————#
import sys
sys.path.append(r"C:\Users\Administrator\Desktop\Rhitta")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import rhitta

"""
数据准备
"""
data = pd.read_csv("../data/dataset/Iris.csv", header=0, index_col="Id")
#data = data.sample(len(data), replace=False,random_state=0)
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder(sparse_output=False)
number_label = labelencoder.fit_transform(data["Species"].values)
Y = onehotencoder.fit_transform(number_label.reshape(-1, 1))
X = data.iloc[:, 0:4].values
print("前3个样本的X为：")
print(X[0:3, :])
print("前3个样本的Y为：")
print(Y[0:3, :])


"""
构造模型（计算图）
"""
initlizer = rhitta.Normal_initializer(0, 1)
x = rhitta.to_tensor(size=(1, 4))
y = rhitta.to_tensor(size=(1, 3))
# x.set_value(X[0, :])
# y.set_value(Y[0, :])
W = rhitta.to_tensor(size=(4, 3), require_gradient=True, initializer=initlizer)
b = rhitta.to_tensor(size=(1, 3), require_gradient=True, initializer=initlizer)
output = rhitta.Add(rhitta.Matmul(x,W),b)
predict = rhitta.Softmax(output)
# 这个虽然也加入计算图中了，但是loss进行forward时并不会计算它的value值，backward时也不会进入这个节点求梯度
loss = rhitta.CrossEntropyLoss(output, y)



"""
临时测试
"""
# W.set_value(value=\
# [[-0.05627218 , 0.01191976,  0.27958796],\
#  [ 0.38072199 , 0.38916712, -0.93648468],\
#  [-0.93496966 ,-0.82606938 , 1.57472047],\
#  [ 0.44814738 ,-1.56665504 ,-0.42643429]])
# b.set_value([[-1.77872043 , 1.40379026,  0.46774446]])
# a = rhitta.to_tensor(dim=(1, 3))
# a.set_value([[1,2,3]])
# b=rhitta.Softmax(a)
# b.forward()
# print("x值为：",a.value)
# # print("W值为：",W.value)
# # print("b值为：",b.value)
# print("输出值为：",b.value)




"""
选择并初始化优化器
"""
learning_rate = 0.01
optimizer = rhitta.Adam(rhitta.default_graph, loss, learning_rate=learning_rate)



"""
开始训练、评估
"""

batch_size = 16
epochs = 200
for epoch in range(epochs):
    count = 0
    N = len(X)

    #训练
    for i in range(N):
        x.set_value(X[i, :])
        y.set_value(Y[i, :])
        optimizer.one_step()  # 执行一次，优化器中就累积了当前样本的梯度
        count += 1
        if count >= batch_size:  # 对于最后一批，数量不够16，是不能执行update的，一直到外层for循环结束
            optimizer.update()  # 执行16次后，利用16个样本loss的平均梯度更新W和b
            count = 0

    #每个epoch后评估模型的准确率
    pred = []
    for i in range(N):
        x.set_value(X[i, :])
        y.set_value(Y[i, :])
        predict.forward()
        pred.append(predict.value.A1)
    pred = np.array(pred).argmax(axis=1)
    accuracy = (number_label == pred).sum()/N
    print("epoch:{} , accuracy:{}".format(epoch+1,accuracy))

