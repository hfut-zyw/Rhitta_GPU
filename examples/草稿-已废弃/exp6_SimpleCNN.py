import sys

sys.path.append(r"D:\Rhitta")
import time
import cupy as cp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import rhitta.nn as nn

"""
数据准备
"""
onehot_encoder = OneHotEncoder(sparse_output=False)

dataloader = nn.MnistLoader()
train_x, train_labels = dataloader.load(r"D:\Rhitta\data\dataset")
number_label_train = cp.array(train_labels)

train_labels = cp.array(onehot_encoder.fit_transform(train_labels.reshape(-1, 1)))
train_x = cp.array(train_x)
print(train_x.shape,train_x.device)
print(train_labels.shape,train_labels.device)

"""
选择模型，实例化
"""

mymodel = nn.SCN(in_channels=1)
print(mymodel)

"""
挖坑：
构造初始输入节点和标签节点：
x:一个to_tensor节点，28*28,训练时把图片填进去即可
y:一个to_tensor节点，1*10，训练时把one-hot标签填进去即可
"""
x = nn.to_tensor(size=(28, 28))
y = nn.to_tensor(size=(1, 10))

"""
选择损失函数，构造计算图
"""

output = mymodel(x)  # 把x丢进来，构造计算图
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

batch_size = 4
epochs = 30
start=time.time()
for epoch in range(epochs):
    count = 0
    N_train = 100

    # 训练
    for i in range(N_train):
        x.set_value(train_x[i, :].reshape(28, 28))
        y.set_value(train_labels[i, :])
        optimizer.one_step()  # 执行一次，优化器中就累积了当前样本的梯度
        count += 1
        if count >= batch_size:  # 对于最后一批，数量不够16，是不能执行update的，一直到外层for循环结束
            optimizer.update()  # 执行16次后，利用16个样本loss的平均梯度更新W和b
            count = 0

    # 每个epoch后评估模型的准确率
    N_test = 100
    pred = []
    for i in range(N_test):
        x.set_value(train_x[i, :].reshape(28, 28))
        y.set_value(train_labels[i, :])
        predict.forward()
        pred.append(predict.value.flatten())
    pred = cp.array(pred).argmax(axis=1)
    accuracy = (number_label_train[0:N_test] == pred).sum() / N_test
    print("epoch:{} , accuracy:{}".format(epoch + 1, accuracy))
end=time.time()
print(end-start)