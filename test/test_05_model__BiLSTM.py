import rhitta.nn as nn
import cupy

# 创建节点
x = [nn.to_tensor((1, 3)) for i in range(4)]
model = nn.BiLSTM(input_size=3, hidden_size=4, time_dimension=4)
h_0 = nn.to_tensor((1, 4))
c_0 = nn.to_tensor((1, 4))
h_1 = nn.to_tensor((1, 4))
c_1 = nn.to_tensor((1, 4))
# 构建计算图
output = model(x, h_0, c_0, h_1, c_1)
# 填坑
for i in range(4):
    x[i].set_value([1, 2, 3])
h_0.set_value([0, 0, 0, 0])
c_0.set_value([0, 0, 0, 0])
h_1.set_value([0, 0, 0, 0])
c_1.set_value([0, 0, 0, 0])
# 前向传播
output.forward()
print(output)
h_0.backward(output)