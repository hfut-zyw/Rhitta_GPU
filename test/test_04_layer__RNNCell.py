"""
RNNCell测试
RNNCell测试
RNNCell测试
"""

import rhitta.nn as nn
import cupy as cp

model = nn.RNNCell(input_size=4, hidden_size=3)
initializer = nn.Normal_initializer(0, 1)
x = nn.to_tensor(size=(1, 4), require_gradient=False, initializer=initializer)
h_0 = nn.to_tensor(size=(1, 3), require_gradient=False, initializer=initializer)
output, h = model(x, h_0)
output.forward()
model.params["W_x"].backward(output)
print("输出是：")
print(output)
print("模型的其中一个参数梯度是：")
print(cp.around(model.params["W_x"].value,2))
