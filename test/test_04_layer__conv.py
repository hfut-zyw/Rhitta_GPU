"""
卷积测试
卷积测试
卷积测试
"""

import rhitta.nn as nn
import cupy as cp



"""
测试1：conv2d()算子
"""
# x=nn.to_tensor(size=(4,4))
# x.set_value([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# kernel=nn.to_tensor(size=(2,2))
# kernel.set_value([[0,1],[2,3]])
# output=nn.conv2d(x,kernel,stride=2,padding=1)
# output.forward()
# print(output)
# print(output.get_jacobi(kernel))
# print(output.get_jacobi(x))


"""
测试2：Conv2D()神经元
"""
print("3输入通道--->>>1输出通道")

# x1=nn.to_tensor(size=(3,3))
# x1.set_value([[1,2,3],[5,6,7],[9,10,11]])
# x2=nn.to_tensor(size=(3,3))
# x2.set_value([[1,2,3],[5,6,7],[9,10,11]])
# x3=nn.to_tensor(size=(3,3))
# x3.set_value([[1,2,3],[5,6,7],[9,10,11]])
# image = [x1,x2,x3]
#
# model = nn.Conv2D(in_channels=3,out_channels=1,kernel_size=2)
# output = model(image)
# output.forward()
# x1.backward(output)
# print(output)
# print(cp.around(x1.grad,2))

# print("3输入通道--->>>5输出通道--->>>1输出通道")
# x1=nn.to_tensor(size=(3,3))
# x1.set_value([[1,2,3],[5,6,7],[9,10,11],[13,14,15]])
# x2=nn.to_tensor(size=(4,4))
# x2.set_value([[1,2,3],[5,6,7],[9,10,11],[13,14,15]])
# x3=nn.to_tensor(size=(4,4))
# x3.set_value([[1,2,3],[5,6,7],[9,10,11],[13,14,15]])
# image = [x1,x2,x3]
#
# model1 = nn.Conv2D(in_channels=3,out_channels=5,kernel_size=2)
# model2 = nn.Conv2D(in_channels=5,out_channels=1,kernel_size=2)
# output = model2(model1(image))
# output.forward()
# print(output)
# print(model1)
# print(model2)

"""
测试三：池化算子
"""
# print("MaxPooling（）")
# x=nn.to_tensor((4,4))
# x.set_value([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# output = nn.MaxPooling(x,stride=2)
# output.forward()
# print(output)
# print(output.get_jacobi(x))
#
# print("AveragePooling（）")
# x=nn.to_tensor((4,4))
# x.set_value([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# output = nn.AveragePooling(x,stride=2)
# output.forward()
# print(output)
# print(output.get_jacobi(x))


"""
测试四：池化神经元
"""

print("3通道池化")
channel1=nn.to_tensor((4,4))
channel1.set_value([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
channel2=nn.to_tensor((4,4))
channel2.set_value([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
channel3=nn.to_tensor((4,4))
channel3.set_value([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
image = [channel1,channel2,channel3]

model = nn.Pooling(in_channels=3,window_size=2,stride=2)
output = model(image)
for i in range(3):
    output[i].forward()
    print(image[i])
    print(output[i])

