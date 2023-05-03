"""
损失函数功能测试
损失函数功能测试
损失函数功能测试
"""
import rhitta as rt
import cupy as cp

# 由于采用静态图机制，没有销毁计算图功能，需要使用 Ctrl+/ 把其他部分注释掉，只运行其中一个测试


"""
测试1：BinaryClassLoss()
predicts: 1x1  
labels:  1x1  
"""

# print("====***测试1：BinaryClassLoss()***====", "\n")
# 
# x = rt.to_tensor(size=(1, 1))
# x.set_value([0.1])
# labels = rt.to_tensor(size=(1, 1))
# labels.set_value([1])
# loss=rt.BinaryClassLoss(x,labels)
# loss.forward()
# print("loss的值为:", "\n", loss.value, "\n")
# x.backward(loss)
# print("predicts的梯度：", "\n", x.grad, "\n")


"""
测试2：CrossEntropyLoss()
features: 1xc , c为类别数         
label：   1xc ，one-hot vector   
features是c个类别的得分，不需要转换成c个类别的概率，这里softmax内置在loss里面了
"""

# print("====***测试2：CrossEntropyLoss()***====", "\n")
#
# x = rt.to_tensor(size=(1, 3))  # 样本在3个类别上的得分
# x.set_value(cp.array([[-1, 1, 3]]))
# label = rt.to_tensor(size=(1, 3))  # one-hot vector
# label.set_value(cp.array([0, 0, 1]))
# loss=rt.CrossEntropyLoss(x,label)
# loss.forward()
# print("loss的值为:", "\n", loss.value, "\n")
# x.backward(loss)
# print("features的梯度：", "\n", x.grad, "\n")


"""
测试2：MSELoss()
    features: 1xd   (parents[0])
    label：   1xd   (parents[1])
"""

print("====***测试3：MSELoss()***====", "\n")

x = rt.to_tensor(size=(1, 3))
x.set_value(cp.array([[1, 2, 3]]))
label = rt.to_tensor(size=(1, 3))
label.set_value(cp.array([6, 5, 4]))
loss = rt.MSELoss(x, label)
loss.forward()
print("loss的值为:", "\n", loss.value, "\n")
x.backward(loss)
print("features的梯度：", "\n", x.grad, "\n")
