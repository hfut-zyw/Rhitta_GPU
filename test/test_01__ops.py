"""
基本运算功能测试
基本运算功能测试
基本运算功能测试
"""
import rhitta as rt
import cupy as cp

# 由于采用静态图机制，没有销毁计算图功能，需要使用 Ctrl+/ 把其他部分注释掉，只运行其中一个测试


"""
测试1：矩阵加法
"""

# print("===***测试1-1：矩阵加矩阵***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# y = rt.to_tensor(size=(2, 2))
# x.set_value([[1, 2], [3, 4]])
# y.set_value([[1, 1], [1, 1]],)
# z = rt.Add(x, y)
# z.forward()
# x.backward(z)
# y.backward(z)
#
# print(x)
# print(y)
# print(z)

# -------------------------------------------------------------------------------

# print("===***测试1-2：矩阵加数值***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# y = rt.to_tensor(size=(1, 1))
# x.set_value([[1, 2], [3, 4]])
# y.set_value(5)
# z = rt.Add(x, y)
# z.forward()
# x.backward(z)
# y.backward(z)
#
# print(x)
# print(y)
# print(z)

# -------------------------------------------------------------------------------

# print("===***测试1-3：数值加矩阵***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# y = rt.to_tensor(size=(1, 1))
# x.set_value([[1, 2], [3, 4]])
# y.set_value(5)
# z = rt.Add(y, x)
# z.forward()
# x.backward(z)
# y.backward(z)
#
# print(x)
# print(y)
# print(z)


"""
测试2：矩阵减法
"""

# print("===***测试2-1：矩阵减矩阵***===","\n")
#
# x = rt.to_tensor(size=(2, 2))
# y = rt.to_tensor(size=(2, 2))
# x.set_value([[1, 2], [3, 4]])
# y.set_value([[1, 1], [1, 1]],)
# z = rt.Sub(x, y)
# z.forward()
# x.backward(z)
# y.backward(z)
#
# print(x)
# print(y)
# print(z)

# -------------------------------------------------------------------------------

# print("===***测试2-2：矩阵减数值***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# y = rt.to_tensor(size=(1, 1))
# x.set_value([[1, 2], [3, 4]])
# y.set_value(5)
# z = rt.Sub(x, y)
# z.forward()
# x.backward(z)
# y.backward(z)
#
# print(x)
# print(y)
# print(z)

# -------------------------------------------------------------------------------

# print("===***测试2-3：数值减矩阵***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# y = rt.to_tensor(size=(1, 1))
# x.set_value([[1, 2], [3, 4]])
# y.set_value(5)
# z = rt.Sub(y, x)
# z.forward()
# x.backward(z)
# y.backward(z)
#
# print(x)
# print(y)
# print(z)


"""
测试3：矩阵乘法
"""

# print("===***测试3-1：矩阵乘矩阵***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# w = rt.to_tensor(size=(2, 3))
# x.set_value([[1, 2], [3, 4]])
# w.set_value([[1, 4, 2], [3, 5, 1]])
# output = rt.Matmul(x,w)
#
# output.forward()
# x.backward(output)
# w.backward(output)
#
# print("x的值为:", "\n", x.value, "\n")
# print("w的值为:", "\n", w.value, "\n")
# print("output的值为:", "\n", output.value, "\n")
#
# print("x的梯度：", "\n", x.grad, "\n")
# print("w的梯度：", "\n", w.grad, "\n")
#
# print("把向量拉直验证梯度的准确性", "\n")
# print("vec(x)", "\n", x.value.flatten().T, "\n")
# print("vec(output)", "\n", output.value.flatten().T, "\n")
# print("x.grad*vec(x)", "\n", cp.matmul(x.grad, x.value.flatten().T), "\n")

# -------------------------------------------------------------------------------

# print("===***测试3-2：矩阵乘数值***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# w = rt.to_tensor(size=(1, 1))
# x.set_value([[1, 2], [3, 4]])
# w.set_value(5)
# output = rt.Matmul(x,w)
#
# output.forward()
# x.backward(output)
# w.backward(output)
#
# print("x的值为:", "\n", x.value, "\n")
# print("w的值为:", "\n", w.value, "\n")
# print("output的值为:", "\n", output.value, "\n")
#
# print("x的梯度：", "\n", x.grad, "\n")
# print("w的梯度：", "\n", w.grad, "\n")
#
# print("把向量拉直验证梯度的准确性", "\n")
# print("vec(x)", "\n", x.value.flatten().T, "\n")
# print("vec(output)", "\n", output.value.flatten().T, "\n")
# print("x.grad*vec(x)", "\n", cp.matmul(x.grad, x.value.flatten().T), "\n")

# -------------------------------------------------------------------------------

# print("===***测试3-3：数值乘矩阵***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# w = rt.to_tensor(size=(1, 1))
# x.set_value([[1, 2], [3, 4]])
# w.set_value(5)
# output = rt.Matmul(w,x)
#
# output.forward()
# x.backward(output)
# w.backward(output)
#
# print("x的值为:", "\n", x.value, "\n")
# print("w的值为:", "\n", w.value, "\n")
# print("output的值为:", "\n", output.value, "\n")
#
# print("x的梯度：", "\n", x.grad, "\n")
# print("w的梯度：", "\n", w.grad, "\n")
#
# print("把向量拉直验证梯度的准确性", "\n")
# print("vec(x)", "\n", x.value.flatten().T, "\n")
# print("vec(output)", "\n", output.value.flatten().T, "\n")
# print("x.grad*vec(x)", "\n", cp.matmul(x.grad, x.value.flatten().T), "\n")


"""
测试4：矩阵除法 
"""
# print("===***测试5：矩阵除以数值***===", "\n")
# a = rt.to_tensor(size=(2, 2))
# a.set_value([[1, 2], [1, 2]])
# b = rt.to_tensor(size=(1, 1))
# b.set_value(2)
# c = a / b
# c.forward()
# a.backward(c)
# b.backward(c)
# print(a)
# print(b)
# print(c)



"""
测试5：矩阵点乘
"""

# print("===***测试5：矩阵点乘***===", "\n")
#
# x = rt.to_tensor(size=(2, 2))
# y = rt.to_tensor(size=(2, 2))
# x.set_value([[1, 2], [3, 4]])
# y.set_value([[1, 4], [3, 5]])
# output = rt.Multiply(x,y)
#
# output.forward()
# x.backward(output)
# y.backward(output)
#
# print("x的值为:", "\n", x.value, "\n")
# print("w的值为:", "\n", y.value, "\n")
# print("output的值为:", "\n", output.value, "\n")
#
# print("x的梯度：", "\n", x.grad, "\n")
# print("w的梯度：", "\n", y.grad, "\n")
#
# print("把向量拉直验证梯度的准确性", "\n")
# print("vec(x)", "\n", x.value.flatten().T, "\n")
# print("vec(output)", "\n", output.value.flatten().T, "\n")
# print("x.grad*vec(x)", "\n", cp.matmul(x.grad, x.value.flatten().T), "\n")


"""
测试6：求和
"""
# print("===***测试5：矩阵求和***===", "\n")
# x=rt.to_tensor(size=(2,2))
# x.set_value([[1,2],[1,2]])
# y=rt.sum(x)
# y.forward()
# x.backward(y)
# print(x)
# print(y)



"""
测试7：求平均
"""
# print("===***测试5：矩阵求平均***===", "\n")
# x=rt.to_tensor(size=(2,2))
# x.set_value([[1,2],[1,2]])
# y=rt.mean(x)
# y.forward()
# x.backward(y)
# print(x)
# print(y)




"""
测试8：矩阵乘方
"""
# print("===***测试8-1：矩阵平方***===", "\n")
# a=rt.to_tensor(size=(2,2))
# a.set_value([[1,2],[3,4]])
# b=a**2
# b.forward()
# a.backward(b)
# print(a)
# print(b)

# -------------------------------------------------------------------------------

# print("===***测试8-2：矩阵平方***===", "\n")
# a=rt.to_tensor(size=(2,2))
# a.set_value([[1,2],[3,4]])
# b=a**3
# b.forward()
# a.backward(b)
# print(a)
# print(b)


"""
测试9：数值开根号
"""
# print("===***测试5：数值开根号***===", "\n")
# x=rt.to_tensor(size=(1,1))
# x.set_value([4])
# y=rt.sqrt(x)
# y.forward()
# x.backward(y)
# print(x)
# print(y)

"""
测试10：多个节点拼接
"""
print("===***测试10：多个节点拼接***===", "\n")
a=rt.to_tensor(size=(1,1))
a.set_value([4])
b=rt.to_tensor(size=(2,2))
b.set_value([[1,2],[3,4]])
c=rt.to_tensor(size=(1,2))
c.set_value([3,8])

d=rt.Concat(a,b,c)
d.forward()
a.backward(d)
print(a)
print(d)