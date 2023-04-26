import rhitta.nn as nn

x=nn.to_tensor((1,4))
x.set_value([1,2,3,4])
W = nn.to_tensor((4,3))
W.set_value([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
y=nn.Matmul(x,W)
y.forward()
x.backward(y)
W.backward(y)
print(x)
print(W)
print(y)