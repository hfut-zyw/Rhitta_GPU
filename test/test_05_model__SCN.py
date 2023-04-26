import sys

sys.path.append(r"C:\Users\Administrator\Desktop\Rhitta")

import cupy as cp
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import rhitta.nn as nn
onehot_encoder = OneHotEncoder(sparse_output=False)

dataloader = nn.MnistLoader()
data = dataloader.load(r"D:\Rhitta\data\dataset")
number_label_train = data[1]

train_x = cp.array(data[0])
train_labels = cp.array(onehot_encoder.fit_transform(data[1].reshape(-1, 1)))

conv1 = nn.Conv2D(in_channels=1, out_channels=3, kernel_size=5)
# self.layernorm1 = LayerNorm()
relu1 = nn.ReLU(in_channels=3)
pool1 = nn.Pooling(in_channels=3, window_size=2, stride=2)
conv2 = nn.Conv2D(in_channels=3, out_channels=5, kernel_size=3, stride=1)
# self.layernorm2 = LayerNorm()
relu2 = nn.ReLU(in_channels=5)
pool2 = nn.Pooling(in_channels=5, window_size=2, stride=2, mode="AveragePooling")
conv3 = nn.Conv2D(in_channels=5, out_channels=20, kernel_size=5, stride=1)
linear = nn.Linear(20, 10)

x = nn.to_tensor(size=(28, 28))
x.set_value(train_x[0, :].reshape(28, 28))
output1 = conv1(x)                    # (24, 24)
# output = self.layernorm1(output)
output2 = relu1(output1)              # (24, 24)
output3 = pool1(output2)              # (12, 12)
output4 = conv2(output3)              # (10, 10)
# output = self.layernorm2(output)
output5 = relu2(output4)              # (10, 10)
output6 = pool2(output5)              # (5, 5)
output7 = conv3(output6)              # (1, 1)
output8 = nn.Concat(*output7)         # (1, 20)
output9 = linear(output8)             # (1, 10)

output9.forward()
print(output1[0].shape())
print(output2[0].shape())
print(output3[0].shape())
print(output4[0].shape())
print(output5[0].shape())
print(output6[0].shape())
print(output7[0].shape())
print(output8.shape())
print(output9.shape())
print(output9.get_jacobi(output8).shape)
