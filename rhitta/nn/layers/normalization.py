from ...tensor import *
from ...operator import *
from ..module import *


class LayerNorm(Module):
    def __init__(self,eps=1e-7,gamma=1.0,beta=0.0):
        """
        可选参数，默认值看上面提示
        :param gamma: 缩放的参数，超参数，自己调节，这里为简单起见就不设置成可训练的参数了
        :param beta: 平移的参数
        """
        Module.__init__(self)
        self.container.add_layer(self)

        # 基本属性,3个无参数节点
        self.eps=to_tensor(size=(1, 1))
        self.gamma=to_tensor(size=(1, 1))
        self.beta=to_tensor(size=(1, 1))
        self.eps.set_value(eps)
        self.gamma.set_value(gamma)
        self.beta.set_value(beta)

        self.num_params=0

    @staticmethod
    def layernorm(x,eps,gamma,beta):
        x_mean = mean(x)                # 均值节点构造
        x_var = mean((x-x_mean)**2)     # 方差节点构造
        y = (x-x_mean)/sqrt(x_var+eps) # 归一化节点构造
        z = gamma*y+beta               # 缩放平移
        return z                       # 丢进来一个x节点，输出一个归一化后的节点

    # 计算图构造
    def __call__(self, inputs):
        if isinstance(inputs,list):
            out_feature_map = [self.layernorm(x,self.eps,self.gamma,self.beta) for x in inputs]
        else:
            out_feature_map = self.layernorm(inputs, self.eps, self.gamma, self.beta)
        return out_feature_map

    def __str__(self):
        return "LayerNorm()"

class LayerNorm2(Module):
    def __init__(self,eps=1e-7,gamma=1.0,beta=0.0):
        """
        可选参数，默认值看上面提示
        :param gamma: 缩放的参数，超参数，自己调节，这里为简单起见就不设置成可训练的参数了
        :param beta: 平移的参数
        """
        Module.__init__(self)
        self.container.add_layer(self)

        # 基本属性,3个无参数节点
        self.eps=to_tensor(size=(1, 1))
        self.gamma=to_tensor(size=(1, 1))
        self.beta=to_tensor(size=(1, 1))
        self.eps.set_value(eps)
        self.gamma.set_value(gamma)
        self.beta.set_value(beta)

        self.num_params=0

    @staticmethod
    def layernorm2(x,eps,gamma,beta):
        x_mean = mean_constant(x)  # 均值节点构造
        x_var = mean_constant((x - x_mean) ** 2)  # 方差节点构造
        y = (x - x_mean) / sqrt(x_var + eps)  # 归一化节点构造
        z = gamma * y + beta  # 缩放平移
        return z  # 丢进来一个x节点，输出一个归一化后的节点

    # 计算图构造
    def __call__(self, inputs):
        if isinstance(inputs,list):
            out_feature_map = [self.layernorm2(x,self.eps,self.gamma,self.beta) for x in inputs]
        else:
            out_feature_map = self.layernorm2(inputs, self.eps, self.gamma, self.beta)
        return out_feature_map

    def __str__(self):
        return "LayerNorm()"