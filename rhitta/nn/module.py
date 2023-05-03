# -*- coding: utf-8 -*-
"""
Created on 2023/4/2
@author: hfut-zyw
"""
import abc
from .container import *


class Module:
    """
    所有神经网络的基类
    目前这个类就是简单的复合函数，还未添加更多功能，有待进一步完善，也有可能就撂在这不完善了
    """

    def __init__(self):
        # 模型参数,给子类用的，基类通过prapmeters()函数接口获取所有参数
        self.params = dict()
        self.num_params = 0
        # 每创建一个网络，送入container容器，container相当于管理员
        self.container = default_container

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        子类实现构建计算图
        """
        # i = 0
        # for layer in self.container.layers:
        #     if i == 0:
        #         outputs = layer(*args)
        #         i += 1
        #     else:
        #         outputs = layer(outputs)
        # return outputs

    """
    以下是外部接口，通过模型访问一些基本属性
    
    """

    # 外部接口，不写call的情况下，可以随时增加神经元
    def add_layer(self, layer):
        self.container.layers.append(layer)

    # 外部接口，返回模型神经元列表
    def layers(self):
        return self.container.layers

    # 外部接口，返回模型参数列表
    def parameters(self):
        parameters = []
        for layer in self.container.layers:
            for key in layer.params.keys():
                parameters.append(layer.params[key])
        return parameters

    # 内部接口，统计模型所有参数个数,给打印输出接口使用
    def _total_params_number(self):
        cout = 0
        for layer in self.container.layers:
            cout += layer.num_params
        return cout

    # 外部接口，打印模型
    def __str__(self):
        print("Model:")
        k = len(self.container.layers)
        for i in range(k):
            print("Layer {}: {}".format(i + 1, self.container.layers[i]))
        return "Total params：{}".format(self._total_params_number())
