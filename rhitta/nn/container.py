# -*- coding: utf-8 -*-
"""
Created on 2023/4/2
@author: hfut-zyw
"""


class Container:
    """
    定义一个容器，用来存储模型的神经元，神经元的参数
    目前功能尚未完善
    """

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)


default_container = Container()  # 创建一个默认图容器
