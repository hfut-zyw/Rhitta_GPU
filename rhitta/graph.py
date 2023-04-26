# -*- coding: utf-8 -*-
"""
Created on 2023/3/26
@author: hfut-zyw
"""


class Graph:
    """
    定义一个Graph容器，用来存储计算图上的所有节点，方便顺序遍历
    """
    def __init__(self):
        self.nodes=[]          #用来保存所有节点
        self.parameters=[]     #用来保存模型中的所有参数节点

    #添加节点，在创建节点的时候调用
    def add_node(self,node):
        self.nodes.append(node)

    #添加参数，在参数初始化的时候调用
    def add_parameter(self):
        self.parameters.append(node)

    #重置所有节点的值
    def reset_value(self):
        for node in self.nodes:
            node.reset_value(recursive=False)   #一个一个删除，不要使用递归删除

    #清空所有节点的梯度值
    def clear_grad(self):
        for node in self.nodes:
            node.clear_grad()
    #这里可以直接用node.grad=None,为了体现graph只是辅助操作的容器，这里对于node的操作全部使用node自身的方法

default_graph=Graph()  #创建一个默认图容器