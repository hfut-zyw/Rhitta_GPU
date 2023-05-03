
import cupy as cp

from ...tensor import *
from ...operator import *
from ..module import *


class Embedding(Module):
    def __init__(self, numembeddings, embeddingdim, paddingidx=None, weightattr=None):
        Module.__init__(self)
        self.container.add_layer(self)

        self.numembeddings = numembeddings
        self.embeddingdim = embeddingdim
        self.vocab = weightattr
        self.weight = self.vocab # weight属性是为了和其他框架保持一致，方便查看，vocab是自己的习惯

        if self.vocab is None:
            initializer = Uniform_initializer(low=-0.1,high=0.1)
            self.vocab = to_tensor(size=(numembeddings,embeddingdim),require_gradient=True,initializer=initializer)
        if paddingidx is not None:
            padding = cp.zeros((1, embeddingdim)).flatten()
            self.vocab.value[paddingidx] = padding

        self.params["vocab"] = self.vocab
        # 参数模型更新的参数，就是词典
        self.num_params = numembeddings*embeddingdim

    def __call__(self, seq):
        # seq = [ word1,word2,word3,...,wordN],每个word是一个数字
        seq_embeddings = []
        for word_number in seq:
            seq_embeddings.append(embedding(self.vocab,word_number))
        return seq_embeddings

    def __str__(self):
        return "Embedding Layer({},{})   vocab_params：{} ".format(self.numembeddings,self.embeddingdim,self.num_params)



# 下面是废弃版本，运行不通！！
# class Embedding_old(Module):
#     """
#     注意，由于静态图不容易动态更改，这里的embedding相当于词典，通过查表给外面的
#     时间序列赋值，当外面的时间序列更新后，一定要调用Embedding实例的update方法更新词典
#
#     这个编码器有点问题，它是传入一个数字列表（句子），然后通过查表得到编码，其中表是随机生成的
#     把得到的编码作为输入节点，丢进模型训练，并更新输入节点，再通过编码器的update将更新后的节点编码传回编码器的词典
#     但是有一个问题：
#     一个batch_size运行完，应该用最后每个样本反向传播的平均梯度来更新每个样本的编码，但是此时模型输入处的节点里面
#     的值是最后一个样本单词的编码，如果用optimizer更新，只会更新最后一个单词，发生这种问题的原因是因为这个需要更新参数
#     的节点，每次都是变化的，不像W这种固定参数节点
#     深挖的话问题应该不止这一点，比如更新词典，需要记录每个样本的句子，但是update方法里只记录了最后一个句子
#
#     embedding应该这样设计：
#     把词典直接当成一个可训练的参数，每次反向传播求的是对整个词典的梯度，
#     难点就是求雅可比，求每个单词节点对词典的雅可比（不能求句子对词典的雅可比，因为这里句子不是节点，句子当节点的话多一维，雅可比更难求）
#     单词对词典的雅可比，实际就是把词典中单词所在位置变成1，其他位置变成0.
#     并且这个写在底层算子当中
#     """
#     def __init__(self, numembeddings, embeddingdim, paddingidx=None, weightattr=None):
#         Module.__init__(self)
#         self.container.add_layer(self)
#
#         self.numembeddings = numembeddings
#         self.embeddingdim = embeddingdim
#         self.vocab = weightattr
#         self.weight = self.vocab # weight属性是为了和其他框架保持一致，方便查看，vocab是自己的习惯
#         if self.vocab is None:
#             self.vocab = np.random.uniform(low=-0.1, high=0.1, size=(numembeddings, embeddingdim))
#         if paddingidx is not None:
#             padding = np.zeros((1, embeddingdim))
#             self.vocab[paddingidx] = padding
#
#         self.params["vocab"] = self.vocab
#         # 参数模型更新的参数，取决于输入时间序列的长度，并不是下面的词表参数量
#         self.num_params = numembeddings*embeddingdim
#
#         self.update_index = None
#         self.update_value = None
#
#     def __call__(self, inputs, outputs):
#         # inputs是一个一维数字列表，代表一句话的每个单词,outputs是接收固定坑位的，把数字转换为词向量后，放到外面计算图的坑位中
#         # 为什么不用weight？因为每次取出的位置都不一样，导致每次参与构建计算图的节点不一样，由于是静态图，无法动态更改计算图
#         self.update_index = inputs
#         self.update_value = outputs
#
#         for i in range(len(inputs)):
#             outputs[i].set_value(self.vocab[inputs[i]])  # 设置每个单词的编码向量
#         return outputs
#
#     def update(self):
#         # 一轮训练后，需要把单词更新到词表weight中
#         for i in range(len(self.update_index)):
#             self.vocab[self.update_index[i]]=self.update_value[i].value
#
#     def __str__(self):
#         return "Embedding Layer({},{})   vocab_params：{} ".format(self.numembeddings,self.embeddingdim,self.num_params)
