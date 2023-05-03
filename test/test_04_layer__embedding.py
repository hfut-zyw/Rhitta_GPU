import rhitta.nn as nn
import cupy as cp


"""
测试1： embedding算子
"""
# initializer=nn.Normal_initializer(0,1)
# vocab = nn.to_tensor(size=(4,3),initializer=initializer)
# word_number=2
#
# word_embedding = nn.embedding(vocab=vocab,word_number=word_number)
#
# word_embedding.forward()
# vocab.backward(word_embedding)
#
# print("词典的值，单词编码后，单词对词典的梯度：",vocab)
# print("单词编码后是：",word_embedding)


"""
测试2：Embedding神经元
"""

seq=[2,1,3,2]
embedding_layer = nn.Embedding(5,3,paddingidx=0)

seq_embedding = embedding_layer(seq)

for i in range(4):
    seq_embedding[i].forward()
    embedding_layer.vocab.backward(seq_embedding[i])
print("词典如下：")
print(embedding_layer.vocab)
print("句子的第一个单词编码如下：")
print(seq_embedding[0])


