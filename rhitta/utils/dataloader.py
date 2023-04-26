import numpy as np
import cupy as cp
import re


class MnistLoader(object):
    def __init__(self):
        pass

    def load(self, path):
        # 训练集
        with open("{}/mnist-train-images.idx3-ubyte".format(path), "rb") as f:
            f.seek(16)
            train_x = cp.fromfile(f, dtype="uint8").reshape(-1, 784)
        with open("{}/mnist-train-labels.idx1-ubyte".format(path), "rb") as f:
            f.seek(8)
            train_labels = cp.fromfile(f, dtype="uint8")
        return train_x, train_labels


class IMDBLoader:
    def __init__(self,max_seq_len=256):
        self.data = None
        self.dict = None
        self.max_seq_len = max_seq_len

    def load(self, path):
        IMDB = self.IMDB_loader(path)
        IMDB = self.IMDB_preprocessing(IMDB)
        IMDB_dict = self.IMDB_word2id_dict(IMDB)
        IMDB = self.IMDBDateset(IMDB, IMDB_dict)
        self.data = self.IMDB_cutandpad(IMDB,max_seq_len=self.max_seq_len)
        self.dict = IMDB_dict
        return self.data

    @staticmethod
    def IMDB_loader(path):
        with open("{}/imdb.tsv".format(path), "r", encoding="utf-8") as f:
            f.readline()  # 第一行没用，去掉
            data = []  # 用于存放每个样本
            for line in f:  # 取出后面的每一行
                # line=[id    sentiment    review]
                line = line.split("\t", maxsplit=2)  # 按制表符tab分隔每一行数据，返回一个列表
                # 处理后：line=[id,label,sentence]
                line = (line[2], line[1])  # 取出句子和标签，组成元组
                # line=(sentence,label)
                data.append(line)
                # data=[样本1，样本2，...样本25000]
        return data

    @staticmethod
    def IMDB_preprocessing(data):
        temp = []
        for sample in data:
            patten = re.compile(r"[a-zA-Z]+")  # 匹配字母
            seq = re.findall(patten, sample[0])
            temp.append((seq, int(sample[1])))  # 情感的0，1转化为整型
        return temp

    @staticmethod
    def IMDB_word2id_dict(data):
        # 构造词频词典
        word_freq_dict = dict()
        for sample in data:
            for word in sample[0]:
                if word not in word_freq_dict:
                    word_freq_dict[word] = 0
                word_freq_dict[word] += 1
        # 按词频由高到低排序
        word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)

        # 按照频率构造word2id词典,前两个字符比较特殊，后面说明
        word2id_dict = {"PAD": 0, "UNK": 1}
        for word, freq in word_freq_dict:
            id = len(word2id_dict)
            word2id_dict[word] = id
        return word2id_dict

    @staticmethod
    def IMDBDateset(data, word2id_dict):
        seqs = []
        labels = []
        for sample in data:
            seq = sample[0]
            label = sample[1]
            seq = [word2id_dict.get(word, word2id_dict["UNK"]) for word in seq]
            seqs.append(seq)
            labels.append(label)
        return seqs, labels

    @staticmethod
    def IMDB_cutandpad(data, max_seq_len):
        seqs, labels = data[0], data[1]
        # 截断
        temp = []
        for seq in seqs:
            seq = seq[:max_seq_len]
            temp.append(seq)
        # padding
        temp2 = []
        for seq in temp:
            seq = seq + [0] * (max_seq_len - len(seq))  # 默认pad字符为0
            temp2.append(seq)
        return temp2, labels
