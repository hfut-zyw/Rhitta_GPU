 # 本框架的内置数据集 

> 简单起见，所有数据集只有训练集

| 文件名                                                       | 文件说明            | 载入方法                                                     | 返回文件类型                                                 |
| ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| imdb.tsv                                                     | IMDB电影评论数据集  | ```imdb=nn.IMDBLoader()   data=imdb.load(r"D:\Rhitta_GPU\data\dataset")``` | (seqs,labels)  其中seqs是一个二维列表，列表里面每个列表都是一条句子的数字表示，labels是一个一维列表，代表每个句子的情感0或1 |
| Iris.csv                                                     | 鸢尾花数据集        | 简单数据集，使用pandas读取即可，格式非常中规中矩             | pandas.DataFrame类型                                         |
| sinx.csv                                                     | 人造sinx数据集      | 简单数据集，使用pandas读取即可，格式非常中规中矩             | pandas.DataFrame类型                                         |
| mnist-train-images.idx3-ubyte  mnist-train-labels.idx1-ubyte | Mnist手写数字数据集 | ```mnist=nn.MnistLoader() data=mnist.load(r"D:\Rhitta_GPU\data\dataset")``` | (x，labels) 其中x是60000x784的图片，labels是60000x1的标签，类型为cupy.ndarray |
|                                                              |                     |                                                              |                                                              |
|                                                              |                     |                                                              |                                                              |
|                                                              |                     |                                                              |                                                              |

