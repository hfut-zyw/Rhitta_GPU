import cupy as cp
from ..tensor import to_tensor
from . import Operator


# 将来有机会再把这串代码变成并行执行的，不然窗口划来划去太浪费时间了
def conv2d_parallel(u, v, m_out, n_out, s, value, padded, kernel):
    for i in range(m_out):
        for j in range(n_out):
            value[i, j] = cp.sum(cp.multiply(padded[s * i:s * i + u, s * j:s * j + v], kernel))


class conv2d(Operator):
    def __init__(self, *parents, **kargs):
        assert len(parents) == 2
        Operator.__init__(self, *parents, **kargs)
        self.kernel_size = self.parents[1].value.shape[0]
        self.stride = kargs.get("stride", 1)
        self.padding = kargs.get("padding", 0)
        self.padded = None

    def compute(self):
        """
        下方引入这么多变量是为了表示方便
        """
        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 卷积核
        m, n = data.shape  # 图像的尺寸
        u, v = kernel.shape  # 卷积核尺寸，虽然我们的卷积核是正方形的，但随时可以修改成矩形的
        p = self.padding  # 填充大小
        s = self.stride  # 卷积步长

        # 先把图像做个填充
        pad_m = m + 2 * p  # 填充后的图像边长
        pad_n = n + 2 * p
        self.padded = cp.array(cp.zeros((pad_m, pad_n)))
        self.padded[p:p + m, p:p + n] = data

        # 计算输出图像尺寸，并用零矩阵填充
        m_out = int((m + 2 * p - u) / s) + 1
        n_out = int((n + 2 * p - v) / s) + 1
        self.value = cp.array(cp.zeros((m_out, n_out)))

        # 开始计算value中每个方格中元素的值
        # for i in range(m_out):
        #     for j in range(n_out):
        #         self.value[i, j] = cp.sum(cp.multiply(self.padded[s * i:s * i + u, s * j:s * j + v], kernel))
        conv2d_parallel(u, v, m_out, n_out, s, self.value, self.padded, kernel)

    def get_jacobi(self, parent):
        data = self.parents[0].value  # 图像
        kernel = self.parents[1].value  # 卷积核
        m, n = data.shape  # 图像的尺寸
        u, v = kernel.shape  # 卷积核尺寸，虽然我们的卷积核是正方形的，但随时可以修改成矩形的
        p = self.padding  # 填充大小
        s = self.stride  # 卷积步长

        # 获取图像的填充尺寸，将卷积核扩充为这个尺寸方便找到跟图片对应相乘的部分

        pad_m = m + 2 * p  # 填充后的图像边长
        pad_n = n + 2 * p

        # 计算输出图像尺寸，并用零矩阵填充
        m_out = int((m + 2 * p - u) / s) + 1
        n_out = int((n + 2 * p - v) / s) + 1

        # 计算雅可比矩阵，将每个输出数值对矩阵的雅可比求出来，再排列一下即可
        # 卷积核在padded中的定位[s * i:s * i + u, s * j:s * j + v]，其中设[i，j]是对应的输出元位置
        jacobi = []
        if parent is self.parents[0]:  # 每个数值对图片求导，其实就是找出与图片对应内积的部分
            for i in range(m_out):
                for j in range(n_out):
                    # 把卷积核定位到它移动到的位置，其他地方用0填充
                    padded_kernel = cp.array(cp.zeros((pad_m, pad_n)))
                    padded_kernel[s * i:s * i + u, s * j:s * j + v] = kernel
                    # 卷积相当于把填充图像与填充卷积核作内积，也相当于把四周填充削去后作内积
                    # 内积的雅可比就是与它作内积的对象拉平，把与图像作内积的部分提取出来拉平即可
                    jacobi.append(padded_kernel[p:p + m, p:p + n].flatten())
        else:
            for i in range(m_out):
                for j in range(n_out):
                    # 与卷积核作内积的部分提取出来拉平就行了
                    jacobi.append(self.padded[s * i:s * i + u, s * j:s * j + v].flatten())
        return cp.array(jacobi, ndmin=2)


class MaxPooling(Operator):
    def __init__(self, *parents, **kargs):
        Operator.__init__(self, *parents, **kargs)
        self.stride = kargs.get("stride", 1)
        self.window_size = kargs.get("window_size", 2)

    def compute(self):
        # 跟卷积一样，记录一些参数方便表示与计算
        data = self.parents[0].value  # 图像
        m, n = data.shape  # 图像的尺寸
        u = self.window_size  # 滑动窗口大小
        v = self.window_size  # 滑动窗口大小，虽然我们的滑动窗口是正方形的，但为了和上面卷积符号表示一致，这里u=v=window_size
        s = self.stride  # 滑动窗口步长

        # 计算输出图像尺寸，并用零矩阵填充
        m_out = int((m - u) / s) + 1
        n_out = int((n - v) / s) + 1
        self.value = cp.array(cp.zeros((m_out, n_out)))

        # 开始计算value中每个方格中元素的值
        for i in range(m_out):
            for j in range(n_out):
                self.value[i, j] = cp.max(data[s * i:s * i + u, s * j:s * j + v])

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        jacobi = []
        data = self.parents[0].value  # 图像
        m, n = data.shape  # 图像的尺寸
        dim = m * n  # 输出每个数值对图像的雅可比形状为 1*dim
        u = self.window_size  # 滑动窗口大小
        v = self.window_size  # 滑动窗口大小，虽然我们的滑动窗口是正方形的，但为了和上面卷积符号表示一致，这里u=v=window_size
        s = self.stride  # 滑动窗口步长

        # 计算输出图像尺寸
        m_out = int((m - u) / s) + 1
        n_out = int((n - v) / s) + 1

        # 获得每个滑动窗口中最大值在原图中的位置
        # 从滑动窗口中取最大值，相当于在整张图大小的矩阵，在那个值的位置为1，其他地方为0，与整张图作内积
        for i in range(m_out):
            for j in range(n_out):
                window = data[s * i:s * i + u, s * j:s * j + v]
                padded_window = cp.array(cp.zeros((m, n)))
                padded_window[s * i:s * i + u, s * j:s * j + v] = window
                pos = cp.argmax(padded_window)
                tmp = cp.zeros(dim)
                tmp[pos] = 1
                jacobi.append(tmp)
        return cp.array(jacobi, ndmin=2)


class AveragePooling(Operator):
    def __init__(self, *parents, **kargs):
        Operator.__init__(self, *parents, **kargs)
        self.stride = kargs.get("stride", 1)
        self.window_size = kargs.get("window_size", 2)

    def compute(self):
        # 跟卷积一样，记录一些参数方便表示与计算
        data = self.parents[0].value  # 图像
        m, n = data.shape  # 图像的尺寸
        u = self.window_size  # 滑动窗口大小
        v = self.window_size  # 滑动窗口大小，虽然我们的滑动窗口是正方形的，但为了和上面卷积符号表示一致，这里u=v=window_size
        s = self.stride  # 滑动窗口步长

        # 计算输出图像尺寸，并用零矩阵填充
        m_out = int((m - u) / s) + 1
        n_out = int((n - v) / s) + 1
        self.value = cp.array(cp.zeros((m_out, n_out)))

        # 开始计算value中每个方格中元素的值
        for i in range(m_out):
            for j in range(n_out):
                self.value[i, j] = cp.sum(data[s * i:s * i + u, s * j:s * j + v]) / (u * v)

    def get_jacobi(self, parent):
        assert parent is self.parents[0]
        jacobi = []
        data = self.parents[0].value  # 图像
        m, n = data.shape  # 图像的尺寸
        dim = m * n  # 输出每个数值对图像的雅可比形状为 1*dim
        u = self.window_size  # 滑动窗口大小
        v = self.window_size  # 滑动窗口大小，虽然我们的滑动窗口是正方形的，但为了和上面卷积符号表示一致，这里u=v=window_size
        s = self.stride  # 滑动窗口步长

        # 计算输出图像尺寸
        m_out = int((m - u) / s) + 1
        n_out = int((n - v) / s) + 1

        # 获得每个滑动窗口中最大值在原图中的位置
        # 从滑动窗口中取最大值，相当于在整张图大小的矩阵，在那个值的位置为1，其他地方为0，与整张图作内积
        for i in range(m_out):
            for j in range(n_out):
                window = cp.ones((u, v))
                padded_window = cp.array(cp.zeros((m, n)))
                padded_window[s * i:s * i + u, s * j:s * j + v] = window
                jacobi.append(padded_window.flatten() / (u * v))
        return cp.array(jacobi, ndmin=2)


class embedding(Operator):
    def __init__(self, vocab=None, word_number=None):
        Operator.__init__(self, vocab)  # 其父节点是词典和单词，词典是底层叶子节点，单词从图上删除，它不需要训练
        self.vocab = vocab
        self.word_number = word_number

    def compute(self):
        self.value = self.vocab.value[self.word_number]

    def get_jacobi(self, parent):
        m = self.vocab.shape()[0]  # 词典长度
        n = self.vocab.shape()[1]  # 词典宽度，编码的长度
        jacobi = []
        for j in range(n):
            temp = cp.zeros(shape=(m, n))
            temp[self.word_number][j] = 1  # 编码向量的1个元素对词典的梯度
            jacobi.append(temp.flatten())  # 依次求出所有元素对词典的梯度，每个梯度都是一行
        jacobi = cp.array(jacobi).reshape(n, m * n)
        return jacobi
