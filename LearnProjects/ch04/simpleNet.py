import sys, os
import numpy as np
from Common.gradient import numerical_gradient
from Common.math_functions import softmax, cross_entropy_error
sys.path.append(os.pardir)

class simpleNet:

    def __init__(self):
        """定义一个随机权重矩阵"""
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        """参数与权重矩阵相乘"""
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)  # 第一层神经单元计算
        y = softmax(z)    # 激活函数
        loss = cross_entropy_error(y, t)  # 损失函数计算

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dw = numerical_gradient(f, net.W)

print(dw)