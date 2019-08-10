# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
    '''梯度值计算函数'''
    h = 1e-4
    grad = np.zeros_like(x)  #梯度值数组

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fx_h1 = f(x)  # f(x + h)

        x[idx] = tmp_val - h
        fx_h2 = f(x)  # f(x - h)
        grad[idx] = (fx_h1 - fx_h2) / (2*h)  #计算梯度值

        x[idx] = tmp_val

    return grad


def numerical_gradient(f, X):
    '''针对传输的参数X为矩阵时计算其相应的梯度'''
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def function_2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)  #输入x的参数行相加


def tangent_line(f, x):
    '''切线函数'''
    k = numerical_gradient(f, x)  #该点的梯度方向为斜率
    print(k)
    b = f(x) - k * x
    return lambda t: k*t + b


if __name__ == '__main__':
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)  #X, Y为 18*18 的矩阵

    X = X.flatten()  #将X, Y转化成以为数组
    Y = Y.flatten()

    grad = numerical_gradient(function_2, np.array([X, Y]))

    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1], angles = 'xy', color = '#666666')
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.draw()
    plt.show()