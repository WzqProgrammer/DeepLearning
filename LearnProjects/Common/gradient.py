import numpy as np


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
    '''针对传输的参数 X 为矩阵时计算其相应的梯度'''
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    '''梯度下降法函数
    :param f: 需要优化的函数
    :param init_x: 起点
    :param lr: 步长/学习率
    :param step_num: 下降步数
    :return x, x_history: 使得函数结果最优的 x , x 在梯度下降过程中的变化值
    '''
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)