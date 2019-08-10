import numpy as np
'''感知机逻辑门实现'''

def AND(x1, x2):
    '''与门'''
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    '''与非门'''
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    '''或门'''
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    '''异或门'''
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def step_function(x):
    '''阶跃函数'''
    return np.array(x > 0, dtype = np.int)


def sigmoid(x):
    '''针对二元分类问题的激活函数'''
    return 1/(1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    '''恒等函数'''
    return x


def softmax(a):
    '''针对多元分类问题的激活函数'''
    c = np.max(a)
    exp_a = np.exp(a - c)    #溢出策略
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp

    return y


def mean_square_error(y, t):
    '''均方误差'''
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    '''交叉熵误差'''
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy_error_onehot(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_not_onehot(y, t):
    if y.adim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size,), t] + 1e-7)) / batch_size


def numerical_diff(f, x):
    '''数值微分函数'''
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2 * h)

