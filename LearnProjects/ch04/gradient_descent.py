#coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from ch04.gradient_2d import numerical_gradient

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


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])

lr = 0.1
step_num = 20
x, x_history = gradient_descent(function_2, init_x, lr, step_num)

plt.plot([-5, 5], [0, 0], '--b')
plt.plot([0, 0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.xlim(-4.5, 4.5)
plt.xlabel('X0')
plt.xlabel('X1')
plt.show()

