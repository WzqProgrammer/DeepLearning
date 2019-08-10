#coding: utf-8
import numpy as np
import matplotlib.pylab as plt
import Common.math_functions as mf

def function_1(x):
    return 0.01*x**2 + 0.1*x

def tangent_line(f, x):
    '''切线函数'''
    k = mf.numerical_diff(f, x)  #切线斜率
    print(k)
    y = f(x) - k * x  #切线截距
    return lambda t: k*t + y    #得出切线方程

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')

tf = tangent_line(function_1, 5)  #得出函在x=5时的切线表达式
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()

