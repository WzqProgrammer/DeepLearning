import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tfa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tfa.compat.v1

np.random.seed(5)  # 设置随机数种子

# 直接采用np生成等差数列的方法，生成100个点，每个点的取值在-1~1之间
x_data = np.linspace(-1, 1, 100)

# np.random.randn(d0, d1, ..., dn)是从标准正态分布中返回一个或多个样本值
y_data = 2 * x_data + 1.0 +np.random.randn(*x_data.shape) * 0.4


def model(x, w, b):
    return tf.multiply(x, w) + b


x = tf.placeholder(dtype=float, name='x')
y = tf.placeholder(dtype=float, name='y')
# 构建线性函数的斜率，变量w
w = tf.Variable(1.0, 'w0')

# 构建线性函数的截距，变量b
b = tf.Variable(0.0, 'b0')

pred = model(x, w, b)  # 通过模型预测结果

train_epochs = 10  # 迭代次数（训练轮数）

learning_rate = 0.05  # 学习率

step = 0  # 记录训练步数
loss_list = []  # 用于保存loss值的列表
display_step = 10   # 控制显示loss值的粒子

# 损失函数计算
loss_function = tf.reduce_mean(tf.square(y - pred))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss_function)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

# 开始训练，轮数为epoch，采用SGD随机梯度下降优化方法
for epoch in range(train_epochs):
    for xs, ys in zip(x_data, y_data):
        _, loss = sess.run([optimizer, loss_function], feed_dict={x:xs, y:ys})
        # 显示损失值loss
        # display_step:控制报告的粒度
        # 如，当display_step设为2，则会每训练2个样本输出一次损失值
        # 与超参数不同，修改display_step不会更改模型所学习的规律
        loss_list.append(loss)
        step += 1
        if step % display_step == 0:
            print("The Epoch:",'%02d' % (epoch+1), "Step: %03d" % (step), "loss=","{:.9f}".format(loss))

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    plt.plot(x_data, w0temp * x_data + b0temp)  # 画图

plt.plot(loss_list, 'r+')
plt.show()
