# MNIST手写数字识别

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tfa
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tfa.compat.v1

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print("训练集 train 数量：", mnist.train.num_examples,
#       ", 验证集 validation 数量：", mnist.validation.num_examples,
#       ", 测试机 test 数量：", mnist.test.num_examples)
#
# print("train images shape: ", mnist.train.images.shape,
#       "labels shape: ", mnist.train.labels.shape)

# def plt_image(image):
#     plt.imshow(image.reshape(28, 28), cmap='binary')
#     plt.show()
#
# plt_image(mnist.train.images[1])


# num = np.argmax(mnist.train.labels[1])  # 获取该数组中最大值的索引
# print(mnist.train.labels[1])
# print(num)

batch_images_xs, batch_labels_ys = mnist.train.next_batch(batch_size=10)

# mnist 中每张图片共有28*28=784个像素点
x = tf.placeholder(tf.float32, [None, 784], name='X')
# 0-9 一共10个数字=>10个类别
y = tf.placeholder(tf.float32, [None, 10], name='Y')

# 定义变量
W = tf.Variable(tf.random_normal([784, 10]), name='W')
b = tf.Variable(tf.zeros([10]), name='b')

# 用单个神经元构建神经网络
forward = tf.matmul(x, W) + b  # 前向计算

pred = tf.nn.softmax(forward)  # Softmax分类

# 设置训练参数
train_epochs = 50  # 训练轮数
batch_size = 100  # 单次训练样本数（批次大小）
total_batch = int(mnist.train.num_examples/batch_size)   # 一轮训练有多少批次
display_step = 1  # 显示粒度
learning_rate = 0.01  # 学习率

# 定义损失函数
loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# 检查预测类别tf.argmax(pred, 1)与实际类别tf.argmax(y, 1)的匹配情况
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

# 准确率，将布尔值转化为浮点数，并计算平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 开始训练
for epoch in range(train_epochs):
    for batch in range(total_batch):
        xs, ys = mnist.train.next_batch(batch_size)  # 读取批次数据
        sess.run(optimizer, feed_dict={x:xs, y:ys})  # 执行批次训练

    # total_batch个批次训练完后，使用验证数据计算误差与准确率，验证集没有分批
    loss, acc = sess.run([loss_function, accuracy], feed_dict={x:mnist.validation.images, y:mnist.validation.labels})

    # 打印训练过程中的详细信息
    if (epoch+1) % display_step == 0:
        print("Train Epoch:", '%02d' % (epoch+1), "Loss:", "{:.9}".format(loss), " Accuracy=", "{:.4}".format(acc))


print("Train Finished")

# 由于pred预测结果为one-hot编码格式，所以需要转换为0-9数字
prediction_result = sess.run(tf.argmax(pred, 1), feed_dict={x:mnist.test.images})

def plot_images_labels_prediction(images, labels, prediction, index, num=10):
    """
    :param images: 图像列表
    :param labels: 标签列表
    :param prediction: 预测值列表
    :param index: 从第index个开始显示
    :param num: 缺省一次显示10幅
    :return:
    """
    fig = plt.gcf()  # 获取当前图表 Get Current Figure
    fig.set_size_inches(10, 12)  # 1英寸等于2.54cm
    if num > 25:
        num = 25    # 最多显示25个子图

    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)  # 获取当前要处理的子图

        ax.imshow(np.reshape(images[index], (28, 28)))   # 显示第index个图像

        title = "label=" + str(np.argmax(labels[index]))  # 构建该图上要显示的title

        if len(prediction) > 0:
            title += ", predict=" + str(prediction[index])

        ax.set_title(title, fontsize=10)   # 显示图上的title信息/
        ax.set_xticks([])    # 不显示坐标轴
        ax.set_yticks([])
        index += 1

    plt.show()

plot_images_labels_prediction(mnist.test.images, mnist.test.labels, prediction_result, 10, 25)