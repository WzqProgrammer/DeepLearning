import numpy as np
from time import time
import tensorflow as tfa
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tfa.compat.v1
# 模型保存路径
ckpt_dir = './ckpt_dir/'

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784], 'X')   # 输入参数
y = tf.placeholder(tf.float32, [None, 10], 'Y')    # 输出参数

# 隐藏神经元的数量
H1_NN = 256
H2_NN = 64


def fcn_layer(inputs, input_dim, output_dim, activation=None):
    """
    全连接层函数
    :param inputs: 输入数据
    :param input_dim: 输入神经元数量
    :param output_dim: 输出神经元数量
    :param activation: 激活函数
    :return: 该层的输出结果
    """
    # 以截断正态分布的随机函数初始化
    W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    # 以0初始化b
    b = tf.Variable(tf.zeros([output_dim]))

    XWb = tf.matmul(inputs, W) + b  # 建立表达式：inputs * W + b

    if activation is None:  # 默认不使用激活函数
        outputs = XWb
    else:                   # 若传入激活函数，则对其输出结果进行变换
        outputs = activation(XWb)

    return outputs

h1 = fcn_layer(inputs=x, input_dim=784, output_dim=H1_NN, activation=tf.nn.relu)
h2 = fcn_layer(inputs=h1, input_dim=H1_NN, output_dim=H2_NN, activation=tf.nn.relu)
forward = fcn_layer(inputs=h2, input_dim=H2_NN, output_dim=10)
pred = tf.nn.softmax(forward)

# TensorFlow提供了结合Softmax的交叉熵损失函数定义方法，其中参数为没有经过Softmax处理的数据
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=forward, labels=y))

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 创建saver
saver = tf.train.Saver()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(ckpt_dir)

if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)   # 从已保存的模型中读取参数，模型读取最新的模型结果
    print("Restore model from " + ckpt.model_checkpoint_path)

print("Accuracy:", accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y:mnist.test.labels}))