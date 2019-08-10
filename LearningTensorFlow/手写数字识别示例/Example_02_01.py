import matplotlib.pyplot as plt
import numpy as np
from time import time
import tensorflow as tfa
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tfa.compat.v1

ckpt_dir = './ckpt_dir/'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

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

# 使用交叉熵作为损失函数，这个方式会造成log(0)值为NaN，最后数据不稳定的问题
# loss_function = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# TensorFlow提供了结合Softmax的交叉熵损失函数定义方法，其中参数为没有经过Softmax处理的数据
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=forward, labels=y))

# 定义准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 输入的样本图像加入summary
image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
tf.summary.image('input', image_shaped_input, 10)  # 10为最大输出显示图像

# 前向输出值以直方图显示
tf.summary.histogram('forward', forward)

# loss损失值和accuracy准确率以标量显示
tf.summary.scalar('loss', loss_function)
tf.summary.scalar('accuracy', accuracy)

# 设置训练参数
train_epochs = 40
batch_size = 50
total_size = int(mnist.train.num_examples/batch_size)
display_step = 1
learning_rate = 0.01
save_step = 5  # 保存粒度

# 选择优化器
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

# 记录训练开始时间
startTime = time()

# 声明完所有变量后，调用该保存函数
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 合并所有summary
merged_summary_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs/', sess.graph)  # 创建写入符

for epoch in range(train_epochs):
    for batch in range(total_size):
        xs, ys = mnist.train.next_batch(batch_size)  # 读取批次数据
        sess.run(optimizer, feed_dict={x:xs, y:ys})  # 执行批次训练

        # 生成summary
        summary_str = sess.run(merged_summary_op, feed_dict={x:xs, y:ys})
        writer.add_summary(summary_str, epoch)  # 将summary写入文件

    # total_batch个批次训练完成后，使用验证数据计算误差与准确率
    loss, acc = sess.run([loss_function, accuracy],
                         feed_dict={x:mnist.validation.images,
                                    y:mnist.validation.labels})

    if (epoch+1) % display_step == 0:
        print("Train Epoch:", "%02d" % (epoch+1),
              "Loss=","{:.9f}".format(loss), "Accuracy=", "{:.4f}".format(acc))

    if (epoch+1) % save_step == 0:
        # 存储模型
        saver.save(sess, os.path.join(ckpt_dir, 'mnist_h256_64_model_{:06d}.ckpt'.format(epoch+1)))
        print('mnist_h256_64_model_{:06d}.ckpt saved'.format(epoch+1))


saver.save(sess, os.path.join(ckpt_dir, 'mnist_h256_64_model.ckpt'))
print("Model saved")
# 显示运行总时间
duration = time() - startTime
print("Train Finished takes:", "{:.2}".format(duration))


def print_predict_errs(labels, prediction):
    """
    打印出出错的预测结果
    :param labels: 标签列表
    :param prediction: 预测值列表
    :return:
    """
    count = 0
    compare_lists = (prediction==np.argmax(labels, 1))
    err_lists = [i for i in range(len(compare_lists)) if compare_lists[i]==False]
    for x in err_lists:
        print("index="+str(x)+" 标签值=", np.argmax(labels[x]), " 预测值=", prediction[x])
        count += 1

    print("总计："+str(count))


prediction_result = sess.run(tf.argmax(pred, 1), feed_dict={x:mnist.test.images})
print_predict_errs(labels=mnist.test.labels, prediction=prediction_result)