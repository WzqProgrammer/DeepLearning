# 波士顿房价预测案例

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tfa
import pandas as pd
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tfa.compat.v1

# 读取数据文件
df = pd.read_csv('F:/PythonProjects/DeepLearning/data/boston.csv', header=0)

# 显示数据摘要描述信息
# print(df.describe())

df = df.values  # 获取df的值
df = np.array(df)  # 把df转换为np的数组格式

# 数据预处理，归一化，特征值/（最大特征值-最小特征值）
for i in range(12):
    df[:,i] = df[:,i]/(df[:,i].max() - df[:,i].min())

x_data = df[:,:12]  # 归一化后前12列特征数据
y_data = df[:,12]   # 最后1列标签数据

x = tf.placeholder(tf.float32, [None, 12], name='X')  # 12个特征数据（12列）
y = tf.placeholder(tf.float32, [None, 1], name='Y')   # 1个标签数据（1列）

# 定义一个命名空间
with tf.name_scope('Model'):
    # 初始化值为shape=(12, 1)的随机数
    w = tf.Variable(tf.random_normal([12, 1], stddev=0.01), name='W')

    # b 的初始值为1.0
    b = tf.Variable(1.0, name='b')

    # w和x是矩阵相乘，用matmul，不能用mutiply或*
    def model(x, w, b):
        return tf.matmul(x, w) + b

    # 预测计算操作，前向计算节点
    pred = model(x, w, b)

# 定义损失函数
with tf.name_scope('LossFunction'):
    loss_function = tf.reduce_mean(tf.pow(y - pred, 2))

train_epochs = 100  # 训练轮数
learning_rate = 0.01  # 学习率

sess = tf.Session()
init = tf.global_variables_initializer()

sess.run(init)

logdir = 'F:\PythonProjects\DeepLearning\LearningTensorFlow'

# 创建一个操作，用于记录损失值loss，后面再TensorBoard中的SCALARS栏中可见
sum_loss_op = tf.summary.scalar('loss', loss_function)

# 把所有需要记录摘要日志文件合并，方便一次性写入
merged = tf.summary.merge_all()

# 创建摘要writer，将计算图写入摘要文件，后面再TensorBoard中GRAPHS栏可见
writer = tf.summary.FileWriter(logdir, sess.graph)

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

loss_list = []  # 用于保存loss值的列表

for epoch in range(train_epochs):
    loss_sum = 0.0
    for xs, ys in zip(x_data, y_data):
        # placeholder与数据的shape需一致
        xs = xs.reshape(1, 12)
        ys = ys.reshape(1,1)

        _, summary_str, loss = sess.run([optimizer, sum_loss_op ,loss_function], feed_dict={x:xs, y:ys})
        writer.add_summary(summary_str, epoch)
        loss_sum += loss

    # 打乱数据顺序
    xvalues, yvalues = shuffle(x_data, y_data)

    b0temp = b.eval(session=sess)
    w0temp = w.eval(session=sess)
    loss_average = loss_sum/len(y_data)

    loss_list.append(loss_average)
    print("epoch=", epoch+1, "loss=", loss_average, "b=", b0temp, "w=", w0temp)

plt.plot(loss_list)
plt.show()