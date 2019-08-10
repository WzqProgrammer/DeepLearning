import tensorflow as tfa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tfa.compat.v1  #TensorFlow1.x的版本

ckpt_dir = './ckpt_dir/'
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

# # 创建一个运行常量，将作为一个节点加入到默认计算图中
# hello = tf.constant("Hello World!")

# # 创建一个TF对话
# sess = tf.Session()
#
# # 运行并获取结果
# print(sess.run(hello))

# # 一个简单的计算图
# node1 = tf.constant(3.0, tf.float32, name='node1')
# node2 = tf.constant(4.0, tf.float32, name='node2')
# node3 = tf.add(node1, node2)
#
# print(node1)  # 输出的是一个张量结构
# print(node3)
#
# sess = tf.Session()
#
# print("运行sess.run(node)的结果：", sess.run(node3))
#
# sess.close()

# tf.reset_default_graph()   # 清除default graph和不断增加的节点
#
# a = tf.Variable(1, name='a')  # 定义变量 a
# b = tf.add(a, 1, name='b')    # 定义操作 b 为 a+1
# c = tf.multiply(b, 4, name='c')  # 定义操作 c 为 b*4
# d = tf.subtract(c, b, name='d')  # 定义操作 d 为 c-b
#
# logdir = 'F:\PythonProjects\DeepLearning\LearningTensorFlow'
#
# # 生成一个写日志的writer, 并将当前的TensorFlow计算图写入日志
# writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# writer.close()


# node1 = tf.constant(3.0, tf.float32, name='node1')
# node2 = tf.constant(4.0, tf.float32, name='node2')
# result = tf.add(node1, node2)
#
# sess = tf.Session()
# print(sess.run(result))
#
# sess.close()
#
# with sess.as_default():
#     print(result.eval())
#
# # 变量初始化
# init = tf.global_variables_initializer()
# sess.run(init)
#
# print(sess.run(result))



# value = tf.Variable(0, name='value')
# sum_value = tf.Variable(0, 'sum')
#
# one = tf.constant(1)
#
# new_value = tf.add(value, one)
# update_value = tf.assign(value, new_value)  # 赋值操作，将 new_value 的值赋给 value
# total_value = tf.add(sum_value, value)     # 新值加上
# result = tf.assign(sum_value, total_value)  # 将加好的值赋值
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for _ in range(10):
#         sess.run(update_value)
#         sess.run(result)
#         print(sess.run(value))
#
#     print(sess.run(sum_value))   # 输出：55

# 此代码生成一个2×3的二维数组， 矩阵中每个元素的类型都是tf.float32, 内部对应的符合名称为tx
# x = tf.placeholder(tf.float32, [2, 3], name='tx')
#
# a = tf.placeholder(tf.float32, name='a')
# b = tf.placeholder(tf.float32, name='b')
# c = tf.multiply(a, b, name='c')
# d = tf.subtract(a, b, name='d')
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     # 通过feed_dict的参数传值， 按字典格式
#     #result = sess.run(c, feed_dict={a:8.0, b:3.5})
#
#     # 多个feed可以在一个语句中运行
#     result = sess.run([c, d], feed_dict={a:[8.0, 2.0, 3.5], b:[1.5, 2.0, 4.0]})
#
#     print(result)
#     # [12., 4., 14.]
#     # [6.5, 0., -0.5]
#
#     print(result[0])
#     # [12., 4., 14.]

# tf.reset_default_graph()
#
# # logdir改为自己机器上合适的路径
# logdir = 'F:\PythonProjects\DeepLearning\LearningTensorFlow'
#
# input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
# input2 = tf.Variable(tf.random_uniform([3]), name='input2')
# output = tf.add_n([input1, input2], name='add')
#
# # 生成一个写日志的writer， 并将当前的TensorFlow计算图写入日志
# writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
# writer.close()
