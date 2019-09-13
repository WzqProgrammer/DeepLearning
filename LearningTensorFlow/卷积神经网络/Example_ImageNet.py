from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tf1.compat.v1

# 创建图和会话
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# 导入Inception网络
model_fn = 'F:\GithubProjects/DeepLearning/data/tensorflow_inception_graph.pb'

with tf.gfile.GFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# 定义输入图像的占位符
t_input = tf.placeholder(np.float, name='input')


# 图像预处理————减均值（在训练Inception模型时做了减均值处理，此处也需减同样的均值以保持一致）
imagenet_mean = 117.0

# 图像预处理————增加维度
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
print(t_preprocessed)

# 导入模型并将经预处理的图像送入网络中
tf.import_graph_def(graph_def, {'input':t_preprocessed})

# 找到卷积层
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]

print('Number of layers', len(layers))
print(layers)