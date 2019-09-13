from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image
import scipy.misc
import tensorflow as tf1
import matplotlib.pyplot as plt
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
    sess.graph.as_default()

# 定义输入图像的占位符
t_input = tf.placeholder(np.float, name='input')


# 把一个numpy.ndarray保存成图像文件
def save_array(img_array, img_name):
    PIL.Image.fromarray(img_array, mode='RGB').save(img_name)
    print('img saved:%s'%img_name)


def render_naive(t_obj, img0, iter_n=20, step=1.0):
    """
    渲染函数
    :param t_obj:layer_output[:,:,channel]，即卷积某个通道的值
    :param img0: 初始图像（噪声图像）
    :param iter_n: 迭代次数
    :param step: 用于控制每次迭代的步长，可以看作学习率
    """

    # 计算平均值，由于我们的目标是调整输入图像使卷积层激活值尽可能大，
    t_score = tf.reduce_mean(t_obj)
    # 为达到上述目的，可以使用梯度下降，计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]

    img = img0.copy()   # 复制新图像避免影响原图像的值

    for i in range(iter_n):
        # 再sess中计算梯度， 以及当前的t_score，对img应用梯度
        g, score = sess.run([t_grad, t_score], {t_input:img})

        # 首先对梯度进行归一化处理
        g /= g.std() + 1e-8

        # 将正规化处理后的梯度应用再图像上，step用于控制每次迭代的步长
        img += g*step

        print('iter:%d'%(i+1), 'score(mean)=%f'%score)

    # 保存图片
    save_array(img, 'naive_deepdream.jpg')


# 将图像放大ratio倍
def resize_ratio(img, ratio):
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min) * 255
    img = np.float(PIL.Image.fromarray(img).resize(ratio))
    img = img/255 * (max - min) + min
    return img


# 调整图像尺寸
def resize(img, hw):
    min = img.min()
    max = img.max()
    img = (img - min)/(max - min) * 255
    img = np.float(PIL.Image.fromarray(img).resize(hw))
    img = img/255 * (max - min) + min
    return img


# 原始图像尺寸可能更大，从而导致内存耗尽问题
# 每次支队tile_size*tile_size大小的图像计算梯度，避免内存问题
def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    # 先在行上做整体运动，再在列上做整体运动
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)

    for y in range(0, max(h - sz//2, sz), sz):
        for x in range(0, max(w - sz//2, sz), sz):
            sub = img_shift[y:y + sz, x:x + sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y + sz, x:x + sz] = g

    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_deepdream(t_obj, img0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]
    img = img0.copy()

    # 将图像进行金字塔分解，从而分为高频、低频部分
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # 首先生成低频的图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step/(np.abs(g).mean() + 1e-7))

    img = img.clip(0, 255)
    save_array(img, 'mountain_deepdream.jpg')
    im = PIL.Image.open('mountain_deepdream.jpg').show()


# 定义卷积层、通道数，并取出对应的tensor
# name = 'mixed4d_3x3_bottleneck_pre_relu'
# name = 'mixed3a_3x3_bottleneck_pre_relu'
name = 'mixed4c'

# 图像预处理————减均值（在训练Inception模型时做了减均值处理，此处也需减同样的均值以保持一致）
imagenet_mean = 117.0

# mixed4d_3x3_bottlenect_pre_relu共有144个通道
# 此处可选任意通道（0~143之间任意整数）进行最大化
#channel = 118

# 图像预处理————增加维度
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)

# 导入模型并将经预处理的图像送入网络中
tf.import_graph_def(graph_def, {'input':t_preprocessed})

layer_output = graph.get_tensor_by_name("import/%s:0" % name)

# 定义噪声图像
#img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
# 用一张背景图像作为起点对图像进行优化
img_test = PIL.Image.open('mountain.jpg')

# 调用render_naive函数渲染
# 不指定特点通道，即表示利用所有通道特征
#render_naive(layer_output, img_test, iter_n=100)
render_deepdream(tf.square(layer_output), img_test)

# im = PIL.Image.open('mountain.jpg')
# im.show()
# im.save('mountain_native.jpg')
# im2 = PIL.Image.open('mountain_native.jpg')
# im2.show()