import tensorflow as tfa
import matplotlib.pyplot as plt
import numpy as np

tf = tfa.compat.v1

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile('dog.jpg', 'rb').read()

with tf.Session() as sess:
    # 对图像进行jpeg的格式解码从而得到图像对应的三维矩阵
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 解码之后的结果是一个张量
    print(img_data.eval())

    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    # 可视化
    plt.imshow(img_data.eval())
    plt.show()

    # 用双线性插值法将图像缩放为指定尺寸
    resized1 = tf.image.resize_images(img_data, [256, 256], method=0)
    # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片
    resized1 = np.array(resized1.eval(), dtype='uint8')
    ax = plt.subplot(2, 5, 1)
    ax.imshow(resized1)
    ax.set_title("method=0", fontsize=10)

    # 用最邻近插值法将图像缩放到指定尺寸
    resized2 = tf.image.resize_images(img_data, [256, 256], method=1)
    resized2 = np.array(resized2.eval(), dtype='uint8')
    ax = plt.subplot(2, 5, 2)
    ax.imshow(resized2)
    ax.set_title("method=1", fontsize=10)

    # 用双立方插值法将图像缩放到指定尺寸 -
    resized3 = tf.image.resize_images(img_data, [256, 256], method=2)
    resized3 = np.array(resized3.eval(), dtype='uint8')
    ax = plt.subplot(2, 5, 3)
    ax.imshow(resized3)
    ax.set_title("method=2", fontsize=10)

    # 用像素区域插值法将图像缩放到指定尺寸
    resized4 = tf.image.resize_images(img_data, [256, 256], method=3)
    resized4 = np.array(resized4.eval(), dtype='uint8')
    ax = plt.subplot(2, 5, 4)
    ax.imshow(resized4)
    ax.set_title("method=3", fontsize=10)

    # 如果目标图像小于原始图像尺寸，则在中心位置减裁，反之则用黑色像素进行填充
    croped_1 = tf.image.resize_image_with_crop_or_pad(img_data, 400, 400)
    ax = plt.subplot(2, 5, 5)
    ax.imshow(croped_1.eval())
    ax.set_title("croped_1", fontsize=10)

    # 随机裁剪
    random_crop_1 = tf.random_crop(img_data, [60, 60, 3])
    ax = plt.subplot(2, 5, 6)
    ax.imshow(random_crop_1.eval())
    ax.set_title("random_crop_1", fontsize=10)

    # 水平翻转
    flip_left_right = tf.image.flip_left_right(img_data)
    ax = plt.subplot(2, 5, 7)
    ax.imshow(flip_left_right.eval())

    # 上下翻转
    flip_up_down = tf.image.flip_up_down(img_data)
    ax = plt.subplot(2, 5, 8)
    ax.imshow(flip_up_down.eval())

    # 改变对比度
    # 将图像的对比度降低至原来的二分之一
    contrast = tf.image.adjust_contrast(img_data, 0.5)
    # 将图像的对比度调高至原来的5倍
    #contrast = tf.image.adjust_contrast(img_data, 5)
    # 在一定范围内随机调整图像对比度
    #contrast = tf.image.random_contrast(img_data, lower=0.1, upper=0.7)
    ax = plt.subplot(2, 5, 9)
    ax.imshow(contrast.eval())

    # 白化处理 将图像的像素值转化成零均值和单位方差
    standardization = tf.image.per_image_standardization(img_data)
    ax = plt.subplot(2, 5, 10)
    ax.imshow(np.array(standardization.eval(), dtype='uint8'))

    plt.show()

