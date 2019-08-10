import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tfa
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf = tfa.compat.v1

input_data = tf.Variable(np.random.rand(10, 9, 9, 4), dtype=np.float)
# 卷积核矩阵
filter_data = tf.Variable(np.random.rand(2, 2, 4, 2), dtype=np.float)

# 卷积操作
y = tf.nn.conv2d(input_data, filter_data, strides=[1,1,1,1], padding='SAME')

# 最大池化
output = tf.nn.max_pool(value=y, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# 平均池化
output = tf.nn.avg_pool(value=y, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(input_data)
print(y)

# def ImgConvolve(image_array, kernel):
#     """
#     对图像进行卷积处理
#     :param image_array:原灰度图像矩阵
#     :param kernel: 卷积核
#     :return: 原图像与算子进行卷积后的结果
#     """
#     image_arr = image_array.copy()
#     img_dim1, img_dim2 = image_arr.shape
#     k_dim1, k_dim2 = kernel.shape
#     # 获取padding填充边缘范围
#     AddW = int((k_dim1-1) / 2)
#     AddH = int((k_dim2-1) / 2)
#
#     # padding填充
#     temp = np.zeros([img_dim1 + AddW*2, img_dim2 + AddH*2])
#     # 将原图拷贝到临时图片的中央
#     temp[AddW:AddW+img_dim1, AddH:AddH+img_dim2] = image_arr[:,:]
#     # 初始化一张同样大小的图片作为输出图片
#     output = np.zeros_like(a=temp)
#     # 将扩充后的图和卷积核进行卷积
#     for i in range(AddW, AddW+img_dim1):
#         for j in range(AddH, AddH+img_dim2):
#             output[i][j] = int(np.sum(temp[i-AddW:i+AddW+1, j-AddW:j+AddW+1] * kernel))
#
#     return output[AddW:AddW+img_dim1, AddH:AddH+img_dim2]
#
#
# # 提取竖直方向特征 sobel_x
# kernel_1 = np.array(
#     [[-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]]
# )
#
# # 提取水平方向特征 sobel_y
# kernel_2 = np.array(
#     [[-1, -2, -1],
#      [0, 0, 0],
#      [1, 2, 1]]
# )
#
# # Laplace扩展算子 二阶微分算子
# kernel_3 = np.array(
#     [[1, 1, 1],
#      [1, -8, 1],
#      [1, 1, 1]]
# )
#
# # 打开图像并转换成灰度图像
# image = Image.open("F:\图片/image1.png").convert("L")
#
# # 将图像转化成数组
# image_array = np.array(image)
#
# # 卷积操作
# sobel_x = ImgConvolve(image_array, kernel_1)
# sobel_y = ImgConvolve(image_array, kernel_2)
# laplace = ImgConvolve(image_array, kernel_3)
#
# # 显示图像
# plt.imshow(image_array, cmap='gray')
# plt.axis("off")
# plt.show()
#
# plt.imshow(sobel_x, cmap='gray')
# plt.axis("off")
# plt.show()
#
# plt.imshow(sobel_y, cmap='gray')
# plt.axis("off")
# plt.show()
#
# plt.imshow(laplace, cmap='gray')
# plt.axis("off")
# plt.show()
