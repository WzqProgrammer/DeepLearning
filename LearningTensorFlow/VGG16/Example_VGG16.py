import os
import tensorflow as tfa
import numpy as np
from time import time
import LearningTensorFlow.VGG16.VGG16_model as model
from LearningTensorFlow.VGG16.vgg_preprocess import preprocess_for_train

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf = tfa.compat.v1


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        print(root, "  ", sub_folders, "  ", files)
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))


    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('\\')[-1]
        print(letter, ":", str(n_img))
        if letter == 'cat':
            labels = np.append(labels, n_img*[0])
        else:
            labels = np.append(labels, n_img*[1])
    print(len(labels))
    print(len(images))
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    # 将顺序打乱
    np.random.shuffle(temp)
    # 将图片与其对应标签分别放入不同列表中
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list


# 通过读取列表来载入批量图片及标签
def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):

    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = preprocess_for_train(image, img_width, img_height)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

# 标签处理
def onehot(labels):
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels


startTime = time()
batch_size = 32
capacity = 256  # 内存中存储的最大数据容量
means = [123.68, 116.779, 103.939]  # VGG训练时图像预处理所减均值（RGB三通道）


data_dir = 'F:\下载\学习课件\kaggle\\train'
# 模型重新训练与保存
xs, ys = get_file(data_dir) # 获取图像列表和标签列表
img_width = 224
img_height = 224
image_batch, label_batch = get_batch(xs, ys, img_width, img_height, batch_size, capacity)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.int32, [None, 2])  # 对“猫”和“狗”两个类别进行判定

vgg = model.VGG16(x)
fc8_finetuining = vgg.probs   # 即softmax(fc8)
# 损失函数
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc8_finetuining, labels=y))
# 优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 通过npz格式的文件获取VGG的相应权重参数，从而将权重注入即可实现复用
vgg.load_weights('F:\下载/学习课件/vgg16_weights.npz', sess)
saver = tf.train.Saver()

# 使用协调器Coordinator来管理线程
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

epoch_start_time = time()

# 迭代训练
for i in range(1000):

    images, labels = sess.run([image_batch, label_batch])
    # 用one-hot形式对标签进行编码
    labels = onehot(labels)

    sess.run(optimizer, feed_dict={x:images, y:labels})
    loss = sess.run(loss_function, feed_dict={x:images, y:labels})
    print("Now the loss is %f" % loss)

    epoch_end_time = time()
    print("Current epoch takes: ", (epoch_end_time - epoch_start_time))
    epoch_start_time = epoch_end_time

    if (i+1) % 500 == 0:
        saver.save(sess, os.path.join('models/', 'epoch_{:06d}.ckpt'.format(i)))

    print("-----------------Epoch %d is finished--------------------"%i)


# 模型保存
saver.save(sess, 'models/')
print("Optimization Finished!")

duration = time() - startTime
print("Train Finished takes : ", "{:.2f}".format(duration))

coord.request_stop()  # 通知其他线程关闭
coord.join(threads)  # join操作等待其他线程结束，其他所有线程关闭之后，这一函数才能返回


