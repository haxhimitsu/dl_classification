import tensorflow as tf


from tensorflow.python.client import device_lib
device_lib.list_local_devices()
"""
# デフォルトGPU (device:0)を指名で行列計算を行う
with tf.compat.v1.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
with tf.Session() as sess:
    print (sess.run(c))
"""
