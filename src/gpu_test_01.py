#!/usr/bin/env python3
# coding: UTF-8
#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :
# sample: python3 gpu_test_01.py --checktype 1
#---------------------------------------------------------------

import tensorflow as tf
import argparse#引数拡張mジュール

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checktype", type=int,required=True, choices=[1, 2, 3])
    a = parser.parse_args()
    if 1==a.checktype:
        # デフォルトGPU (device:0)を指名で行列計算を行う
        with tf.compat.v1.device('/gpu:0'):
            a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(a, b)
        with tf.Session() as sess:
            print (sess.run(c))
    elif 2==a.checktype:
        from tensorflow.python.client import device_lib
        device_lib.list_local_devices()
    elif 3==a.checktype:
        mnist = tf.keras.datasets.mnist

        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test)

if __name__ == '__main__':
    main()