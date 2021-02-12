#!/usr/bin/env python3
# coding: UTF-8
#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :
#Usage
# python3 src/tf_sample_ver2.0.py  --dataset_path "{your input directory}" --log_dir "{your output directry}
#---------------------------------------------------------------

import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras.callbacks
from keras.models import Sequential, model_from_json
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import cv2
import os
import csv
import copy
import random
import argparse

#my module
import sys
from utils.myutils import myutil

myutil=myutil()
myutil.sayStr("Hello")


#################setting GPU useage#####################
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8, # 最大値の80%まで
        allow_growth=True # True->必要になったら確保, False->全部
      ))
sess = sess = tf.Session(config=config)

#####################################################


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path",required=True,help="path to root dataset directory")
parser.add_argument("--train_path",help="path to train_data")
parser.add_argument("--val_path",  help="path to val_data")
parser.add_argument("--max_epochs", type =int ,default=100,help="set max epoch(int)")
parser.add_argument("--batch_size", type =int ,default=32,help="set batch size 2,4,6,8,..")
parser.add_argument("--save_weight_name", type=str,default="test",help="set_network_weight_name")
parser.add_argument("--test_path",  help="pat to test_path")
parser.add_argument("--log_dir", required=True, help="set_to_log_directory")
a = parser.parse_args()

log_dir=a.log_dir
myutil.create_directory(log_dir)
weight_filename=a.save_weight_name+".hdf5"
max_epochs=a.max_epochs

if a.train_path is None:
    train_path=a.dataset_path+"trains/"
    #print("train_path",train_path)
else:
    train_path=a.train_path
    #print("train_path",train_path)
if a.val_path is None:
    val_path=a.dataset_path+"valids/"
else:
    val_path=a.val_path
if a.test_path is None:
    test_path=a.dataset_path+"tests/"
else:
    test_path=a.test_path


print(len(os.listdir(train_path)))

model=myutil.create_network(category_num=len(os.listdir(train_path)))
try:
    model.load_weights(os.path.join(log_dir,weight_filename))#学習結果がある場合，weightを読み込み
except OSError:
    print(".h5 file not found")
    print("start loading the data set")

    train_img,train_label,val_img,val_label=myutil.create_dataset(train_path,val_path)

    ###################EalyStopping#######################
    """
    検証データｎ対する誤差が増加してくるタイミングが訓練データにオーバーフィッティング
    し始めているタイミングと考えることができるので，エポックごとの検証データに対する誤差の値を監視し，
    一定のエポック数連続して誤差がそれまでの最小値をしたまわることがなければ打ち切る．
    monitor='監視する値の指定'
    patience='監視している値が何エポック連続で上回ったら早期終了するか'
    verbose='早期終了したかどうかをログで出力するか'
    """
    es = EarlyStopping(monitor='val_loss',
                        patience=20,
                        verbose=1)

    # コンパイル
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #####################################################


    history = model.fit(train_img, train_label, batch_size=a.batch_size, epochs=max_epochs,validation_data = (val_img, val_label), verbose = 1,callbacks=[es])#学習開始　パラメータは名前から察して

    model.save_weights(os.path.join(log_dir,weight_filename))#このコードがあるフォルダに重みを保存する
    
    score = model.evaluate(val_img, val_label, verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])

    myutil.check_acc(model,test_path,log_dir)

    del train_img,train_label,val_img,val_label

