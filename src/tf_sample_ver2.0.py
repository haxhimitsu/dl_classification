#!/usr/bin/env python3
# coding: UTF-8
#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :
# sample:python3 tf_sample_ver2.0.py  --train_path  ~/Desktop/dataset_smple/train/ --val_path ~/Desktop/dataset_smple/val/  --log_dir  ../test/ --test_data_path ~/Desktop/dataset_smple/test/

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
#import pandas as pd
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
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
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9, # 最大値の80%まで
        allow_growth=True # True->必要になったら確保, False->全部
      ))
sess = sess = tf.Session(config=config)

#####################################################


parser = argparse.ArgumentParser()
parser.add_argument("--train_path", help="path to folder containing images")
parser.add_argument("--val_path",  help="set image size e.g.'0.5,0.8...'")
parser.add_argument("--max_epochs", type =int ,default=100,help="set trim width")
parser.add_argument("--save_weight_name", type=str,default="test",help="set trim height")
parser.add_argument("--test_data_path",  help="output path")
parser.add_argument("--log_dir",  help="log_path")
a = parser.parse_args()

log_dir=a.log_dir
myutil.create_directory(log_dir)
weight_filename=a.save_weight_name+".hdf5"
max_epochs=a.max_epochs
test_data_path=a.test_data_path


print(len(os.listdir(a.train_path)))

model=myutil.create_network(category_num=len(os.listdir(a.train_path)))
try:
    model.load_weights(os.path.join(log_dir,weight_filename))#学習結果がある場合，weightを読み込み
except OSError:
    print(".h5 file not found")
    print("start loading the data set")

    train_img,train_label,val_img,val_label=myutil.create_dataset(a.train_path,a.val_path)

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
                        patience=5,
                        verbose=1)

    # コンパイル
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    #####################################################


    history = model.fit(train_img, train_label, batch_size=20, epochs=max_epochs,
                    validation_data = (val_img, val_label), verbose = 1,callbacks=[es])#学習開始　パラメータは名前から察して
    
    model.save_weights(os.path.join(log_dir,weight_filename))#このコードがあるフォルダに重みを保存する
    
    score = model.evaluate(val_img, val_label, verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])

myutil.check_acc(model,test_data_path,log_dir)

