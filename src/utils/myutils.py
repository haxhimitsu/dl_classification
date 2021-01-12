#!/usr/bin/env python3
# coding: UTF-8

#---------------------------------------------------------------
# author:"Haxhimitsu"
# date  :"2021/01/06"
# cite  :
# sample:python3 imgtrim_gui_ver.2.0.py  --input_dir ../assets/original_img/cbn_test_01/ --output_dir ../assets/sample_output/  --trim_width 32 --trim_height 64
#---------------------------------------------------------------
import keras
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import keras.callbacks
from keras.models import Sequential, model_from_json
import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt


import os

class myutil:

    """
    # function      : careate  directry
    # input arg     : directry path
    # output        : none
    # func detail   : if not already exsits arg directry path,this function create directry.
    """
    def create_directory(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        else:
            print("already exsists"+directory_path)

    def create_network(self,category_num):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same',input_shape=(32,32,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        #CNN_weight_2_1の時はこの部分を使わない
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(400))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(category_num))  #分類数を決めている
        model.add(Activation('softmax'))

        #model.summary()

        return model

    def create_dataset(self,train_path,val_path):

        TrainIMG = []
        TrainLABEL = []
        ValIMG = []
        ValLABEL = []
        TestIMG = []
        TestLABEL = []
        label =0

        img_dirs=os.listdir(train_path)
        for i, d in enumerate(img_dirs):
            files0 = os.listdir(train_path+ d)
            files1 = os.listdir(val_path+ d)
            print(train_path+d)
            print(val_path+d)
            for f0 in files0:
                img1 = img_to_array(load_img(train_path + d + '/' + f0,target_size=(32,32,3)))
                TrainIMG.append(img1)
                TrainLABEL.append(label)
    
            for f1 in files1:
                img2 = img_to_array(load_img(val_path + d + '/' + f1,target_size=(32,32,3)))
                ValIMG.append(img2)
                ValLABEL.append(label)

            label = label + 1
            print("now:" + img_dirs[i])

        TrainIMG = np.asarray(TrainIMG)
        TrainLABEL = np.asarray(TrainLABEL)
        TrainIMG = TrainIMG.astype('float32')
        TrainIMG = TrainIMG / 255.0
        TrainLABEL = np_utils.to_categorical(TrainLABEL, label)

        ValIMG = np.asarray(ValIMG)
        ValLABEL = np.asarray(ValLABEL)
        ValIMG = ValIMG.astype('float32')
        ValIMG = ValIMG / 255.0
        ValLABEL = np_utils.to_categorical(ValLABEL, label)
        
        TestIMG = np.asarray(TestIMG)
        TestLABEL = np.asarray(TestLABEL)
        TestIMG = TestIMG.astype('float32')
        TestIMG = TestIMG / 255.0
        TestLABEL = np_utils.to_categorical(TestLABEL, label)

        label=0
        print("completed loading the data set")

        return TrainIMG,TrainLABEL,ValIMG,ValLABEL


    def sayStr(self, str):
        print (str)
 
if __name__ == '__main__':
    test = myutil()
    test.sayStr("Hello")   # Hello