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
    # function      : careate  directory
    # input arg     : directory path
    # output        : none
    # func detail   : if not already exist argument directory path,this function create directory.
    """
    def create_directory(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        else:
            print("already exists"+directory_path)
    """
    # function      : create network
    # input arg     : category number
    # output        : network model
    # func detail   : please following comments
    """
    def create_network(self,category_num):
        model = Sequential()

        #Conv2D 
        #input shape(width,height,channel)
        #filter (3,3) 16channel
        model.add(Conv2D(32, (3, 3), padding='same',input_shape=(64,64,3)))
        #Downsamples the input representation by taking the maximum by pool_size
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        #randomly sets input units to 0, which helps prevent overfitting.
        model.add(Dropout(0.5))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

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
    """
    # function      : create dataset
    # input arg     : train data path,validation data path
    # output        : train img,train label,validation img,validation label
    # func detail   : please following comments
    """
    def create_dataset(self,train_path,val_path):

        TrainIMG = []
        TrainLABEL = []
        ValIMG = []
        ValLABEL = []
        TestIMG = []
        TestLABEL = []
        label =0
        """
        #trainディレクトリ内のディレクトリをリストで取得
        train--class1
             --class2
             --class3
        ->[class1,class2,class3]
        """
        img_dirs=os.listdir(train_path)

        for i, d in enumerate(img_dirs):#classごとに画像を読み込んでいく
            files0 = os.listdir(train_path+ d)
            files1 = os.listdir(val_path+ d)
            print(train_path+d)#print->train/class1
            print(val_path+d)#print->val/class1
            for f0 in files0:#class内の画像を順番に読み込む
                #load_img->train/class1/picturename.png
                img1 = img_to_array(load_img(train_path + d + '/' + f0,target_size=(64,64,3)))
                #train imgのlistに追加
                TrainIMG.append(img1)
                #train labelのlistに追加
                TrainLABEL.append(label)

            for f1 in files1:
                img2 = img_to_array(load_img(val_path + d + '/' + f1,target_size=(64,64,3)))
                ValIMG.append(img2)
                ValLABEL.append(label)
            #1class読み込みが終わったらlabelをインクリメント
            #次は，train/class2 を読み込んでいく
            label = label + 1
            print("now:" + img_dirs[i])
        
        #tensor flowの入力に合わせるためにデータを整える
        TrainIMG = np.asarray(TrainIMG)#np.asarrayに変換
        TrainLABEL = np.asarray(TrainLABEL)
        TrainIMG = TrainIMG.astype('float32')#float32に変換
        TrainIMG = TrainIMG / 255.0#0~255に正規化
        #ラベルを数値ではなく,0or1を要素に持つベクトルで表現
        #ラベルが 1の場合[0, 1, 0]
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


    def check_acc(self,model,test_img_path,log_dir):
        result_count = []
        all_count = 0

        img_dirs=os.listdir(test_img_path)
        for i,name in enumerate(img_dirs):
            result_count.append(0)
        print("\t",end =" ")

        for j, name in enumerate(img_dirs):
            print(img_dirs[j] + "\t",end = "")
        print("")

        for i, d in enumerate(img_dirs):
            files2 = os.listdir(test_img_path + d)
            for f2 in files2:
                test_img = np.array(load_img(test_img_path + d + '/' + f2).resize((64, 64)))
                result = model.predict_classes(np.array([test_img / 255.]))
                
                all_count = all_count + 1
                for j, name in enumerate(img_dirs):
                    if result == j:
                        result_count[j] = result_count[j] + 1
            
                #print(result)
            
            print(d + "\t",end="")
            for j, name in enumerate(img_dirs):
                print("{0:.2f}\t".format(result_count[j]/all_count*100),end="")
            print("")

            all_count = 0
            for j, name in enumerate(img_dirs):
                result_count[j] = 0
        f = open(log_dir+'test_img_result.txt', 'w')
        for i, d in enumerate(img_dirs):
            files2 = os.listdir(test_img_path + d)
            for f2 in files2:
                test_img = np.array(load_img(test_img_path + d + '/' + f2).resize((64, 64)))
                
                y1 = model.predict(np.array([test_img / 255.])) #判別精度(確率)の表示
                y2 = model.predict_classes(np.array([test_img / 255.])) #判別精度(ラベル)の表示
                f.write(f2 + "\t" + str(y1) + "\t" + str(y2) +"\n")
                #print(y)
        f.close()
        print("check_accc")

    """
    # function      : acc2
    # input arg     : network model, test img data path, log save directory
    # output        : none
    # func detail   : テスト画像を読み込み，学習したネットワークで分類をする．
    #                 各々の正解ラベルに対する正答率を算出する
    # testdata_architecture :test--class1
    #                            --class2
    #                            --class3
    """
    def acc2(self,model,test_img_path,log_dir):
        score=[]
        test_img_path=test_img_path
        print(test_img_path)
        img_dirs=os.listdir(test_img_path)#->
        print(img_dirs)
        
        # for i, d in enumerate(img_dirs):
        #     files2 = os.listdir(test_img_path + d)
        # #test_img_path="~/Desktop/nagase_1200_20201021_trim/dataset_06_tmp/tests/0100_0201"
        #     print(test_img_path)
        for f2 in range(len(img_dirs)):
            #print(test_img_path+f2)
            test_img = np.array(load_img(test_img_path + img_dirs[f2]).resize((64, 64)))
            result = model.predict_classes(np.array([test_img / 255.]))
            score.append(result)
        label0 = [i for i in score if i == 0]
        label1 = [i for i in score if i == 1]
        label2 = [i for i in score if i == 2]
        label3 = [i for i in score if i == 3]
        print("label0",(len(label0)/len(score))*100)
        print("label1",(len(label1)/len(score))*100)
        print("label2",(len(label2)/len(score))*100)
        print("label3",(len(label3)/len(score))*100)
        #print(score)
        return




    def sayStr(self, str):
        print (str)
 
if __name__ == '__main__':
    test = myutil()
    test.sayStr("Hello")   # Hello