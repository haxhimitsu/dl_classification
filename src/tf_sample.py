# coding: UTF-8
##################################################################
# memo
# it1701 Kazushi Uchida
#ディレクトリ構成  
#
##################################################################

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

import cv2
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# set_session(tf.Session(config=config))

from tensorflow.keras.callbacks import EarlyStopping

import csv
import copy
import random

#################setting GPU useage#####################

config = tf.ConfigProto(
      gpu_options=tf.GPUOptions(
          per_process_gpu_memory_fraction=0.9, # 最大値の80%まで
          allow_growth=True # True->必要になったら確保, False->全部
      )
    )
sess = sess = tf.Session(config=config)

#####################################################




EPOCHS = 100
h5 = 'CNN_weight_goki_original_01.h5' #使用する重みファイル名(存在しないファイル名を書くと学習を開始し，その名前で重みファイルを保存)
#h5 = 'weight\cnn_model05-loss0.35-acc0.87-vloss0.31-vacc0.89.hdf5'
#epoch_count = 1

#####################################################
TrainIMG = []
TrainLABEL = []
ValIMG = []
ValLABEL = []
TestIMG = []
TestLABEL = []

img_dirs = ['edge','fuchaku','haikei','kuro']#データセットを保存しているフォルダ名 可変 
img_dirs2=['test2']
img_dirs3=['testy']
filename='result_grindstone2-test.csv'
label = 0

result_count = []
all_count = 0
#####################################################

################モデル作成############################
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
model.add(Dense(4))  #分類数を決めている
model.add(Activation('softmax'))

model.summary()


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


####################学習関連#########################
train_path='dataset2/train/'
val_path='dataset2/val/'
try:#学習結果がある場合読み込み
    model.load_weights(h5)
except OSError:#学習結果がなかった場合学習開始
    print(".h5 file not found")
    print("start loading the data set")
    ################train and val用データセット作成###################
    for i, d in enumerate(img_dirs):
        files0 = os.listdir(train_path+ d)
        files1 = os.listdir( val_path+ d)
#        files2 = os.listdir('dataset\\test\\' + d)
        for f0 in files0:
            img1 = img_to_array(load_img(train_path + d + '/' + f0,target_size=(32,32,3)))
            TrainIMG.append(img1)
            TrainLABEL.append(label)
    
        for f1 in files1:
            img2 = img_to_array(load_img(val_path + d + '/' + f1,target_size=(32,32,3)))
            ValIMG.append(img2)
            ValLABEL.append(label)
            
#        for f2 in files2:
#            img3 = img_to_array(load_img('dataset\\test\\' + d + '/' + f2,target_size=(64,64,3)))
#            ValIMG.append(img3)
#            ValLABEL.append(label)
        
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

    label = 0
    print("completed loading the data set")
    ############################################################
    
    print("training start")
       
    history = model.fit(TrainIMG, TrainLABEL, batch_size=20, epochs=EPOCHS,
                    validation_data = (ValIMG, ValLABEL), verbose = 1,callbacks=[es])#学習開始　パラメータは名前から察して
    
    model.save_weights(h5)#このコードがあるフォルダに重みを保存する
    
    score = model.evaluate(ValIMG, ValLABEL, verbose=0)
    print('Test loss :', score[0])
    print('Test accuracy :', score[1])
#####################################################



####################判別精度確認用####################
test_img_path='dataset2/test/'
for i,name in enumerate(img_dirs):
    result_count.append(0)
print("\t",end =" ")
for j, name in enumerate(img_dirs):
    print(img_dirs[j] + "\t",end = "")
print("")

for i, d in enumerate(img_dirs):
    files2 = os.listdir(test_img_path + d)
    for f2 in files2:
        test_img = np.array(load_img(test_img_path + d + '/' + f2).resize((32, 32)))
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
#####################################################

#####################################################
f = open('result.txt', 'w')
for i, d in enumerate(img_dirs):
    files2 = os.listdir(test_img_path + d)
    for f2 in files2:
        test_img = np.array(load_img(test_img_path + d + '/' + f2).resize((32, 32)))
        
        y1 = model.predict(np.array([test_img / 255.])) #判別精度(確率)の表示
        y2 = model.predict_classes(np.array([test_img / 255.])) #判別精度(ラベル)の表示
        f.write(f2 + "\t" + str(y1) + "\t" + str(y2) +"\n")
        #print(y)
f.close()
#####################################################

#testフォルダ内の画像で間違った画像をresult_imgフォルダ内に保存する
#for i, d in enumerate(img_dirs):
#    files3 = os.listdir('dataset2\\test\\' + d)
#    for f3 in files3:
#        test_img = np.array(load_img('dataset2\\test\\' + d + '/' + f3).resize((32, 32)))
#        test_img_2 = cv2.imread('dataset2\\test\\' + d + '/' + f3,1)
#        
#        #y1 = model.predict(np.array([test_img / 255.]))
#        y = model.predict_classes(np.array([test_img / 255.]))
#        
#        if (y == 0) and (y != i):
#            cv2.imwrite('result_img\\edge\\'+ d +'_'+ f3, test_img_2)
#        if (y == 1) and (y != i):
#            cv2.imwrite('result_img\\fuchaku\\'+ d +'_'+ f3, test_img_2)
#        if (y == 2) and (y != i):
#            cv2.imwrite('result_img\\haikei\\'+ d +'_'+ f3, test_img_2)
#        if (y == 3) and (y != i):
#            cv2.imwrite('result_img\\kuro\\'+ d +'_'+ f3, test_img_2)
            
##############実行用##################################
#c0=0
#c1=0
#c2=0
#c3=0
#for i, d in enumerate(img_dirs2):
#    files3 = os.listdir('dataset2\\' + d)
#    for f3 in files3:
#        test_img = np.array(load_img('dataset2\\' + d + '/' + f3).resize((32, 32)))
#        test_img_2 = cv2.imread('dataset2\\' + d + '/' + f3,1)
#        
#        #y1 = model.predict(np.array([test_img / 255.]))
#        y = model.predict_classes(np.array([test_img / 255.]))
#        
#        if (y == 0):
#            cv2.imwrite('dataset2\\trim\\edge\\'+ d +'_'+ f3, test_img_2)
#            c0=c0+1
#        if (y == 1):
#            cv2.imwrite('dataset2\\trim\\fuchaku\\'+ d +'_'+ f3, test_img_2)
#            c1=c1+1
#        if (y == 2):
#            cv2.imwrite('dataset2\\trim\\haikei\\'+ d +'_'+ f3, test_img_2)
#            c2=c2+1
#        if (y == 3):
#            cv2.imwrite('dataset2\\trim\\kuro\\'+ d +'_'+ f3, test_img_2)
#            c3=c3+1
#            
#    print('edge:'+str((c0*100)/20)+'\n')
#    print('fucyaku:'+str((c1*100)/24)+'\n')
#    print('haikei:'+str((c2*100)/20)+'\n')
#    print('kuro:'+str((c3*100)/20)+'\n')

###############実行用２####################################
#背景画像生成
height=960
width=600
flag=0

ORG_trim_X = 32#画像から切り抜く矩形のサイズを指定
ORG_trim_Y = 32
test_count = 0
for i, d in enumerate(img_dirs3):
    files4 = os.listdir('dataset2/' + d)
    with open(filename,mode='a') as f:
        f.write('ディレクトリ名,画像名,摩耗砥粒[%],付着物[%]\n')
        f.write(d)
    for f4 in files4:
       test_img = np.array(load_img('dataset2/' + d + '/' + f4).resize((960, 600))) #960,600
       test_img_2 = cv2.imread('dataset2/' + d + '/' + f4,1)
       print(f4)
       #HSV化
       test_1=copy.copy(test_img_2)
       test_org_h, test_org_w, test_org_ch = test_1.shape[:3]

       b, g, r = cv2.split(test_1)  # 画像をチャンネルごとに分離する。
       
       ret, res = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)#220
       ret, nichi2 = cv2.threshold(r, 50, 255, cv2.THRESH_BINARY)
       nichi2=cv2.bitwise_not(nichi2)#kuro
       nichi=cv2.bitwise_and(res,nichi2)
       #カーネルサイズの設定(8近傍)
       neiborhood8 = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]],np.uint8)

       #膨張処理
       #膨張処理
       nichi = cv2.dilate(nichi, neiborhood8, iterations=1)
       nichi = cv2.erode(nichi, neiborhood8, iterations=1)

       #cv2.imshow("BIN",nichi)
       label = cv2.connectedComponentsWithStats(nichi)#ラベリング
       ##label[0]=ラベルの個数，label[2]=data,label[3]=ラベル重心
       n = label[0] - 1#ラベルの個数取得とラベル重心cogを取得
       cog = np.delete(label[3], 0, 0)
                       
#       n=nLabels-1
#       cog=np.delete(data,0,0)
       
       font = cv2.FONT_HERSHEY_PLAIN
    
       count=0
       count2=0
       count3=0
       count4=0
       flag=0
       for i in range(n):
           ag_X_s = int(cog[i][0]) - ORG_trim_X / 2
           ag_Y_s = int(cog[i][1]) - ORG_trim_Y / 2
           ag_X_f = int(cog[i][0]) + ORG_trim_X / 2
           ag_Y_f = int(cog[i][1]) + ORG_trim_Y / 2
        
           if ag_X_s < 0:ag_X_s = 0
           if ag_Y_s < 0:ag_Y_s = 0
           if ag_X_f > test_org_w:ag_X_f = test_org_w
           if ag_Y_f > test_org_h:ag_Y_f = test_org_h
            
           trim = test_img_2[int(ag_Y_s):int(ag_Y_f),int(ag_X_s):int(ag_X_f)]#トリミング　im[y:y+h, x:x+w]
           trim2 =nichi[int(ag_Y_s):int(ag_Y_f),int(ag_X_s):int(ag_X_f)]#トリミング　im[y:y+h, x:x+w]
           cv2.imwrite('dataset2/trim1/trim_test.jpg', trim)
        
#           cv2.rectangle(test_img_2, (int(ag_X_s), int(ag_Y_s)), (int(ag_X_f), int(ag_Y_f)), (0, 255, 0), 2)
            
           trim_h, trim_w, trim_ch = trim.shape[:3]
        
           if (trim_w == ORG_trim_X) and (trim_h == ORG_trim_Y):
               test_img = np.array(trim)
               test_img = np.array(load_img('dataset2/trim1/trim_test.jpg').resize((32, 32)))
               result = model.predict_classes(np.array([test_img / 255.]))
           
           if flag==0:
               s=(height,width)
               blank=np.zeros_like((trim2))
               blank_1=cv2.resize(blank,s)
               blank2=np.zeros_like((trim2))
               blank2_1=cv2.resize(blank2,s)
               flag=1
            
           if result == 0:
               text = 'edge'
#               cv2.rectangle(test_1, (int(ag_X_s), int(ag_Y_s)), (int(ag_X_f), int(ag_Y_f)), (0, 0,255), 2) #red
#               cv2.circle(test_1, (int(cog[i][0]),int(cog[i][1])), 2,(0, 0,255), -1) #red
               count=count+1
               blank_1[int(ag_Y_s):int(ag_Y_f),int(ag_X_s):int(ag_X_f)]=trim2
#               print(text)
#               print(n)
               
           if result == 1:
              text = 'fuchaku'
              cv2.rectangle(test_1, (int(ag_X_s), int(ag_Y_s)), (int(ag_X_f), int(ag_Y_f)), (0, 0, 0), 2) #black
              blank2_1[int(ag_Y_s):int(ag_Y_f),int(ag_X_s):int(ag_X_f)]=trim2
              count2=count2+1
              
           if result == 2:
              text = 'haikei'
#              cv2.rectangle(test_1, (int(ag_X_s), int(ag_Y_s)), (int(ag_X_f), int(ag_Y_f)), (255, 255, 0), 2) #cyan
              count3=count3+1
              
           if result == 3:
              text = 'kuro'
#              cv2.rectangle(test_1, (int(ag_X_s), int(ag_Y_s)), (int(ag_X_f), int(ag_Y_f)), (255, 0,255), 2) #magenta
              count4=count4+1
                
#       cv2.putText(test_img_2,count,(900,500),font, 1,(0,0,255)) 800,550  1.2 2
#       cv2.putText(test_1, 'stone:'+str(count), (0,30), cv2.FONT_HERSHEY_COMPLEX_SMALL | cv2.FONT_ITALIC, 0.5, (0,255,255), 1, cv2.LINE_AA) #yellow
       print(f4+'--> Class1:'+str(count))
       print(f4+'--> Class2:'+str(count2))
       print(f4+'--> Class3:'+str(count3))
       print(f4+'--> Class4:'+str(count4))
       #cv2.imshow("desktop", test_1)
#       cv2.imwrite('dataset2\\trim1\\res3_'+f4, test_1)
#       cv2.imwrite('dataset2\\trim2\\'+f4, test_1)
       
       #cv2.imshow("abrasive", blank_1)
       #cv2.imshow("extraneous", blank2_1)
       img_size=height*width
       whitePx=cv2.countNonZero(blank_1)
       whitePx2=cv2.countNonZero(blank2_1)
       whiteArea=(whitePx/img_size)*100#[%]
       whiteArea2=(whitePx2/img_size)*100#[%]
       with open(filename,mode='a') as f:
            f.write(','+f4)
            f.write(','+str(whiteArea))
            f.write(','+str(whiteArea2)+'\n')


