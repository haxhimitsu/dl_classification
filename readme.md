
# Tensorflow image classification 
GPUの認識やmnistの画像分類．
簡単な画像分類を行うやつ．

# gpuの動作確認
* useage
 ```
python3 src/gpu_test_01.py --checktype 1
```
```python3 src/gput_test_01.py -h```でその他の引数を見れる．

 ```--checktype 1```
GPUを使って行列計算を行う．

 ```--checktype 2```
GPUがtensorflow で認識されているかの確認．

 ```--checktype 3```
mnistで簡単な画像分類を学習．

# 画像分類
* datasetの構成
```python
--dataset
    --trains
        --class1
        --class2
        --class3
    --vals
        --class1
        --class2
        --class3
    --tests
        --class1
        --class2
        --class3
```
* input image type

  3ch(color), height : 64, width : 64

  64pixel 以上の画像を使用する時は，データをtensorflowで読み込む時に自動でリサイズされる．



  

## Usage
```python3
python3 src/tf_sample_ver2.0.py  --dataset_path "{your input directory}" --log_dir "{your output directry}"
```
- ```python3 src/tf_sample_ver2.0.py -h```でその他の引数を見れる．以下は主要な引数．
    * ```--max_epochs``` : type =int, default=100
    * ```--save_weight_name``` : type=str,default="test"


* log directoryは自動的に作成される．logには，重みファイルと，テスト用画像の評価結果が保存されている．
* 学習済みの重みファイルがlog　diretoryに存在する場合，学習は行わず評価シーケンスのみを行う．再度学習させたい時は，重みファイルを消去する．

* networkの詳細
    - ```src/utils/myutils.py```内の```def create_network(self,category_num)```が該当する．
    - ```category_num```は分類数．データセットのclass数から算出している．

    - ```input_shape=(32,32,3)```で画像サイズを指定している```(width,height,channel)```の順番．


# Requirement
## Envioroments
* Ubuntu 18.04 LTS
* CUDA 10.0
* cudnn 7.4
## python packeages
* python 3.3～3.7
* opencv 4.4.0
* matplotlib 3.2.2
``pip3 install matplotlib``
* argparse 
``pip3 install argparse``
* tensorflow-gpu 1.14.0
``pip3 install tensorflow_gpu==1.14.0``
* Keras 2.3.1
```pip3 install keras==2.3.1```

# Author
* haxhimitu
* National Institute of Technology, Sasebo college
* it1915[@]st.sasebo.ac.jp
* haxhimitsu.lab[@]gmail.com

# License
this repository is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).