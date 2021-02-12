
# Tensorflow image classification 
簡単な画像分類を行うやつ．

# General specification
* datasetの構成
```python
--dataset
    --train
        --class1
        --class2
        --class3
    --val
        --class1
        --class2
        --class3
    --test
        --class1
        --class2
        --class3
```
* input image type

  3ch(color), height : 64, width : 64

  64pixel 以上の画像を使用する時は，データをtensorflowで読み込む時に自動でリサイズされる．

* log directoryは自動的に作成される．logには，重みファイルと，テスト用画像の評価結果が保存されている．
* 学習済みの重みファイルがlogdiretoryに存在する場合，学習は行わず評価シーケンスのみを行う．再度学習させたい時は，重みファイルを消去する．
  

# Usage
```python3
python3 src/tf_sample_ver2.0.py  --dataset_path "{your input directory}" --log_dir "{your output directry}"
```
```python3 src/tf_sample_ver2.0.py -h```でその他の引数を見れる．以下は主要な引数．
* --max_epochs : type =int, default=100
* --save_weight_name : type=str,default="test"

# Requirement
## Envioroments
* Ubuntu 18.04 LTS
* CUDA 10.0
* cudnn 7.4
## python packeages
* python 3.6.9
* opencv 4.4.0
* matplotlib 3.2.2
``pip3 install matplotlib``
* argparse 
``pip3 install argparse``
* tensorflow-gpu 1.14.0
* Keras 2.3.1

# Author
* haxhimitu
* National Institute of Technology, Sasebo college
* it1915[@]st.sasebo.ac.jp
* haxhimitsu.lab[@]gmail.com

# License
this repository is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).