# Do networks really need so many different feature maps for deep learning ?

## Introduction

* This project aims to explore how to use feature maps efficiently.

- A new free operation ` Reuse` is proposed to replace the 1x1 convolution for increasing the dimension of feature map.

- We applied our new proposed operation on three well-known networks, they are ` MobileNetV2` ` MobileNetV3` ` SqueezeNet` 

- The datasets we have already used by now are ` cifar10`  ` cifar100`  `tiny-imagenet ` 

* We are further testing models with our proposed free operation `Reuse` on dataset `Imagenet`

## Environment

1.  Pytorch 1.0

2.  Cuda 9

3.  Cudnn 7

4.  Python 3.6

***

## If you want to use our model on cifar 100, Please follow the steps below：

#### `Step one: setting path` 

1.  give your own checkpoint path in main.py (line 45) 

2.  give your own logdir path in main.py (line 58) 

3.  give your own dataset downloading path in main.py (line 94)(line 108)



#### `Step two: model summary(optional)`


1.  pip install flopth 

2.  pip install torchsummary 

3.  pip install thop 

4.  %cd path to .../Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100/models 

5.  %run net.py 


looks like this:

`%cd /content/drive/My Drive/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100/models`

`%run cheapV3.py`




#### `Step three: Train and Test`

1.  %cd path to .../Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100 

2.  run main.py and choose gpu(cuda device) and give the netname 


looks like this:

`%cd /content/drive/My Drive/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100`

`%env CUDA_VISIBLE_DEVICES=0,1`

`%run main.py -net cheapV3`


## Results Table

|  cifar100   | top1 error   | top5 error   |
| ---------- | :-----------:  | :-----------: |
| MobileNetV2   | 31.5     | 9.01     |
| cheapV2     | 28.76     | 8.13     |
| MobileNetV3     | 26.5    | 6.12    |
| cheapV3     | 25.88     | 6.7    |
| cheapV3_shuffle     | 25.74   | 6.73    |
## Runs on Tensorboard

![Test Accuracy][1](https://github.com/kai-pixel/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/blob/master/IMG/Test%20Accuracy.png)

![Local Zoom to Test Accuracy][2](https://github.com/kai-pixel/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/blob/master/IMG/local%20zoom%20to%20Test%20Accuracy.png)
