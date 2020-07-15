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

## If you want to train our model on cifar 100, Please follow the steps below：

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


## If you want to test the best models on cifar 100 after training:

looks like this:

`%cd /content/drive/My Drive/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100`

`%run test.py -net MobileNetV3 -weights 'path/to/checkpoint/models/MobileNetV3-165-best.pth`




## Model Description

1. `MobileNetV3`: the original net from google. (mode='large', width_multiplier=1.0, dropout=0.2, BN_momentum=0.1, zero_gamma=False)

2. `cheapV3`: we modified the original net `MobileNetV3` with the pure `Reuse` operation.

3. `cheapV3_shuffle`: we modified the original net `MobileNetV3` with the ` upgrade Reuse` operation.


## Results Table the for Test on cifar100\

#### (we rank the best models by top1 error, not by top5 error，which means the following top5 error is `not the best performance of our model`)

|  cifar100   | top1 error   | top5 error  corresponding  |
| ---------- | :-----------:  | :-----------: |
| MobileNetV2   | 31.5     | 9.01     |
| cheapV2     | 28.76     | 8.13     |
| MobileNetV3     | 26.5    | 6.12    |
| cheapV3     | 25.88     | 6.7    |
| cheapV3_shuffle(g=input_channel)     | 25.74   | 6.73    |
| cheapV3_shuffle(g=6)     | 24.82  | 7.17   |

## Runs on Tensorboard(`MobileNetV3` `cheapV3` `cheapV3_shuffle`)

#### Test Accuracy
![image](https://github.com/kai-pixel/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/blob/master/IMG/Test%20Accuracy(cifar100).png)

#### Test Accuracy(local zoom)
![image](https://github.com/kai-pixel/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/blob/master/IMG/local%20zoom%20to%20the%20Test%20Accuracy.png)

#### Train Loss
![image](https://github.com/kai-pixel/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/blob/master/IMG/Train%20Loss(cifar100).png)
