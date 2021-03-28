# Optimization targets MobileNetV2/3

## Introduction

* This project aims to improve the efficiency of MobileNetV2/3.

- The related datasets are Cifar10/100 and ImageNet. The corresponding task is Image-Classification.

-  In this project, I proposed several cheap operations, based on an extremely idea, to modify the target model MobileNetV2 / 3. Through the modifications, the accuracy of the top 1 of the classification task on the Cifar 100 can be increased maximum by 2.42%, and the parameter cost can be simultaneously reduced by up to 47%. For the same task on ImageNet, my modification to MobileNetV3 can also bring performance improvements while the costs slightly reduced.


## Environment

1.  Pytorch 1.0

2.  Cuda 9

3.  Cudnn 7

4.  Python 3.6

***

## Train-Policy
 

|  settings   | Cifar10   | Cifar100  | ImageNet  |
| ---------- | :-----------:  | :-----------: | :-----------: |
| target_model-version   | x 1.0     | x 1.0     | x 1.0     |
| batch-size   | 128    | 128     | 256    |
| EMA decay   | 0    | 0     | 0     |
| initial-lr  | 0.35     | 0.35     | 0.05     |
| Lr-decay   | cos     | cos     | cos     |
| min-lr   | 0    | 0     | 0     |
| warmup-epochs   | 5     | 5     | 0     |
| weight-decay   | 6e-5     | 6e-5     | 4e-5     |
| epochs   | 400     | 400     | 150     |
| workers   | 2     | 2     | 8     |
| label-smooth   | -     | 0.1    | -     |
| optimizer   | SGD     | SGD     | SGD     |
| drop out   | 0     | 0     | 0.2    |


## Where to apply my cheap operations in the target Models

1. the First-Conv

2. the 1x1 convolution in the bottlenecks before the depthwise separable convolution

3. the 1x1 convolution in the bottlenecks after the depthwise separable convolution

4. the shortcut of MobileNetV2

5. the Last-Stage

6. SE-Module in MobileNetV3

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
