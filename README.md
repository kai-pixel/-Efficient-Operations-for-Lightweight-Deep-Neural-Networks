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
| target_model-version   | 31.5     | 9.01     | 9.01     |
| batch-size   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |
| MobileNetV2   | 31.5     | 9.01     | 9.01     |






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
