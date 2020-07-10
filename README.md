# Train and Test cifar 100 

## Introduction



## Environment

1.  Pytorch 1.0

2.  Cuda 9

3.  Cudnn 7

4.  Python 3.6

***
## Please follow the steps below：

`Step one: setting path` 

1.  give your own checkpoint path in main.py (line 45) 

2.  give your own logdir path in main.py (line 58) 

3.  give your own dataset downloading path in main.py (line 94)(line 108)



`Step two: model summary(optional)`


1.  pip install flopth 

2.  pip install torchsummary 

3.  pip install thop 

4.  %cd path to .../Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100/models 

5.  %run net.py 

looks like this:

`%cd /content/drive/My Drive/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100/models`

`%run cheapV3.py`




`Step three: Train and Test`

1.  %cd path to .../Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100 

2.  run main.py and choose gpu(cuda device) and give the netname 

looks like this:

`%cd /content/drive/My Drive/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100`

`%env CUDA_VISIBLE_DEVICES=0,1`

`%run main.py -net cheapV3`


