__Train and Test cifar 100 with one file main.py__
***
Please follow the steps below：

`Step one: setting path` 

1， give your own checkpoint path in main.py (line 45) 

2,  give your own logdir path in main.py (line 58) 

3,  give your own dataset downloading path in main.py (line 94)(line 108)



`Step two: model summary`

1,  pip install tensorboardX 

2,  pip install flopth 

3,  pip install torchsummary 

4,  pip install thop 

5,  %cd path to .../Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100/models 

6,  %run net.py 

(looks like this:

%cd /content/drive/My Drive/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100/models

!python cheapV3.py

)


`Step three: Train and Test`

1,  %cd path to .../Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100 

2，run main.py and choose gpu(cuda device) and give the netname 

(looks like this:

%cd /content/drive/My Drive/-Efficient-Operations-for-Lightweight-Deep-Neural-Networks/Cifar100

%env CUDA_VISIBLE_DEVICES=0,1

%run main.py -net cheapV3

)
