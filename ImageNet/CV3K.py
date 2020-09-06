from __future__ import print_function

import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
#from utils.dataloaders import *
from tensorboardX import SummaryWriter
################################################################################################# net
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

#from thop import profile
#from torchsummaryX import summary
#from flopth import flopth


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x

class CF(nn.Module):
   def __init__(self, groups):
       super(CF,self).__init__()
       self.groups = groups
    

   def forward(self, x):
     batchsize, num_channels, height, width = x.data.size()

     channels_per_group = num_channels // self.groups
    
     # reshape
     x = x.view(batchsize, self.groups, 
        channels_per_group, height, width)

     # transpose
     x = torch.transpose(x, 1, 2).contiguous()

     # flatten
     x = x.view(batchsize, -1, height, width)
     return x


def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num

class H_sigmoid(nn.Module):
    '''
    hard sigmoid
    '''
    def __init__(self, inplace=True):
        super(H_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class H_swish(nn.Module):
    '''
    hard swish
    '''
    def __init__(self, inplace=True):
        super(H_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class SEModule(nn.Module):
    '''
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            #nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            reuse4(),
            H_sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y
    
class reuse(nn.Module):
   def __init__(self, in_channels_num, exp_size, BN_momentum):
       super(reuse,self).__init__()
       self.exp_size = exp_size
       self.in_channels_num = in_channels_num
       self.reuse3 = reuse3()
       self.reuse4 = reuse4()
       self.reuse6 = reuse6()

       g = (self.exp_size)//(self.in_channels_num)

       self.conv = nn.Sequential(
         nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=1, stride=1, padding=0, groups=(exp_size*3)//(in_channels_num), bias=False),
         nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum)
       )

   def forward(self, x):
     if (self.exp_size)//(self.in_channels_num) == 3:
       x1 = self.reuse3(x)
       x2 = self.conv(x1)
       return x2+x1
     if (self.exp_size)//(self.in_channels_num) == 4:
       x3 = self.reuse4(x)
       x4 = self.conv(x3)
       return x4+x3
     if (self.exp_size)//(self.in_channels_num) == 6:
       x5 = self.reuse6(x)
       x6 = self.conv(x5)
       return x6+x5

class reuse6A(nn.Module):
   def __init__(self,in_channels_num):
       super(reuse6A,self).__init__()

   def forward(self, x):
     batch_size, channel_num, _, _ = x.size()
     x = channel_shuffle(x,6)
     out = torch.cat((x,x,x,x,x,x),dim=1)
     return out

class reuse6(nn.Module):
   def __init__(self):
       super(reuse6,self).__init__()

   def forward(self, x):
     out = torch.cat((x,x,x,x,x,x),dim=1)
     return out

class reuse4(nn.Module):
   def __init__(self):
       super(reuse4,self).__init__()

   def forward(self, x):
     out = torch.cat((x,x,x,x),dim=1)
     return out

class reuse3A(nn.Module):
   def __init__(self,in_channels_num):
       super(reuse3A,self).__init__()

   def forward(self, x):
     batch_size, channel_num, _, _ = x.size()
     x = channel_shuffle(x,3)
     out = torch.cat((x,x,x),dim=1)
     return out

class reuse3(nn.Module):
   def __init__(self):
       super(reuse3,self).__init__()

   def forward(self, x):
     out = torch.cat((x,x,x),dim=1)
     return out

class reuse2(nn.Module):
   def __init__(self):
       super(reuse2,self).__init__()

   def forward(self, x):
     out = torch.cat((x,x),dim=1)
     return out

class Bottleneck(nn.Module):
    '''
    The basic unit of MobileNetV3
    '''
    def __init__(self, in_channels_num, exp_size, out_channels_num, kernel_size, stride, use_SE, NL, BN_momentum):
        '''
        use_SE: True or False -- use SE Module or not
        NL: nonlinearity, 'RE' or 'HS'
        '''
        super(Bottleneck, self).__init__()

        assert stride in [1, 2]
        NL = NL.upper()
        assert NL in ['RE', 'HS']

        use_HS = NL == 'HS'
        
        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)

        if exp_size == in_channels_num:
            # Without expansion, the first depthwise convolution is omitted
            self.conv = nn.Sequential(
                # Depthwise Convolution
                nn.Conv2d(in_channels=in_channels_num, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=in_channels_num, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                    nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
            )
        else:
            # With expansion
            self.conv = nn.Sequential(
                # Pointwise Convolution for expansion
                reuse(in_channels_num, exp_size, BN_momentum),
                H_swish() if use_HS else nn.ReLU(inplace=True),

                # Depthwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=exp_size, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, groups=exp_size, bias=False),
                nn.BatchNorm2d(num_features=exp_size, momentum=BN_momentum),

                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
                # Linear Pointwise Convolution
                nn.Conv2d(in_channels=exp_size, out_channels=out_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.Sequential(OrderedDict([('lastBN', nn.BatchNorm2d(num_features=out_channels_num))])) if self.use_residual else
                    nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum),
                #CF
                CF(exp_size//in_channels_num),
            )

    def forward(self, x):
        if self.use_residual:
            return self.conv(x) + x
        else:
            return self.conv(x)


class CV3K(nn.Module):
    def __init__(self, mode='large', classes_num=1000, input_size=224, width_multiplier=1.0, dropout=0.2, BN_momentum=0.1, zero_gamma=False):
        '''
        configs: setting of the model
        mode: type of the model, 'large' or 'small'
        '''
        super(CV3K, self).__init__()

        mode = mode.lower()
        assert mode in ['large', 'small']
        s = 2
        #if input_size == 32 or input_size == 56:
         #   # using cifar-10, cifar-100 or Tiny-ImageNet
          #  s = 1

        # setting of the model
        if mode == 'large':
            # Configuration of a MobileNetV3-Large Model
            configs = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 24, 24, False, 'RE', 1],
                [3, 72, 24, False, 'RE', s],
                [3, 72, 24, False, 'RE', 1],
                [5, 144, 48, True, 'RE', 2],
                [5, 144, 48, True, 'RE', 1],
                [5, 144, 48, True, 'RE', 1],
                [3, 288, 72, False, 'HS', 2],
                [3, 216, 72, False, 'HS', 1],
                [3, 216, 72, False, 'HS', 1],
                [3, 216, 72, False, 'HS', 1],
                [3, 432, 108, True, 'HS', 1],
                [3, 648, 108, True, 'HS', 1],
                [5, 648, 162, True, 'HS', 2],
                [5, 972, 162, True, 'HS', 1],
                [5, 972, 162, True, 'HS', 1]
            ]
        elif mode == 'small':
            # Configuration of a MobileNetV3-Small Model
            configs = [
                #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
                [3, 16, 16, True, 'RE', s],
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1],
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1],
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1]
            ]

        first_channels_num = 24

        # last_channels_num = 1280
        # according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # if small -- 1024, if large -- 1280
        last_channels_num = 1280 if mode == 'large' else 1024

        divisor = 1

        ########################################################################################################################
        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier, divisor) if width_multiplier > 1 else last_channels_num
        feature_extraction_layers = []
        first_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=input_channels_num, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(num_features=input_channels_num, momentum=BN_momentum),
            H_swish()
        )
        feature_extraction_layers.append(first_layer)
        # Overlay of multiple bottleneck structures
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            exp_size = _ensure_divisible(exp_size * width_multiplier, divisor)
            feature_extraction_layers.append(Bottleneck(input_channels_num, exp_size, output_channels_num, kernel_size, stride, use_SE, NL, BN_momentum))
            input_channels_num = output_channels_num
        
        # the last stage
        last_stage_channels_num = _ensure_divisible(exp_size * width_multiplier, divisor)
        last_stage_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=input_channels_num, out_channels=last_stage_channels_num, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_features=last_stage_channels_num, momentum=BN_momentum),
                H_swish()
            )
        feature_extraction_layers.append(last_stage_layer1)

        # SE Module
        # remove the last SE Module according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # feature_extraction_layers.append(SEModule(last_stage_channels_num) if mode == 'small' else nn.Sequential())

        feature_extraction_layers.append(nn.AdaptiveAvgPool2d(1))
        feature_extraction_layers.append(nn.Conv2d(in_channels=last_stage_channels_num, out_channels=last_channels_num, kernel_size=1, stride=1, padding=0, bias=False))
        feature_extraction_layers.append(H_swish())

        self.features = nn.Sequential(*feature_extraction_layers)

        ########################################################################################################################
        # Classification part
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channels_num, classes_num)
        )

        ########################################################################################################################
        # Initialize the weights
        self._initialize_weights(zero_gamma)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, zero_gamma):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        if zero_gamma:
            for m in self.modules():
	            if hasattr(m, 'lastBN'):
	                nn.init.constant_(m.lastBN.weight, 0.0)




def test():
    net = CV3K()
    print(net)

    x = torch.randn(2,3,224,224)
    y = net(x)
    #print(y.size())
    
    #net=net.cuda()
    summary(net,x)
    #print(flopth(net,in_size=(3,224,224)))

#test()
################################################################################################ functions from utils

################################################################################################ parse argument
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-d', '--data', metavar='DIR',default='/input/ILSVRC2012',
                    help='path to dataset')
parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch',
                    )
#parser.add_argument('-m', '--model' type=str, default='cheapV3_Reuse_Shuffle6_ImageNet', required=True, help='net type')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--lr-decay', type=str, default='cos',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('-c', '--checkpoint', default='/ImageNet/CV3K_checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')

parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--input-size', type=int, default=224, help='MobileNet model input resolution')
parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
#################################################################################################  data loading
DATA_BACKEND_CHOICES = ['pytorch']
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
args = parser.parse_args()

def get_pytorch_train_loader(data_path, batch_size, workers=4, input_size=224):
    traindir = os.path.join(data_path, 'train')
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

    return train_loader, len(train_loader)

def get_pytorch_val_loader(data_path, batch_size, workers=4, input_size=224):
    valdir = os.path.join(data_path, 'val')
    val_dataset = datasets.ImageFolder(
            valdir, transforms.Compose([
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
                ]))


    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

    return val_loader, len(val_loader)

train_loader, train_loader_len = get_pytorch_train_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)
val_loader, val_loader_len = get_pytorch_val_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)

########################################################################################################################## model


################################################################################################################## global settings

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_prec1 = 0



##################################################################################################################### main



def main():
    global args, best_prec1
    args = parser.parse_args()
    
    # create model
    print("=> creating model")
    model = CV3K()
    print(model)
    
    
    #gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model) # make parallel
        #cudnn.benchmark = True
        
    cudnn.benchmark = True
    

    
    #loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    
     # optionally resume from a checkpoint
    title = 'ImageNet-' + 'CV3K'
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'CV3K_log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'CV3K_log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
        
        
        # Data loading code
    if args.data_backend == 'pytorch':
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
    
    train_loader, train_loader_len = get_train_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)
    val_loader, val_loader_len = get_val_loader(args.data, args.batch_size, workers=args.workers, input_size=args.input_size)

    if args.evaluate:
        from collections import OrderedDict
        if os.path.isfile(args.weight):
            print("=> loading pretrained weight '{}'".format(args.weight))
            source_state = torch.load(args.weight)
            target_state = OrderedDict()
            for k, v in source_state.items():
                if k[:7] != 'module.':
                    k = 'module.' + k
                target_state[k] = v
            model.load_state_dict(target_state)
        else:
            print("=> no weight found at '{}'".format(args.weight))

        validate(val_loader, val_loader_len, model, criterion)
        return

    # visualization
    writer = SummaryWriter(os.path.join(args.checkpoint, 'CV3K_logs'))

    for epoch in range(args.start_epoch, args.epochs):

        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        # train for one epoch
        train_loss, train_acc = train(train_loader, train_loader_len, model, criterion, optimizer, epoch)
        print(train_acc)
        sys.stdout.flush()
        print(train_loss)
        sys.stdout.flush()

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, val_loader_len, model, criterion)
        print(val_loss)
        sys.stdout.flush()
        print(prec1)
        sys.stdout.flush()

        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('CV3K_learning rate', lr, epoch + 1)
        writer.add_scalars('CV3K_loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('CV3K_accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))
    writer.close()

    print('Best accuracy:')
    print(best_prec1)


def train(train_loader, train_loader_len, model, criterion, optimizer, epoch):
    

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    for i, (inputs, targets) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)
        
        
        inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        targets = targets - 1


        # compute output
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=train_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
             )
        sys.stdout.flush()


        # plot progress
        '''bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=train_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()'''
    return (losses.avg, top1.avg)


def validate(val_loader, val_loader_len, model, criterion):    
    global best_acc
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
            data_time.update(time.time() - end)
        
        
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
            targets = targets - 1
        


            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss        
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
            

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
            print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=i + 1,
                    size=val_loader_len,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )    )
            sys.stdout.flush()
        return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='CV3K_checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'CV3K_model_best.pth.tar'))

def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


from math import cos, pi
def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()














