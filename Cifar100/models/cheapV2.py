import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from thop import profile
from torchsummary import summary
from flopth import flopth



def reuse6(x):

  x=torch.cat((x,x,x,x,x,x),dim=1)

  return x


def reuse4(x):

  x=torch.cat((x,x,x,x),dim=1)

  return x

class conv1(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, padding):
        super(conv1, self).__init__()  
        
        inplanes=3      
        planes=32
        stride=1
        kernel_size1=3
        padding=1
      
       
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn=  nn.BatchNorm2d(planes)

    def forward(self, x):

        out = F.relu(self.bn(self.conv(x)))

        return out

#expansion size = 1
class Block1(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block1, self).__init__()
        self.stride = stride

        in_planes = 32
        planes = in_planes
        out_planes = 16
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
       
    def forward(self, x):
        out = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(out))

        out1,out2=torch.split(x,16,dim=1)
        
        out = out + out1
        return out


class Block2(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block2, self).__init__()
        self.stride = stride

        in_planes = 16
        planes = in_planes*6
        out_planes = 24
        stride = 1


        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))

        out1,out2=torch.split(x,8,dim=1)
        

        out4=torch.cat((x,out1),dim=1)
        out = out + out4
        return out


class Block3(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block3, self).__init__()
        self.stride = stride

        in_planes = 24
        planes = in_planes*6
        out_planes = 24
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out

class Block4(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block4, self).__init__()
        self.stride = stride

        in_planes = 24
        planes = in_planes*6
        out_planes = 32
        stride = 1

        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
       
    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        return out


class Block5(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block5, self).__init__()
        self.stride = stride

        in_planes = 32
        planes = in_planes*6
        out_planes = 32
        stride = 1
        

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out


class Block6(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block6, self).__init__()
        self.stride = stride

        in_planes = 32
        planes = in_planes*6
        out_planes = 32
        stride = 1
        

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x
        return out


class Block7(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block7, self).__init__()
        self.stride = stride

        in_planes = 32
        planes = in_planes*6
        out_planes = 64
        stride = 1

        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        return out

class Block8(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block8, self).__init__()
        self.stride = stride

        in_planes = 64
        planes = in_planes*6
        out_planes = 64
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out



class Block9(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block9, self).__init__()
        self.stride = stride

        in_planes = 64
        planes = in_planes*6
        out_planes = 64
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out



class Block10(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block10, self).__init__()
        self.stride = stride

        in_planes = 64
        planes = in_planes*6
        out_planes = 64
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out



class Block11(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block11, self).__init__()
        self.stride = stride

        in_planes = 64
        planes = in_planes*6
        out_planes = 96
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))


        out1,out2=torch.split(x,32,dim=1)
        

        out4=torch.cat((x,out2),dim=1)
        out = out + out4 
        return out


class Block12(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block12, self).__init__()
        self.stride = stride

        in_planes = 96
        planes = in_planes*6
        out_planes = 96
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out


class Block13(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block13, self).__init__()
        self.stride = stride

        in_planes = 96
        planes = in_planes*6
        out_planes = 96
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out


class Block14(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block14, self).__init__()
        self.stride = stride

        in_planes = 96
        planes = in_planes*6
        out_planes = 160
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        return out

class Block15(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block15, self).__init__()
        self.stride = stride

        in_planes = 160
        planes = in_planes*6
        out_planes = 160
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out

class Block16(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block16, self).__init__()
        self.stride = stride

        in_planes = 160
        planes = in_planes*6
        out_planes = 160
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))
        out = out + x 
        return out



class Block17(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Block17, self).__init__()
        self.stride = stride

        in_planes = 160
        planes = in_planes*6
        out_planes = 320
        stride = 1
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        

        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)


    def forward(self, x):

        x1=reuse6(x)
        out = F.relu(self.bn2(self.conv2(x1)))
        out = self.bn3(self.conv3(out))

        out1=torch.cat((x,x),dim=1)
        out = out + out1
        return out


class cheapV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
        
    def __init__(self, num_classes=100):
        super(cheapV2, self).__init__()

        self.conv1 = conv1(3, 32, 3, 1, 1)

        self.Block1 = Block1(32, 16, 3, 1)
        self.Block2 = Block2(16, 24, 3, 1)
        self.Block3 = Block3(24, 24, 3, 1)
        self.Block4 = Block4(24, 32, 3, 2)
        self.Block5 = Block5(32, 32, 3, 1)
        self.Block6 = Block6(32, 32, 3, 1)
        self.Block7 = Block7(32, 64, 3, 2)
        self.Block8 = Block8(64, 64, 3, 1)
        self.Block9 = Block9(64, 64, 3, 1)
        self.Block10 = Block10(64, 64, 3, 1)
        self.Block11 = Block11(64, 96, 3, 1)
        self.Block12 = Block12(96, 96, 3, 1)
        self.Block13 = Block13(96, 96, 3, 1)
        self.Block14 = Block14(96, 160, 3, 2)
        self.Block15 = Block15(160, 160, 3, 1)
        self.Block16 = Block16(160, 160, 3, 1)
        self.Block17 = Block17(160, 320, 3, 1)
        
        #self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        #self.bn2 = nn.BatchNorm2d(1280)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1280, num_classes)



    def forward(self, x):
       
        out=self.conv1(x)
        out = self.Block1(out)
        out = self.Block2(out)
        out = self.Block3(out)
        out = self.Block4(out)
        out = self.Block5(out)
        out = self.Block6(out)
        out = self.Block7(out)
        out = self.Block8(out)
        out = self.Block9(out)
        out = self.Block10(out)
        out = self.Block11(out)
        out = self.Block12(out)
        out = self.Block13(out)
        out = self.Block14(out)
        out = self.Block15(out)
        out = self.Block16(out)
        out = self.Block17(out)

        out = reuse4(out)
        out = self.avgpool(out)        
        out= out.view(out.size(0),-1)      
        out = self.linear(out)
        
        return out


def test():
    net = cheapV2()
    print(net)

    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())
    
    net=net.cuda()
    summary(net,input_size=(3, 224, 224))
    print(flopth(net,in_size=(3,224,224)))

test()



















