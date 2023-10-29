import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from models.generators.resblocks import Block


class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator, self).__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),
            dconv_bn_relu(dim * 4, dim * 2),
            dconv_bn_relu(dim * 2, dim),
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y



from links import CategoricalConditionalBatchNorm2d


class dconv_bn_relu_cond(nn.Module):

    def __init__(self, in_ch, out_ch, num_classes=0):
        super(dconv_bn_relu_cond, self).__init__()

        self.num_classes = num_classes

        # Register layrs
        self.conv1 =nn.ConvTranspose2d(in_ch, out_ch, 5, 2,
                            padding=2, output_padding=1, bias=False)
        self.b1 = CategoricalConditionalBatchNorm2d(num_classes, in_ch)
        self.activation = nn.ReLU()       

    def forward(self, x, y=None,**kwargs):
        
        h = self.b1(x, y, **kwargs)
        h = self.conv1(h)
        h = self.activation(h)
        return h

  
class Generator_cond_wgp(nn.Module):
    def __init__(self, in_dim=100, dim=64,num_classes=0):
        super(Generator_cond_wgp, self).__init__()
        

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        
        self.l2 = dconv_bn_relu_cond(dim * 8, dim * 4, num_classes)
        
        self.l3 = dconv_bn_relu_cond(dim * 4, dim * 2, num_classes)
        
        self.l4 = dconv_bn_relu_cond(dim * 2, dim * 1, num_classes)
        
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())
        

    def forward(self, x,label, **kwargs):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)

        y = self.l2(y,label, **kwargs)
        y = self.l3(y,label, **kwargs)
        y = self.l4(y,label, **kwargs)
        y = self.l5(y)

        return y

class dconv_bn_relu_cond2(nn.Module):

    def __init__(self, in_ch, out_ch, num_classes=0):
        super(dconv_bn_relu_cond2, self).__init__()

        self.num_classes = num_classes

        # Register layrs
        self.conv1 =nn.ConvTranspose2d(in_ch, out_ch, 5, 2,
                            padding=2, output_padding=1, bias=False)
        self.b1 = CategoricalConditionalBatchNorm2d(num_classes, out_ch)
        self.activation = nn.ReLU()       

    def forward(self, x, y=None,**kwargs):
        
        h = self.conv1(x)
        h = self.b1(h, y, **kwargs)
        h = self.activation(h)
        return h

  
class Generator_cond_wgp2(nn.Module):
    def __init__(self, in_dim=100, dim=64,num_classes=0):
        super(Generator_cond_wgp2, self).__init__()
        

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU())
        
        self.l2 = dconv_bn_relu_cond2(dim * 8, dim * 4, num_classes)
        
        self.l3 = dconv_bn_relu_cond2(dim * 4, dim * 2, num_classes)
        
        self.l4 = dconv_bn_relu_cond2(dim * 2, dim * 1, num_classes)
        
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())
        

    def forward(self, x,label, **kwargs):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)

        y = self.l2(y,label, **kwargs)
        y = self.l3(y,label, **kwargs)
        y = self.l4(y,label, **kwargs)
        y = self.l5(y)

        return y

class linear_bn_relu_cond(nn.Module):

    def __init__(self, in_dim, dim, num_classes=0):
        super(linear_bn_relu_cond, self).__init__()

        self.num_classes = num_classes
        # self.l1 = nn.Sequential(
        #     nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
        #     nn.BatchNorm1d(dim * 8 * 4 * 4),
        #     nn.ReLU())
        

        # Register layrs
        self.linear1 =nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False)
        self.b1 = CategoricalConditionalBatchNorm2d(num_classes, dim * 8 * 4 * 4)
        self.activation = nn.ReLU()       

    def forward(self, x, y=None,**kwargs):
        
        h = self.linear1(x)
        h = self.b1(h, y, **kwargs)
        h = self.activation(h)
        return h

  
  
class Generator_cond_wgp3(nn.Module):
    def __init__(self, in_dim=100, dim=64,num_classes=0):
        super(Generator_cond_wgp3, self).__init__()
        

        self.l1 = linear_bn_relu_cond(in_dim, dim)
        
        self.l2 = dconv_bn_relu_cond2(dim * 8, dim * 4, num_classes)
        
        self.l3 = dconv_bn_relu_cond2(dim * 4, dim * 2, num_classes)
        
        self.l4 = dconv_bn_relu_cond2(dim * 2, dim * 1, num_classes)
        
        self.l5 = nn.Sequential(
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),
            nn.Sigmoid())
        

    def forward(self, x,label, **kwargs):
        y = self.l1(x,label)
        y = y.view(y.size(0), -1, 4, 4)

        y = self.l2(y,label, **kwargs)
        y = self.l3(y,label, **kwargs)
        y = self.l4(y,label, **kwargs)
        y = self.l5(y)

        return y
# source: https://github.com/opetrova/SemiSupervisedPytorchGAN/blob/master/SemiSupervisedGAN.ipynb
class Generator224(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator224, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( in_dim, dim * 8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 7 x 7
            nn.ConvTranspose2d(dim * 8, dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 14 x 14
            nn.ConvTranspose2d( dim * 4, dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 28 x 28
            nn.ConvTranspose2d( dim * 2, dim, 4, 4, 0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            # state size. (ngf) x 112 x 112
            nn.ConvTranspose2d( dim, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 224 x 224
        )

    def forward(self, input):
        return self.main(input)



class ResNetGenerator(nn.Module):
    """Generator generates 64x64."""

    def __init__(self, num_features=64, dim_z=128, bottom_width=4,
                 activation=F.relu, num_classes=0, distribution='normal'):
        super(ResNetGenerator, self).__init__()
        self.num_features = num_features
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.activation = activation
        self.num_classes = num_classes
        self.distribution = distribution

        self.l1 = nn.Linear(dim_z, 16 * num_features * bottom_width ** 2)

        self.block2 = Block(num_features * 16, num_features * 8,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block3 = Block(num_features * 8, num_features * 4,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block4 = Block(num_features * 4, num_features * 2,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.block5 = Block(num_features * 2, num_features,
                            activation=activation, upsample=True,
                            num_classes=num_classes)
        self.b6 = nn.BatchNorm2d(num_features)
        self.conv6 = nn.Conv2d(num_features, 3, 1, 1)

    def _initialize(self):
        init.xavier_uniform_(self.l1.weight.tensor)
        init.xavier_uniform_(self.conv7.weight.tensor)

    def forward(self, z, y=None, **kwargs):
        h = self.l1(z).view(z.size(0), -1, self.bottom_width, self.bottom_width)
        for i in range(2, 6):
            h = getattr(self, 'block{}'.format(i))(h, y, **kwargs)
        h = self.activation(self.b6(h))
        return torch.tanh(self.conv6(h))
