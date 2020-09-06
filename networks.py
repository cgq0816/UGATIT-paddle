#import torch
#import torch.nn as nn
#from torch.nn.parameter import Parameter


import paddle.fluid as fluid
import numpy as np
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph import Sequential
import paddle.fluid.dygraph.nn as nn
#from paddle.fluid.Tensor import tensor

class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

      
        DownBlock = []
        DownBlock += [
                      ReflectionPad2d(3),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(),
                      ReLU(True)
                      ]
        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [
                          ReflectionPad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(),
                          ReLU(True)                        
                          ]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True)
                  ]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(True)
                  ]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [
                         Upsample(),
                         ReflectionPad2d(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU(True)
                         ]

        UpBlock2 += [
                     ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False),
                     Tanh()
                     ]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
  
        x = self.DownBlock(input)

        gap=fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) #torch.Size([1, 1])
        gap_logit = self.gap_fc(gap)
        gap_weight = list(self.gap_fc.parameters())[0] #torch.Size([1, 256])
        gap_weight=fluid.layers.reshape(gap_weight, shape=[x.shape[0], -1])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])
        gap = x * gap_weight    #torch.Size([1, 256, 64, 64])

        gmp=fluid.layers.adaptive_pool2d(x,1,pool_type='max')
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight=fluid.layers.reshape(gmp_weight, shape=[x.shape[0], -1])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])     
        gmp = x * gmp_weight  #torch.Size([1, 256, 64, 64])

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)   #torch.Size([1, 512, 64, 64])      
        #x = self.conv1x1(x) #torch.Size([1, 256, 64, 64])
        x = self.relu(self.conv1x1(x))
        #torch.Size([1, 256, 64, 64])
        #heatmap = torch.sum(x, dim=1, keepdim=True)
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        #heatmap torch.Size([1, 1, 64, 64])
        if self.light:
            x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
            x_=fluid.layers.reshape(x_, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        else:
            x_=fluid.layers.reshape(x, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        gamma, beta = self.gamma(x_), self.beta(x_)
        # gamma torch.Size([1, 256]) beta torch.Size([1, 256])

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)
        #out torch.Size([1, 3, 256, 256]) cam_logit torch.Size([1, 2])  heatmap torch.Size([1, 1, 64, 64])
        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [
                       ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(),
                       ReLU(True)                         
                       ]

        conv_block += [
                       ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm()]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU(True)

        self.pad2 = ReflectionPad2d(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x
class adaILN(fluid.dygraph.Layer):

    def __init__(self, in_channels, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho =self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.9))

    def var(self, input, dim,unbiased=True):
        rank =len(input.shape)
        dims=dim if dim!=None and dim != [] else range(rank)
        dims = [e if e >= 0 else e + rank for e in dims]
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        inp_shape=input.shape
        if unbiased:
            n= 1
            for i in dims:
                n*=inp_shape[i]
            factor=n/(n-1.0) if n> 1.0 else 0.0
            tmp*=factor
        return tmp         

    def forward(self, input, gamma, beta):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1 - self.rho)*out_ln
        out=out*fluid.layers.unsqueeze(fluid.layers.unsqueeze(gamma,2),3)+fluid.layers.unsqueeze(fluid.layers.unsqueeze(beta,2),3)
        return out

class ILN(fluid.dygraph.Layer):

    def __init__(self, in_channels, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
        self.gamma = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
      
        
    def var(self, input, dim,unbiased=True):
        rank =len(input.shape)
        dims=dim if dim!=None and dim != [] else range(rank)
        dims = [e if e >= 0 else e + rank for e in dims]
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        inp_shape=input.shape
        if unbiased:
            n= 1
            for i in dims:
                n*=inp_shape[i]
            factor=n/(n-1.0) if n> 1.0 else 0.0
            tmp*=factor
        return tmp       

    def forward(self, input):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1 - self.rho)*out_ln
        out = out * self.gamma + self.beta
        return out

# class adaILN(fluid.dygraph.Layer):
#     def __init__(self, num_features, eps=1e-5):
#         super(adaILN, self).__init__()
#         self.eps = eps
#         # self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
#         # self.rho.data.fill_(0.9)
#         self.rho = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=fluid.initializer.Constant(0.9))

#     def forward(self, input, gamma, beta):
#         # in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
#         # out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
#         # ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
#         # out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
#         # out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
#         # out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
#         in_mean = mean(input, dim=[2, 3], keep_dim=True)
#         in_var = var(input, dim=[2, 3], keep_dim=True)
#         out_in = (input - in_mean) / layers.sqrt(in_var + self.eps)
#         ln_mean = mean(input, dim=[1, 2, 3], keep_dim=True)
#         ln_var = var(input, dim=[1, 2, 3], keep_dim=True)
#         out_ln = (input - ln_mean) / layers.sqrt(ln_var + self.eps)
#         # rho_expand = layers.expand(self.rho, [input.shape[0], 1, 1, 1])
#         # out = rho_expand * out_in + (1-rho_expand) * out_ln
#         out = self.rho * out_in + (1 - self.rho) * out_ln
#         out = out * layers.unsqueeze(layers.unsqueeze(gamma, 2), 3) + layers.unsqueeze(layers.unsqueeze(beta, 2), 3)
#         return out

# def var(input, dim=None, keep_dim=False, unbiased=True, name=None):
#     rank = len(input.shape)
#     dims = dim if dim != None and dim != [] else range(rank)
#     dims = [e if e >= 0 else e + rank for e in dims]
#     inp_shape = input.shape
#     mean = layers.reduce_mean(input, dim=dim, keep_dim=True, name=name)
#     tmp = layers.reduce_mean((input - mean)**2, dim=dim, keep_dim=keep_dim, name=name)
#     if unbiased:
#         n = 1
#         for i in dims:
#             n *= inp_shape[i]
#         factor = n / (n - 1.0) if n > 1.0 else 0.0
#         tmp *= factor
#     return tmp

# class adaILN(fluid.dygraph.Layer):
#     def __init__(self, num_features, eps=1e-5):
#         super(adaILN, self).__init__()
#         self.eps = eps
#         # self.rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(0.9))
#         # self.rho = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32',is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(0.9))
#         self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.9, dtype='float32')
#         #self.rho = Parameter(fluid.Tensor(1, num_features, 1, 1))
#         #self.rho = fluid.Tensor()
#         #self.rho.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())        
#         #self.rho = fluid.Tensor(1, num_features, 1, 1)
#         #self.rho=np.ndarray([1, num_features, 1, 1])
#         #self.rho.data.fill_(0.9)

#     def forward(self, input, gamma, beta):
#         #torch.Size([1, 256, 64, 64])
#         ninput=input.numpy()
#         #self.rho=self.rho.numpy()
#         in_mean=np.mean(ninput, axis=(2, 3), keepdims=True)
#         in_var=np.var(ninput, axis=(2, 3), keepdims=True)
#         out_in = (ninput - in_mean) / np.sqrt(in_var + self.eps)
#         ln_mean, ln_var = np.mean(ninput, axis=(1, 2, 3), keepdims=True), np.var(ninput, axis=(1, 2, 3), keepdims=True)
#         out_ln = (ninput - ln_mean) / np.sqrt(ln_var + self.eps)
#         out_in = fluid.dygraph.base.to_variable(out_in)
#         out_ln = fluid.dygraph.base.to_variable(out_ln)
#         ninput = fluid.dygraph.base.to_variable(ninput)
#         #out = fluid.dygraph.base.to_variable(out)
#         out = self.rho * out_in + (1-self.rho) * out_ln
#         # print(self.rho)
#         #t = fluid.Tensor()
#         #t.set(out, fluid.CPUPlace())   
#         gamma = fluid.layers.unsqueeze(input=gamma, axes=[2])
#         gamma = fluid.layers.unsqueeze(input=gamma, axes=[3])   
#         beta = fluid.layers.unsqueeze(input=beta, axes=[2])
#         beta = fluid.layers.unsqueeze(input=beta, axes=[3])  
#         out = out *gamma+beta
#         #in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
#         #out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
#         #ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
#         #out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
#         #out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
#         #out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
#         #out torch.Size([1, 256, 64, 64])
#         return out


# class ILN(fluid.dygraph.Layer):
#     def __init__(self, num_features, eps=1e-5):
#         super(ILN, self).__init__()
#         self.eps = eps
#         #self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
#         #self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
#         #self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
#         #self.rho = fluid.Tensor(1, num_features, 1, 1)
#         #self.gamma = fluid.Tensor(1, num_features, 1, 1)
#         #self.beta = fluid.Tensor(1, num_features, 1, 1)  
#         self.rho = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.0, dtype='float32')
#         self.gamma = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=1.0, dtype='float32')
#         self.beta = fluid.layers.fill_constant(shape=[1, num_features, 1, 1], value=0.0, dtype='float32')
#         # self.rho = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(0.0))
#         # self.gamma = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(1.0))
#         # self.beta = fluid.layers.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(0.0))
#         # self.rho = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(0.0))
#         # self.gamma = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(1.0))
#         # self.beta = self.create_parameter(shape=[1, num_features, 1, 1], dtype='float32', is_bias=True,default_initializer=fluid.initializer.ConstantInitializer(0.0))
#         #self.rho = fluid.Tensor()
#         #self.rho.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())       
#         #self.gamma = fluid.Tensor()
#         #self.gamma.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())   
#         #self.beta = fluid.Tensor()
#         #self.beta.set(np.ndarray([1, num_features, 1, 1]), fluid.CPUPlace())   
        
#         #self.rho.data.fill_(0.0)
#         #self.gamma.data.fill_(1.0)
#         #self.beta.data.fill_(0.0)

#     def forward(self, input):
#         #torch.Size([1, 128, 128, 128])
#         ninput=input.numpy()
#         #self.rho=self.rho.numpy()
#         #self.gamma=self.gamma.numpy()
#         #self.beta=self.beta.numpy()
#         in_mean=np.mean(ninput, axis=(2, 3), keepdims=True)
#         in_var=np.var(ninput, axis=(2, 3), keepdims=True)
#         out_in = (ninput - in_mean) / np.sqrt(in_var + self.eps)
#         ln_mean, ln_var = np.mean(ninput, axis=(1, 2, 3), keepdims=True), np.var(ninput, axis=(1, 2, 3), keepdims=True)
#         out_ln = (ninput - ln_mean) / np.sqrt(ln_var + self.eps)
#         out_in = fluid.dygraph.base.to_variable(out_in)
#         out_ln = fluid.dygraph.base.to_variable(out_ln)
#         ninput = fluid.dygraph.base.to_variable(ninput)        
#         out = self.rho * out_in + (1-self.rho) * out_ln
#         out = out * self.gamma + self.beta
#         #out = fluid.dygraph.base.to_variable(out)
#         #t = fluid.Tensor()
#         #t.set(out, fluid.CPUPlace()) 
        
#         #in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
#         #out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
#         #ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
#         #out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
#         #out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
#         #out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
#         #out torch.Size([1, 128, 128, 128])
#         return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [
                 ReflectionPad2d(1),
                 Spectralnorm(Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True)),
                 LeakyReLU(0.2, inplace=True)
                 
        ]        

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [
                      ReflectionPad2d(1),
                      Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True)),
                      LeakyReLU(0.2, inplace=True)
                      ]        

        mult = 2 ** (n_layers - 2 - 1)
        model += [
                  ReflectionPad2d(1),
                  Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True)),
                  LeakyReLU(0.2, inplace=True)
                  ]        

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 =Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu =LeakyReLU(0.2, True)        

        self.pad=ReflectionPad2d(1)
        self.conv = Spectralnorm(Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))   

        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input) #[1, 2048, 2, 2]

        #gap = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
        gap=fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) 
        gap_logit = self.gap_fc(gap)#torch.Size([1, 1])
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight=fluid.layers.reshape(gap_weight, shape=[x.shape[0], -1])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])   
        gap = x * gap_weight #[1, 2048, 2, 2]
        #gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        #gmp = fluid.layers.adaptive_pool2d(x, 1,pool_type='max')
        gmp=fluid.layers.adaptive_pool2d(x,1,pool_type='max')
        #gmp =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])        
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight=fluid.layers.reshape(gmp_weight, shape=[x.shape[0], -1])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])          
        #gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp = x * gmp_weight
        
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap

# 定义上采样模块
class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
#         out = fluid.layers.resize_nearest(input=inputs, scale=self.scale)        
#         get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

#         reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, out_shape=out_shape)
        return out


class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out
        
        
        
class Spectralnorm(fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")
    
    
    
class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace=inplace

    def forward(self, x):
        return fluid.layers.relu(x)
        #if self.inplace:
            #x.set_value(fluid.layers.relu(x))
            #return x
        #else:
            #y=fluid.layers.relu(x)
            #return y
class InstanceNorm(fluid.dygraph.Layer):
    def __init__(self):
        super(InstanceNorm, self).__init__()

    def forward(self, input):
        return fluid.layers.instance_norm(input=input)

class adaptive_pool2d(fluid.dygraph.Layer):
    def __init__(self):
        super(adaptive_pool2d, self).__init__()

    def forward(self, x,pool_size,pool_type):
        return fluid.layers.adaptive_pool2d(x, pool_size=pool_size,pool_type=pool_type)
    
class LeakyReLU(fluid.dygraph.Layer):
    def __init__(self, alpha, inplace=False):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.inplace=inplace

    def forward(self, x):
        return fluid.layers.leaky_relu(x, self.alpha)
        #if self.inplace:
            #x.set_value(fluid.layers.leaky_relu(x, self.alpha))
            #return x
        #else:
            #return fluid.layers.leaky_relu(x, self.alpha)


class Tanh(fluid.dygraph.Layer):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        return fluid.layers.tanh(x)
class MSELoss():
    def __init__(self):
        pass

    def __call__(self, prediction, label):
        return fluid.layers.mse_loss(prediction, label)

class L1Loss():
    def __init__(self):
        pass

    def __call__(self, prediction, label):
        return fluid.layers.reduce_mean(fluid.layers.elementwise_sub(prediction, label, act='abs'))
    
    
    
    
