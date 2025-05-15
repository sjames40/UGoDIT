from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from models import *
import torch
import torch.optim
import torch.nn.functional as F
import time
from utils.denoising_utils import *
import _pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
import torch.optim as optim
from tqdm.notebook import tqdm
from models import *
from torchmetrics.functional import total_variation as TV
import scipy
from guided_diffusion.measurements import get_operator
import yaml
from torch.nn import init

def np_plot(np_matrix, title, opt = 'RGB'):
    plt.figure(2)
    plt.clf()
    if opt == 'RGB':
        fig = plt.imshow(np_matrix.transpose(1, 2, 0), interpolation = 'nearest')
    elif opt == 'map':
        fig = plt.imshow(np_matrix, interpolation = 'bilinear', cmap = cm.RdYlGn)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title(title)
    plt.axis('off')
    plt.pause(0.05) 



def compare_psnr(img1, img2):
    MSE = np.mean(np.abs(img1-img2)**2)
    psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    #psnr = 10*math.log10(float(1.**2)/MSE)
    return psnr


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



from torchvision import transforms
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


task_config= '/configs/super_resolution_config.yaml'


task_config = load_yaml(task_config)
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])


ffname = '00001'
data_root = 'ffhq/'
img_path = data_root + ffname + '.png'
num_layers = 2

imsize = -1
dim_div_by = 64

img_pil, img_np = get_image(img_path, imsize)

img_pil      = crop_image(img_pil,      dim_div_by)

img_np      = pil_to_np(img_pil)

image_tensor = torch.tensor(img_np).unsqueeze(0).to(device)  # Add batch dimension


blurred_image_tensor = operator.forward(image_tensor)#blur_kernel(image_tensor)
blurred_image_tensor = torch.clip(blurred_image_tensor, 0, 1)

img_mask_np = operator.forward(image_tensor)
img_mask_np = img_mask_np[0].cpu().detach().numpy()#.permute(1,2,0)

nsize_H, nsize_W = img_np.shape[1], img_np.shape[2]


exp_weight = 0.99
input_depth = 3
output_depth = 3
INPUT = 'noise'
show_every = 500

## Loss
mse = torch.nn.MSELoss().type(dtype)


def init_decoder_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('Initialize decoder with %s' % init_type)
    net.decoders.apply(init_func)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


pad = 'reflection'

OPT_OVER = 'net'
learning_rate = LR = 0.01
exp_weight = 0.99
input_depth = 3
output_depth = 3
INPUT = 'noise'


net = get_net(input_depth, 'skip_shared', pad,
            skip_n33d=128, 
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode='bilinear').type(dtype)

encoder_weights = torch.load('encoder_weights_sr.pth', map_location=device)
net.encoder.load_state_dict(encoder_weights)
init_decoder_weights(net, init_type='normal', init_gain=0.02)

num_epochs = 500

show_every = 50


# In[13]:


optimizer = optim.Adam(net.parameters(), lr = learning_rate)



img_var = torch.tensor(img_np).unsqueeze(0).to(device)



INPUT = 'noise'
pad = 'reflection'
OPT_OVER = 'net' # optimize over the net parameters only
reg_noise_std = 1./3.
learning_rate = LR = 0.01
exp_weight=0.99
input_depth = 3 
roll_back = True # to prevent numerical issues
num_iter = 5000 # max iterations
burnin_iter = 7000 # burn-in iteration for SGLD
weight_decay = 5e-8
show_every =  10



net_input =get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)


# In[19]:


losses = []
psnrs = []
avg_psnrs = []
out = []
exp_weight = .99
out_avg = torch.zeros_like(torch.abs(img_var)).to(device)

for epoch in range(2000):

    for _ in range(10):

        optimizer.zero_grad()

        net_output = net(net_input)

        pred_proj = operator.forward(net_output[0])

        loss = F.mse_loss(pred_proj, blurred_image_tensor) +0.1* F.mse_loss(net_input, net_output[0])
        + 1e-6 * TV(net_output[0])
        loss.backward()

        optimizer.step()

    net_input = net_output[0].detach()






