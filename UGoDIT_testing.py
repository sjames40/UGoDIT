from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import cv2
import os
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
import torch.optim as optim
from tqdm import tqdm
from models import *
from torchmetrics.functional import total_variation as TV
import scipy
from guided_diffusion.measurements import get_operator
import yaml
from torch.nn import init
from torchvision import transforms

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

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


task_config= './configs/super_resolution_config.yaml'


task_config = load_yaml(task_config)
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])


ffname = '00002'
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
mse = torch.nn.MSELoss().to(device)


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
            upsample_mode='bilinear').to(device)

encoder_weights = torch.load('encoder_weights_sr_4_image_multidecoder.pth', map_location=device)
net.encoder.load_state_dict(encoder_weights)
init_decoder_weights(net, init_type='normal', init_gain=0.02)

num_epochs = 500

show_every = 50


optimizer = optim.Adam(net.decoders.parameters(), lr = learning_rate) # KEEP DECODER FROZEN DURING INFERENCE



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



#net_input =get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

# get the target shape from the original high-resolution image tensor
target_shape = image_tensor.shape[2:] # This will be (H, W), e.g., (256, 256)

# upsample the low-resolution degraded image to the target shape
upsampled_blurred_tensor = F.interpolate(blurred_image_tensor, size=target_shape, mode='bilinear', align_corners=False)

# set the network input 'z' to this upsampled tensor
net_input = upsampled_blurred_tensor.detach()

# shape check 
print("--- Shape Check for Initialization ---")
print(f"Ground Truth Shape:       {image_tensor.shape}")
print(f"Degraded Image Shape (y): {blurred_image_tensor.shape}")
print(f"Initialized Input Shape (z): {net_input.shape}")
print("------------------------------------")



losses = []
psnrs = []
avg_psnrs = []
out = []
exp_weight = .99
out_avg = torch.zeros(img_np.shape).to(device)  # [3, H, W]

print(f"Input image shape: {img_np.shape}")
print(f"Target PSNR improvement from low-res to high-res reconstruction")

for epoch in range(2000):
    
    for _ in range(10):
        optimizer.zero_grad()
        
        net_output = net(net_input)
        
        pred_proj = operator.forward(net_output[0])
        
        loss = (F.mse_loss(pred_proj, blurred_image_tensor) + 
                2 * F.mse_loss(net_input, net_output[0]))
        
        loss.backward()
        optimizer.step()
    
    net_input = net_output[0].detach()
    
    with torch.no_grad():
        out_np = net_output[0].cpu().numpy()[0]
        out_np = np.clip(out_np, 0, 1)
        
        psnr = compare_psnr(img_np, out_np)
        psnrs.append(psnr)
        
        # exponential moving average
        out_tensor = torch.tensor(out_np).to(device)
        out_avg = out_avg * exp_weight + out_tensor * (1 - exp_weight)
        
        # avg PSNR
        out_avg_np = out_avg.cpu().numpy()
        avg_psnr = compare_psnr(img_np, out_avg_np)
        avg_psnrs.append(avg_psnr)
        
        losses.append(loss.item())
    
    if epoch % show_every == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | PSNR: {psnr:.2f} dB | Avg PSNR: {avg_psnr:.2f} dB")
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.imshow(img_np.transpose(1, 2, 0))
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(img_mask_np.transpose(1, 2, 0))
        plt.title('Low Resolution Input')
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(out_np.transpose(1, 2, 0))
        plt.title(f'Current Output\nPSNR: {psnr:.2f} dB')
        plt.axis('off')
        
        plt.subplot(144)
        plt.imshow(out_avg_np.transpose(1, 2, 0))
        plt.title(f'Sliding Average\nPSNR: {avg_psnr:.2f} dB')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'progress_epoch_{epoch}.png')
        plt.show()

# save psnrs as np array
psnrs_npy = np.array(psnrs)
np.save('psnrs.npy', psnrs)
print(f"Saved PSNRS array of this run with shape {psnrs_npy.shape}")
