from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import cv2
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
from tqdm import tqdm
from torchmetrics.functional import total_variation as TV
import scipy
from guided_diffusion.measurements import get_operator
import yaml

## display images
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
    return psnr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torchvision import transforms
def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

task_config= './configs/super_resolution_config.yaml'
task_config = load_yaml(task_config)
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])

num_layers = 2
imsize = -1
dim_div_by = 64

# M=4 training images
fnames  = ['00013', '00016', '00019', '00018']
roots   = ['ffhq/'] * len(fnames)
M = len(fnames)  

# load and preprocess each into a (C,H,W) np array → tensor
tensors = []
for root, name in zip(roots, fnames):
    path = os.path.join(root, f"{name}.png")
    img_pil, _ = get_image(path, imsize)
    img_pil    = crop_image(img_pil, dim_div_by)
    img_np     = pil_to_np(img_pil)
    t = torch.tensor(img_np)
    tensors.append(t)

# Stack into a batch: shape (M, C, H, W)
batch = torch.stack(tensors, dim=0).to(device)
print("batch shape:", batch.shape)

# Generate degraded images
blurred_batch = operator.forward(batch)
blurred_batch = torch.clamp(blurred_batch, 0, 1)

exp_weight = 0.99
input_depth = 3
output_depth = 3
INPUT = 'noise'
show_every = 500
pad = 'reflection'
learning_rate = LR = 0.01
lambda_reg = 2.0 

# initialize network with M decoders
net = get_net(input_depth, 'skip_shared', pad,
            skip_n33d=128, 
            skip_n33u=128,
            skip_n11=4,
            num_scales=5,
            upsample_mode='bilinear',
            num_decoders=M
            ).type(dtype)  #num_decoders=M

print("M =", M)
print("type(net.decoders) =", type(net.decoders))
print("len(net.decoders) =", len(net.decoders))

optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# g the target shape from the original clean batch
target_shape = batch.shape[2:] 

# initialize network inputs (z_i) with upsampled degraded images
upsampled_blurred_batch = F.interpolate(blurred_batch, size=target_shape, mode='bilinear', align_corners=False)
net_inputs = [upsampled_blurred_batch[i:i+1] for i in range(M)]  # list of individual inputs

print("Training UGoDIT...")
print(f"Number of training images (M): {M}")
print(f"Input shape for each image: {net_inputs[0].shape}")
print(f"Target shape: {target_shape}")

K = 2000 
N = 10    # num of gradient updates per input update

losses = []
iteration = 0

for k in range(K):  # input updates
    if (iteration % 100 == 0):
        print(f"Iteration {iteration}, Input update {k}/{K}")
    
    # gradient updates
    for n in range(N):
        optimizer.zero_grad()
        
        # forward pass through shared encoder
        encoded, skips = net.encoder(torch.cat(net_inputs, dim=0))
        
        # process each image through its corresponding decoder
        total_loss = 0
        net_outputs = []
        
        for i in range(M):
            # get features for image i
            encoded_i = encoded[i:i+1]
            skips_i = [skip[i:i+1] if skip is not None else None for skip in skips]
            
            # pass through decoder i
            output_i = net.decoders[i](encoded_i, skips_i)
            net_outputs.append(output_i)
            
            output_downsampled = operator.forward(output_i)
            loss_measurement = F.mse_loss(output_downsampled, blurred_batch[i:i+1])
            
            loss_autoencoding = F.mse_loss(output_i, net_inputs[i])
            
            loss_i = loss_measurement + lambda_reg * loss_autoencoding
            total_loss += loss_i
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        if n == 0 and k % 100 == 0:
            losses.append(total_loss.item())
            print(f"  Loss: {total_loss.item():.6f}")
    
    with torch.no_grad():
        for i in range(M):
            # pass current input through encoder and decoder i
            encoded, skips = net.encoder(net_inputs[i])
            net_inputs[i] = net.decoders[i](encoded, skips).detach()
    
    iteration += 1

# save only the shared encoder weights
encoder_sd = net.encoder.state_dict()
torch.save(encoder_sd, "encoder_weights_sr_4_image_multidecoder.pth")
print("Training complete. Encoder weights saved.")
