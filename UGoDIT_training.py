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


task_config= '/configs/super_resolution_config.yaml'


task_config = load_yaml(task_config)
measure_config = task_config['measurement']
operator = get_operator(device=device, **measure_config['operator'])

num_layers = 2

imsize = -1
dim_div_by = 64

fnames  = ['00013', '00016', '00019', '00018']
roots   = ['ffhq/'] * len(fnames)   # or a list of different dirs

# 2) load + preprocess each into a (C,H,W) np array → tensor
tensors = []
for root, name in zip(roots, fnames):
    path = os.path.join(root, f"{name}.png")
    img_pil, _ = get_image(path, imsize)        # loads PIL + np
    img_pil    = crop_image(img_pil, dim_div_by)
    img_np     = pil_to_np(img_pil)             # (C,H,W), floats in [0,1]
    t = torch.tensor(img_np)                    # → (C,H,W)
    tensors.append(t)

# 3) stack into a batch: shape (N, C, H, W)
batch = torch.stack(tensors, dim=0).to(device)
print("batch shape:", batch.shape)  # should be (5, 3, H, W) or (5,1,H,W) depending on your channels

blurred_batch = operator.forward(batch)        # → (5, C, H, W)
blurred_batch = torch.clamp(blurred_batch, 0, 1)

# 5) if you need back in numpy for plotting:
blurred_np = blurred_batch.cpu().detach().numpy()  # (5, C, H, W)

# ### Hyper-parameters

exp_weight = 0.99
input_depth = 3
output_depth = 3
INPUT = 'noise'
show_every = 500

## Loss
mse = torch.nn.MSELoss().type(dtype)


def init_weights(net, init_type='normal', init_gain=0.02):
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
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


pad = 'reflection'
#pad = 'nearest'
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

num_epochs = 500
# learning_rate = 4e-4
show_every = 50


optimizer = optim.Adam(net.parameters(), lr = learning_rate)


noise_list = [
    get_noise(input_depth, INPUT, img_np.shape[1:]).squeeze(0).type(dtype)
    for _ in range(4)
]

noise_batch = torch.stack(noise_list, dim=0)
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


net_input =noise_batch



losses = []
psnrs = []
avg_psnrs = []

exp_weight = .99
out_avg = torch.zeros_like(torch.abs(img_var)).to(device)
for epoch in range(2000):

    for _ in range(10):

        optimizer.zero_grad()
        net_out_before = torch.stack([
        out for out in net(net_input)
        ], dim=0).squeeze(0)
        net_output_combine = torch.stack([
        operator.forward(out).squeeze(0) for out in net(net_input)
        ], dim=0).squeeze(0)
        loss = F.mse_loss(net_output_combine, blurred_batch) +1* F.mse_loss(net_input, net_out_before )
        + 1e-6 * TV(net_output_combine)
        loss.backward()

        optimizer.step()

    net_input = net_out_before.detach()


encoder_sd = net.encoder.state_dict()

# 2. Write them to disk
torch.save(encoder_sd, "encoder_weights_sr_4_image.pth")






