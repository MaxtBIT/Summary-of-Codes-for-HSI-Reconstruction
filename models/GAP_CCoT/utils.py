import torch
import scipy.io as scio
import numpy as np
from torch import nn 
import logging 
import time 
import os 
import os.path as osp
import cv2
import math 
from models.GAP_CCoT.ssim import ssim
def ssim(img1, img2):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compare_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            # for i in range(3):
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
# def compare_ssim(img1,img2):
#     img1 = torch.from_numpy(img1)
#     img2 = torch.from_numpy(img2)
#     img1 = torch.unsqueeze(img1,0)
#     img2 = torch.unsqueeze(img2,0)
#     return ssim(torch.unsqueeze(img1,0), torch.unsqueeze(img2,0))



def compare_psnr(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1. / rmse)

def random_masks(frames=8,size_h=256,size_w=256,mask_path=None):
    if mask_path is None:
        mask = np.random.randint(0,high=2,size=(frames,size_h,size_w)).astype(np.float32)
        np.save("mask.npy",mask)
    else:
        mask = np.load(mask_path)
    mask_s = np.sum(mask,axis=0)
    mask_s[mask_s==0] = 1
    return torch.from_numpy(mask,),torch.from_numpy(mask_s)

def save_image(out,gt,image_name,show_flag=False):
    sing_out = out.transpose(1,0,2).reshape(out.shape[1],-1)
    if gt is not None:
        sing_gt = gt.transpose(1,0,2).reshape(gt.shape[1],-1)
        result_img = np.concatenate([sing_out,sing_gt],axis=0)*255
    else:
        result_img = sing_out*255
    cv2.imwrite(image_name,result_img)
    if show_flag:
        cv2.namedWindow("image",0)
        cv2.imshow("image",result_img.astype(np.uint8))
        cv2.waitKey(0)
        
class double_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.d_conv(x)
        return x

def generate_masks(mask_path,size_h=256,size_w=256):
    mask = scio.loadmat(osp.join(mask_path,'mask.mat'))
    mask = mask['mask']
    mask_h,mask_w = mask.shape
    rand_h_begin = np.random.randint(0,mask_h-size_h+1)
    rand_w_begin = np.random.randint(0,mask_w-size_w+1)
    mask = mask[rand_h_begin:rand_h_begin+size_h,rand_w_begin:rand_w_begin+size_w]
    mask = np.tile(mask[:,:,np.newaxis],(1,1,28))
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0,dtype=np.float32)
    mask_s[mask_s==0] = 1
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask_s = torch.from_numpy(mask_s)
    mask_s = mask_s.float()
    return mask, mask_s

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def Logger(log_dir):
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s")

    localtime = time.strftime("%Y_%m_%d_%H_%M_%S")
    logfile = osp.join(log_dir,localtime+".log")
    fh = logging.FileHandler(logfile,mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 
