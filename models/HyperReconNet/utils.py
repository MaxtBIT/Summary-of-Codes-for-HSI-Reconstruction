import argparse
import time
import numpy as np
from torch import save
from os import path as path
import cv2

# function for parsing input arguments
def parse_arg():
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-mode', type=str, default="optim", help="baseline: random mask, optim: learnable mask")
    # training settings
    parser.add_argument('-gpuid', type=int, default=0)
    parser.add_argument('-batch_size', type=int, default=128)  
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-batch_num', type=int, default=118)  # the number of batches for each .h5 file
    parser.add_argument('-report_every', type=int, default=59)   # report log after x batches
    parser.add_argument('-train_len', type=int, default=2) # get the number of .h5 files
    parser.add_argument('-valid_len', type=int, default=2)
    # paths
    parser.add_argument('-save_dir', type=str, default='./model/')
    parser.add_argument('-save_log', type=str, default='./data_log/')
    parser.add_argument('-save_loss', type=str, default='./graph_log/')
    parser.add_argument('-pretrained_path', type=str, default='./pretrained_model/model_Harvard.pth', help="icvl:model_ICVL.pth, havard: model_Harvard.pth, cave: model_CAVE_baseline/optim.pth") # set the path of the pre-trained model
    parser.add_argument('-trainset_path', type=str, default='./data/Harvard_train/', help="icvl: ./data/ICVL_train/, havard: ./data/Harvard_train/, cave: ./data/CAVE_train/") # set the path of the trainset
    parser.add_argument('-testset_path', type=str, default='./data/Harvard_test/', help="icvl: ./data/ICVL_test/, havard: ./data/Harvard_test/, kaist: ./data/KAIST_test/") # set the path of the testset
    # model settings
    parser.add_argument('-save_name', type=str, default='trained_model')   
    parser.add_argument('-pretrained', type=bool, default=False) # used for loading the pre-trained model
    parser.add_argument('-channel', type=int, default=31, help="ICVL/Havard: 31, CAVE: 28") 
    parser.add_argument('-patch_size', type=int, default=64) # get the size of patches
    # optimizer settings
    parser.add_argument('-optim_type', type=str, default='adam', help="adam or sgd")
    parser.add_argument('-lr', type=float, default=0.0001, help="adam:0.0001, sgd: 0.05")
    parser.add_argument('-weight_decay', type=float, default=0)
    parser.add_argument('-momentum', type=float, default=0.9, help="sgd: 0.9")
    # reduce the learning rate after each milestone
    parser.add_argument('-milestones', type=list, default=[20,40,60,80])
    # how much to reduce the learning rate
    parser.add_argument('-gamma', type=float, default=0.1)

    opt = parser.parse_args()
    return opt

# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)

# the function for model saving
def get_save_dir(opt, str_type=None):

    root = opt.save_dir
    save_name = path.join(root, opt.save_name) 
    save_name += '_'
    save_name += time.asctime(time.localtime(time.time()))
    save_name += '.pth'

    return save_name

def save_model(model, opt):

    save_name = get_save_dir(opt)
    save(model, save_name)

    return

def Cal_mse(im1, im2):

    return np.mean(np.square(im1 - im2), dtype=np.float64)

# calculate the peak signal-to-noise ratio (PSNR)
def Cal_PSNR_by_gt(im_true, im_test):

    channel  = im_true.shape[2]
    im_true  = 255*im_true
    im_test  = 255*im_test
    
    psnr_sum = 0
    for i in range(channel):
        band_true = np.squeeze(im_true[:,:,i])
        band_test = np.squeeze(im_test[:,:,i])
        err       = Cal_mse(band_true, band_test)
        max_value = np.max(np.max(band_true))
        psnr_sum  = psnr_sum+10 * np.log10((max_value ** 2) / err)
    
    return psnr_sum/channel

def Cal_PSNR_by_default(im_true, im_test):

    channel  = im_true.shape[2]   
    psnr_sum = 0.
    for i in range(channel):
        band_true = np.squeeze(im_true[:,:,i])
        band_test = np.squeeze(im_test[:,:,i])
        err       = Cal_mse(band_true, band_test)
        psnr_sum  = psnr_sum+10 * np.log10(1.0 / err)
    
    return psnr_sum/channel

# calculate the structural similarity (SSIM)
# cited by https://github.com/cszn
def Cal_SSIM(im_true, im_test):

    if not im_true.shape == im_test.shape:
        raise ValueError('Input images must have the same dimensions.')

    channel  = im_true.shape[2]
    im_true  = 255*im_true
    im_test  = 255*im_test
    ssim_sum = 0.

    for k in range(channel):
        ssim_sum = ssim_sum + ssim(im_true[:, :, k], im_test[:, :, k])

    return ssim_sum / channel

def ssim(img1, img2):

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# calculate the spectral angle mapping (SAM)
def Cal_SAM(im_true, im_test):

    a = sum(im_true * im_test, 2) + FLT_MIN
    b = pow(sum(im_true * im_true, 2) + FLT_MIN, 1/2)
    c = pow(sum(im_test * im_test, 2) + FLT_MIN, 1/2)
    d = np.arccos(a/(b * c))

    return np.mean(d)