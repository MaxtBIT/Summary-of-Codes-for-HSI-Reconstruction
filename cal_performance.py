import numpy as np
import cv2

# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)

def Cal_mse(im1, im2):

    return np.mean(np.square(im1 - im2), dtype=np.float64)

# calculate the peak signal-to-noise ratio (PSNR) by max pixel value
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

# calculate the peak signal-to-noise ratio (PSNR) by 1.0
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