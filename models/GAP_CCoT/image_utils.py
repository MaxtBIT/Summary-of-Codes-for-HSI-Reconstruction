import torch 
import numpy as np 
import torch.nn.functional as F

def shift(inputs, step=2):
    [nC, row, col] = inputs.shape
    output = torch.zeros(nC, row, col+(nC-1)*step)
    for i in range(nC):
        output[i,:,step*i:step*i+col] = inputs[i,:,:]
    return output
def batch_shift(inputs, step=2):
    [b,nC, row, col] = inputs.shape
    output = torch.zeros(b,nC, row, col+(nC-1)*step).to(inputs.device)
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]
    return output
def batch_shift_back(inputs,step=2):          # input [bs,256,310]  output [bs, 28, 256, 256]
    [b,c,row, col] = inputs.shape
    output = torch.zeros(b,c, row, col-(c-1)*step).to(inputs.device)
    for i in range(c):
        output[:,i,:,:] = inputs[:,i,:,step*i:step*i+col-(c-1)*step]
    return output
def gen_meas(data, mask3d):
    nC = data.shape[0]
    temp = shift(mask3d *data, 2)
    # meas = torch.sum(temp, 0)/nC*2          # meas scale
    meas = torch.sum(temp, 0)          # meas scale
    
    # y_temp = shift_back(meas,nC)
    # meas = torch.mul(y_temp, mask3d)
    return meas 

def shuffle_crop(train_data,size=660):
    h, w, _ = train_data.shape
    x_index = np.random.randint(0, h - size)
    y_index = np.random.randint(0, w - size)
    crop_data = train_data[x_index:x_index + size, y_index:y_index + size, :] 
    crop_data = np.transpose(crop_data,(2,0,1))
    return crop_data
def random_h_flip(data):
    p = np.random.randint(0,10)>5
    if p:
        data = data[:,::-1,:].copy()
    return data
def random_v_flip(data):
    p = np.random.randint(0,10)>5
    if p:
        data = data[:,:,::-1]
    return data
def random_scale(data):
    random_scale = np.random.uniform(0.5,1.5)
    data = torch.from_numpy(data.transpose((2,0,1))).unsqueeze(0)
    scale_data = F.interpolate(data,scale_factor=random_scale)
    scale_data = scale_data.squeeze(0).permute(1,2,0).numpy()
    return scale_data