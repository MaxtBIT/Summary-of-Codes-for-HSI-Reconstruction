import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import models.DTLP.utils as utils

# smallest positive float number
FLT_MIN = float(np.finfo(np.float32).eps)

class ReconNet(nn.Module):

    def __init__(self):
 
        super(ReconNet, self).__init__()

        opt = utils.parse_arg()
        self.filter_size = 3
        self.filter_num  = 64
        self.rank = opt.rank
        self.channel = opt.channel
        self.phase = opt.phase

        # set learnable parameters for every stage
        self.deta = nn.Parameter(torch.Tensor([0.04 for _ in range(self.phase)]), requires_grad = True)
        self.eta = nn.Parameter(torch.Tensor([0.8 for _ in range(self.phase)]), requires_grad = True)

        self.Conv1 = nn.Sequential()
        for i in range(self.phase):
            self.Conv1.add_module('Conv1_' + str(i), nn.Sequential(nn.Conv2d(self.channel,64,(3,3), padding = 1),  ))
        
        self.Encoding = nn.Sequential()
        for i in range(self.phase):
            self.Encoding.add_module('Encoding_' + str(i), nn.Sequential(nn.Conv2d(64,64,(3,3), padding = 1), nn.ReLU(inplace=True), nn.Conv2d(64,64,(3,3), padding = 1),))

        self.RTGB_up_H = nn.Sequential()
        self.RTGB_up_W = nn.Sequential()
        self.RTGB_up_C = nn.Sequential()
        self.RTGB_down_H = nn.Sequential()
        self.RTGB_down_W = nn.Sequential()
        self.RTGB_down_C = nn.Sequential()
        for i in range(self.rank * self.phase):
            self.RTGB_up_H.add_module('RTGB_up_H_' + str(i), nn.Sequential(nn.Conv2d(1, 1, 1, padding = 0), nn.Sigmoid(),))
            self.RTGB_up_W.add_module('RTGB_up_W_' + str(i), nn.Sequential(nn.Conv2d(1, 1, 1, padding = 0), nn.Sigmoid(),))
            self.RTGB_up_C.add_module('RTGB_up_C_' + str(i), nn.Sequential(nn.Conv2d(64, 64, 1, padding = 0), nn.Sigmoid(),))
            self.RTGB_down_H.add_module('RTGB_down_H_' + str(i), nn.Sequential(nn.Conv2d(1, 1, 1, padding = 0), nn.Sigmoid(),))
            self.RTGB_down_W.add_module('RTGB_down_W_' + str(i), nn.Sequential(nn.Conv2d(1, 1, 1, padding = 0), nn.Sigmoid(),))
            self.RTGB_down_C.add_module('RTGB_down_C_' + str(i), nn.Sequential(nn.Conv2d(64, 64, 1, padding = 0), nn.Sigmoid(),))

        self.Fusion = nn.Sequential()
        for i in range(self.phase):
            self.Fusion.add_module('Fusion_' + str(i), nn.Sequential(nn.Conv2d(self.filter_num * self.rank, self.filter_num, (3,3), padding = 1),))

        self.Conv2 = nn.Sequential()
        for i in range(self.phase):
            self.Conv2.add_module('Conv2_' + str(i), nn.Sequential(nn.Conv2d(64,self.channel,(3,3), padding = 1),  nn.ReLU(inplace=True),))

        #set init weights for layers
        self._initialize_weights()
        
    #set xavier init
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=np.sqrt(0.5))
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def encoding(self, input, stage):
        output = self.Encoding[stage](input)
        return output

    # RTGB modules
    def rtgb(self, input, i, stage, flag):
        gap_Height = torch.mean(torch.mean(input, 3, keepdim = True), 1, keepdim = True)
        gap_Weight = torch.mean(torch.mean(input, 2, keepdim = True), 1, keepdim = True)
        gap_Channel = torch.mean(torch.mean(input, 2, keepdim = True), 3, keepdim = True)

        if flag == 'up':
            convHeight_GAP = self.RTGB_up_H[stage * self.rank + i](gap_Height)
            convWeight_GAP = self.RTGB_up_W[stage * self.rank + i](gap_Weight)
            convChannel_GAP = self.RTGB_up_C[stage * self.rank + i](gap_Channel)
        else:
            convHeight_GAP = self.RTGB_down_H[stage * self.rank + i](gap_Height)
            convWeight_GAP = self.RTGB_down_W[stage * self.rank + i](gap_Weight)
            convChannel_GAP = self.RTGB_down_C[stage * self.rank + i](gap_Channel)

        vecConHeight_GAP = torch.reshape(convHeight_GAP, [convHeight_GAP.shape[0], convHeight_GAP.shape[2],1])
        vecConWeight_GAP = torch.reshape(convWeight_GAP, [convWeight_GAP.shape[0], 1, convWeight_GAP.shape[3]])
        vecConChannel_GAP = torch.reshape(convChannel_GAP, [convChannel_GAP.shape[0], 1, convChannel_GAP.shape[1]])

        matHWmulT = torch.matmul(vecConHeight_GAP, vecConWeight_GAP)
        vecHWmulT = torch.reshape(matHWmulT, [matHWmulT.shape[0], matHWmulT.shape[1] * matHWmulT.shape[2], 1])
        matHWCmulT = torch.matmul(vecHWmulT, vecConChannel_GAP)
        recon = torch.reshape(matHWCmulT, [input.shape[0], input.shape[3], input.shape[2], input.shape[1]])
        recon = recon.transpose(3,1).transpose(3,2)
        return recon
    
    def resBlock(self, input, i, stage):
        xup = self.rtgb(input, i, stage, flag = 'up')
        x_res = input - xup
        xdn = self.rtgb(x_res, i, stage, flag = 'down')
        xdn = xdn + x_res
        return xup, xdn

    def drtlm(self, input, stage):
        (xup, xdn) = self.resBlock(input, 0, stage)
        temp_xup = xdn
        output = xup
        for i in range(1,self.rank):
            (temp_xup,temp_xdn) = self.resBlock(temp_xup, i, stage)
            xup = xup + temp_xup       
            output = torch.cat((output, xup),1)
            temp_xup = temp_xdn
        return output

    def fusion(self, input, xt, stage):
        attention_map = self.Fusion[stage](input)
        output = torch.mul(xt, attention_map)
        return output

    def recon(self, xt, x0, Cu, stage):

        xt = xt.transpose(3,1).transpose(3,2)
        # Low-rank Tensor Recovery
        x_feature_0 = self.Conv1[stage](xt)
        x_feature_1 = self.encoding(x_feature_0, stage)
        attention_map_cat = self.drtlm(x_feature_1, stage)

        x_feature_lowrank = self.fusion(attention_map_cat, x_feature_1, stage)
        x_mix = x_feature_lowrank + x_feature_0
        z = self.Conv2[stage](x_mix)

        z = z.transpose(3,1).transpose(2,1)
        xt = xt.transpose(3,1).transpose(2,1)

        # Linear Projection (deep unfolding)
        yt = torch.mul(xt, Cu)
        yt = yt.sum(3)
        yt1 = yt.unsqueeze(3)
        yt2 = yt1.repeat(1, 1, 1, self.channel)
        xt2 = torch.mul(yt2, Cu)  # PhiT*Phi*xt
        x = torch.mul(1 - self.deta[stage] * self.eta[stage], xt) - torch.mul(self.deta[stage], xt2) + torch.mul(self.deta[stage], x0) + torch.mul(self.deta[stage] * self.eta[stage], z)

        return x

    def forward(self, x):

        # only to calculate the parameters and FLOPs, modified by MaxtBIT
        Cu = torch.Tensor([1.0]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        Cu = Cu.expand([1, 256, 256, 28]).cuda()

        xt = x
        for stage in range(self.phase):
            xt = self.recon(xt, x, Cu, stage)
        return xt

def ReconModel():
    return ReconNet()
 