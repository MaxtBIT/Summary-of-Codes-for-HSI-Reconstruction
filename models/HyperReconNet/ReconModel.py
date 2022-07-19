import torch
import torch.nn as nn
from torch.nn import init
import models.HyperReconNet.utils as utils

class ReconNet(nn.Module):

    def __init__(self):
 
        super(ReconNet, self).__init__()

        opt = utils.parse_arg()
        self.patch_size = opt.patch_size
        self.channel = opt.channel
        
        #set spatialCNN (need padding in column)
        self.spatialCNN_1 = nn.Sequential()
        for i in range(opt.patch_size):
            self.spatialCNN_1.add_module('spatialCNN_1_' + str(i), nn.Sequential(nn.Conv2d(1,opt.channel,(opt.patch_size, 3), padding = (0,1)), nn.ReLU(inplace=True),))

        self.spatialCNN_2 = nn.Sequential()
        for i in range(opt.patch_size):
            self.spatialCNN_2.add_module('spatialCNN_2_' + str(i), nn.Sequential(nn.Conv2d(opt.channel,opt.channel,(opt.patch_size, 3), padding = (0,1)), nn.ReLU(inplace=True),))

        self.spatialCNN_3= nn.Sequential()
        for i in range(opt.patch_size):
            self.spatialCNN_3.add_module('spatialCNN_3_' + str(i), nn.Sequential(nn.Conv2d(opt.channel,opt.channel,(opt.patch_size, 3), padding = (0,1)), nn.ReLU(inplace=True), ))
        
        #set spectralCNN (need padding)
        self.spectralCNN_1= nn.Sequential()
        for i in range(2):
            self.spectralCNN_1.add_module('SpectralCNN_1_' + str(i), nn.Sequential(nn.Conv2d( 2, 64, 9, padding = 4), nn.ReLU(inplace=True), 
                nn.Conv2d(64,32,1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 5, padding = 2), ))

        self.spectralCNN_2= nn.Sequential()
        for i in range(self.channel - 2):
            self.spectralCNN_2.add_module('SpectralCNN_2_' + str(i), nn.Sequential(nn.Conv2d( 3, 64, 9, padding = 4), nn.ReLU(inplace=True), 
                nn.Conv2d(64,32,1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, 5, padding = 2), ))

        #init weights for layers
        self._initialize_weights()
        
    #set xavier init
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):               #[B, 64, 64]

        batch_size = x.size(0)      #batchsize大小
        x = x.unsqueeze(1)          #[B, 1, 64, 64]
        x = x.cuda()

        # Spatial CNN
        scnn_1 = torch.zeros(batch_size, self.channel, self.patch_size, self.patch_size).cuda()     #[B, 31, 64, 64]
        for i in range(self.patch_size):
            scnn_1[:, :, i, :] = self.spatialCNN_1[i](x).squeeze(2)   #[B, 31, 1, 64]
    
        scnn_2 = torch.zeros(batch_size, self.channel, self.patch_size, self.patch_size).cuda()     #[B, 31, 64, 64]
        for i in range(self.patch_size):
            scnn_2[:, :, i, :] = self.spatialCNN_2[i](scnn_1).squeeze(2)   #[B, 31, 1, 64]

        scnn_3 = torch.zeros(batch_size, self.channel, self.patch_size, self.patch_size).cuda()     #[B, 31, 64, 64]
        for i in range(self.patch_size):
            scnn_3[:, :, i, :] = self.spatialCNN_3[i](scnn_2).squeeze(2)   #[B, 31, 1, 64]

        # Spectral CNN
        recon_hsi = torch.zeros(batch_size, self.channel, self.patch_size, self.patch_size).cuda()     #[B, 31, 64, 64]
        for i in range(self.channel):
            if i == 0:
                recon_hsi[:, i, :, :] = self.spectralCNN_1[0](scnn_3[:, 0 : 2, :, :]).squeeze(1)  #[B, 1, 64, 64]
            elif i == (self.channel - 1):
                recon_hsi[:, i, :, :] = self.spectralCNN_1[1](scnn_3[:, (self.channel - 2) : self.channel, :, :]).squeeze(1)  #[B, 1, 64, 64]
            else:
                recon_hsi[:, i, :, :] = self.spectralCNN_2[i-1](scnn_3[:, i-1 : i+2, :, :]).squeeze(1)  #[B, 1, 64, 64]
        
        # residual
        output = scnn_3 + recon_hsi      #[B, 31, 64, 64]

        return output.transpose(3,1).transpose(2,1)            #[B, 64, 64, 31]

def ReconModel():
    return ReconNet()