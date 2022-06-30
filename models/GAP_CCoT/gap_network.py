from models.GAP_CCoT.basicblock import CotBlock
from torch import nn 
import torch
from models.GAP_CCoT.utils import A, At
from models.GAP_CCoT.image_utils import batch_shift_back,batch_shift
class GAP_CCoT(nn.Module):
    def __init__(self,ratio=28):
        super(GAP_CCoT, self).__init__()
        channels_list=[32,64,128]
        self.unet1 = CotBlock(ratio,channels_list)
        self.unet2 = CotBlock(ratio,channels_list)
        self.unet3 = CotBlock(ratio,channels_list)
        self.unet4 = CotBlock(ratio,channels_list)
        self.unet5 = CotBlock(ratio,channels_list)
        self.unet6 = CotBlock(ratio,channels_list)
        self.unet7 = CotBlock(ratio,channels_list)
        self.unet8 = CotBlock(ratio,channels_list)
        self.unet9 = CotBlock(ratio,channels_list)
     
        self.shift=2
        
    def forward(self, y):
        
        # only to calculate the parameters and FLOPs, modified by MaxtBIT
        Phi = torch.Tensor([1.0]).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        Phi = Phi.expand([1, 28, 256, 310]).cuda()
        Phi_s  = torch.Tensor([1.0]).unsqueeze(1).unsqueeze(1)
        Phi_s = Phi_s.expand([1, 256, 310]).cuda()

        self.norm = Phi_s
        x_list = []
        x = At(y,Phi)
        ### 1-3
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet1(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet2(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet3(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        ### 4-6
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet4(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet5(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet6(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        #7-9
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet7(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet8(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = batch_shift_back(x,step=self.shift)
        x = self.unet9(x)
        x_list.append(x)
        x = batch_shift(x,step=self.shift)

        output_list = x_list[-3:]
        return output_list
