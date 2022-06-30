import torch.nn.functional as F
from models.GAP_Net.my_tools import *
from models.GAP_Net.utils import A, At
import torch
from models.GAP_Net.utils import *

class GAP_net(nn.Module):

    def __init__(self):
        super(GAP_net, self).__init__()
        
        # only to calculate the parameters and FLOPs, modified by MaxtBIT
        self.unet1 = Unet(28, 28)
        self.unet2 = Unet(28, 28)
        self.unet3 = Unet(28, 28)
        self.unet4 = Unet(28, 28)
        self.unet5 = Unet(28, 28)
        self.unet6 = Unet(28,28)
        self.unet7 = Unet(28, 28)
        self.unet8 = Unet(28, 28)
        self.unet9 = Unet(28, 28)

    def forward(self, y): 
        x_list = []

        # only to calculate the parameters and FLOPs, modified by MaxtBIT
        mask_path = "./models/GAP_Net/mask_3d_shift.mat"
        Phi, Phi_s = generate_masks(mask_path, 1)

        x = At(y,Phi)
        ### 1-3
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet1(x)
        x = shift(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet2(x)
        x = shift(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet3(x)
        x = shift(x)
        ### 4-6
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet4(x)
        x = shift(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet5(x)
        x = shift(x)
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet6(x)
        x = shift(x)
        ### 7-9
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet7(x)
        x = shift(x)
        x_list.append(x[:,:,:,0:256])
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet8(x)
        x = shift(x)
        x_list.append(x[:,:,:,0:256])
        yb = A(x,Phi)
        x = x + At(torch.div(y-yb,Phi_s),Phi)
        x = shift_back(x)
        x = self.unet9(x)
        x = shift(x)
        x_list.append(x[:,:,:,0:256])

        output_list = x_list[-3:]
        return output_list