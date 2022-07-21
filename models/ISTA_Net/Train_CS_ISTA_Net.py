import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from argparse import ArgumentParser

parser = ArgumentParser(description='ISTA-Net')

parser.add_argument('--start_epoch', type=int, default=0, help='epoch number of start training')
parser.add_argument('--end_epoch', type=int, default=200, help='epoch number of end training')
parser.add_argument('--layer_num', type=int, default=9, help='phase number of ISTA-Net')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--group_num', type=int, default=1, help='group number for training')
parser.add_argument('--cs_ratio', type=int, default=25, help='from {1, 4, 10, 25, 40, 50}')
parser.add_argument('--gpu_list', type=str, default='0', help='gpu index')

parser.add_argument('--matrix_dir', type=str, default='sampling_matrix', help='sampling matrix directory')
parser.add_argument('--model_dir', type=str, default='model', help='trained or pre-trained model directory')
parser.add_argument('--data_dir', type=str, default='data', help='training data directory')
parser.add_argument('--log_dir', type=str, default='log', help='log directory')

args = parser.parse_args()

# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

        self.Conv1 = nn.Sequential()
        for i in range(9):
            self.Conv1.add_module('Conv1_' + str(i), nn.Sequential(nn.Conv2d(31,32,(3,3), padding = 1),  nn.ReLU(inplace=True),))

        self.Conv2 = nn.Sequential()
        for i in range(9):
            self.Conv2.add_module('Conv2_' + str(i), nn.Sequential(nn.Conv2d(32,32,(3,3), padding = 1),))

        self.Conv3 = nn.Sequential()
        for i in range(9):
            self.Conv3.add_module('Conv3_' + str(i), nn.Sequential(nn.Conv2d(32,32,(3,3), padding = 1),  nn.ReLU(inplace=True),))

        self.Conv4 = nn.Sequential()
        for i in range(9):
            self.Conv4.add_module('Conv4_' + str(i), nn.Sequential(nn.Conv2d(32,31,(3,3), padding = 1),))

    def forward(self, x, PhiTPhi, PhiTb, i):
        patch_size = 257
        x = x.view(patch_size, patch_size)
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, patch_size, patch_size)

        # x = F.conv2d(x_input, self.conv1_forward, padding=1)
        # x = F.relu(x)
        x_input = x_input.expand(1, 31, 257, 257).cuda()
        x = self.Conv1[i](x_input)

        # x_forward = F.conv2d(x, self.conv2_forward, padding=1)
        x_forward = self.Conv2[i](x)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        # x = F.conv2d(x, self.conv1_backward, padding=1)
        # x = F.relu(x)
        x = self.Conv3[i](x)

        # x_backward = F.conv2d(x, self.conv2_backward, padding=1)
        x_backward = self.Conv4[i](x)

        x_backward = x_backward[:, 0, :, :]
        x_pred = x_backward.view(-1, patch_size * patch_size)

        # x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        # x = F.relu(x)
        x = self.Conv3[i](x_forward)

        # x_est = F.conv2d(x, self.conv2_backward, padding=1)
        x_est = self.Conv4[i](x)

        symloss = x_est - x_input

        return [x_pred, symloss]

# Define ISTA-Net
class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix):

        # only to calculate the parameters and FLOPs, modified by MaxtBIT
        Phix = Phix.squeeze(0)
        Phi = torch.Tensor([1.0]).unsqueeze(1)
        Phi = Phi.expand([257, 257]).cuda()
        Qinit  = torch.Tensor([1.0]).unsqueeze(1)
        Qinit = Qinit.expand([257, 257]).cuda()

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []   # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb, i)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]