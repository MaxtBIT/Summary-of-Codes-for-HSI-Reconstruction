import torch
import torch.nn as nn
import models.HyperReconNet.utils as utils

class CodedNet(nn.Module):

    def __init__(self):

        super(CodedNet, self).__init__()
        
        opt = utils.parse_arg()
        self.patch_size = opt.patch_size
        self.channel = opt.channel
        self.mask_size = int(opt.patch_size / 2)   #32 * 32

        #randomly init a fixed bk in range{0, 1}
        self.bk_fixed = torch.round(torch.rand(self.mask_size, self.mask_size))   #[32, 32]

        # init bk_input
        bk_1 = torch.cat((self.bk_fixed, self.bk_fixed), 0)            #[64, 32]
        bk_2 = torch.cat((bk_1, bk_1), 1)              #[64, 64]
        self.bk = bk_2.unsqueeze(2).repeat(1, 1, self.channel).cuda()   #[64, 64, 28]

    def forward(self, x):               #[B, 64, 64, 28]

        batch_size = x.size(0)      #batchsize大小
        measure_input = torch.zeros([batch_size, self.patch_size, self.patch_size, self.channel]).cuda()

        # roll x
        for ch in range(self.channel):
            measure_input[:, :, :, ch] = torch.roll(x[:, :, :, ch], shifts = -ch, dims = 1)

        # prepare bk
        bk_input = self.bk.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # imaging process
        x = torch.mul(measure_input, bk_input)                     #[B, 64, 64, 31]

        # unroll x
        for ch in range(self.channel):
            measure_input[:, :, :, ch] = torch.roll(x[:, :, :, ch], shifts = ch, dims = 1)
        
        # product the measurement
        measure_input = torch.sum(measure_input, 3)                                      #[B, 64, 64]

        return measure_input

def CodedModel():
    return CodedNet()
 
