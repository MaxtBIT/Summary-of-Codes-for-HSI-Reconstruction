import torch.nn as nn
import models.HyperReconNet.CodedModel as CodedModel
import models.HyperReconNet.ReconModel as ReconModel
import models.HyperReconNet.utils as utils

# set the entire structure of the network
class HyperReconNet(nn.Module):
    def __init__(self, codednet, reconnet):

        super(HyperReconNet, self).__init__()
        self.codednet = codednet
        self.reconnet = reconnet
        
    def forward(self, x):

        measure_pic = self.codednet(x) #Coded
        Output_hsi = self.reconnet(measure_pic) #Reconstruction

        return Output_hsi

def prepare_model():
    # HyperReconNet consists of two parts:
    #1. the compression
    #2. the HSI reconstruction
    opt = utils.parse_arg()
    codedmodel = CodedModel.CodedModel()
    reconmodel = ReconModel.ReconModel()   
    model = HyperReconNet(codedmodel, reconmodel)     

    return model