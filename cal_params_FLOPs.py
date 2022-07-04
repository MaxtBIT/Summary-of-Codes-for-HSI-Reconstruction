from models.Lambda_Net.Lambda import LambdaNet
from models.DSSP.DeepSSPrior import DeepSSPrior
from models.DNU.model import DNU
from models.TSA_Net.models import TSA_Net
from models.GAP_Net.models import GAP_net
from models.PnP_HSI.hsi import HSI_SDeCNN
from models.PnP_DIP_HSI.models import UNet_noskip
from models.DGSMP.Model import HSI_CS
from models.DTLP.ReconModel import ReconNet
from models.HerosNet.HerosNet import HerosNet
from models.CAE_SRN.models_resblock_v2 import SSI_RES_UNET
from models.HDNet.HDNet import HDNet
from models.MST.MST import MST
from models.MST_Plus_Plus.MST_Plus_Plus import MST_Plus_Plus
from models.GAP_CCoT.gap_network import GAP_CCoT
from models.BIRNAT.BIRNAT import BIRNAT
from ptflops import get_model_complexity_info

#options: Lambda_Net DSSP DNU TSA_Net GAP_Net PnP_HSI PnP_DIP_HSI DGSMP DTLP HerosNet  CAE_SRN HDNet MST MST_Plus_Plus GAP_CCoT BIRNAT
method = 'Lambda_Net' # select the method

# citied tensorflow version by https://github.com/xinxinmiao/lambda-net
# citied pytorch version by https://github.com/mlplab/Lambda
if method == 'Lambda_Net':
    model = LambdaNet(28, 28).cuda()
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True)

# citied tensorflow version by https://github.com/wang-lizhi/DSSP
# citied pytorch version by https://github.com/mlplab/Lambda
elif method == 'DSSP': 
    model = DeepSSPrior(28, 28).cuda()
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True)

# citied pytorch version by https://github.com/wang-lizhi/DeepNonlocalUnrolling
elif method == 'DNU': 
    model = DNU(28, K=10)
    #If out of memory, the input size can be reduced to 128*182. Then, expand the FLOPs by 4 times.
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True)

# citied pytorch version by https://github.com/mengziyi64/TSA-Net
elif method == 'TSA_Net': 
    model = TSA_Net(28, 28).cuda()
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True)

# citied pytorch version by https://github.com/mengziyi64/GAP-net
elif method == 'GAP_Net': 
    model = GAP_net().cuda()
    flops, params = get_model_complexity_info(model, (256, 310), True, True)

# citied pytorch version by https://github.com/zsm1211/PnP-CASSI
elif method == 'PnP_HSI': 
    model = HSI_SDeCNN().cuda() #only consider the FLOPs of the denoiser
    flops, params = get_model_complexity_info(model, (7, 28, 256), True, True)

# citied pytorch version by https://github.com/mengziyi64/CASSI-Self-Supervised
elif method == 'PnP_DIP_HSI': 
    model = UNet_noskip(28, 28, bilinear=False).cuda() #only consider the FLOPs of the denoiser
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True)

# citied pytorch version by https://github.com/MaxtBIT/DGSMP
elif method == 'DGSMP':
    model = HSI_CS().cuda()
    #If out of memory, the input size can be reduced to 128*182. Then, expand the FLOPs by 4 times.
    flops, params = get_model_complexity_info(model, (256, 310), True, True)

# citied pytorch version by https://github.com/wang-lizhi/DTLP_Pytorch
elif method == 'DTLP': 
    model = ReconNet().cuda()
    flops, params = get_model_complexity_info(model, (256, 256, 28), True, True)

# citied pytorch version by https://github.com/jianzhangcs/HerosNet
elif method == 'HerosNet': 
    model = HerosNet(Ch=28, stages=8, size=256).cuda()
    #If out of memory, the input size can be reduced to 128*128. Then, expand the FLOPs by 4 times.
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True)

# citied pytorch version by https://github.com/Jiamian-Wang/HSI_baseline
elif method == 'CAE_SRN': 
    model = SSI_RES_UNET().cuda()
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True)

# citied pytorch version by https://github.com/caiyuanhao1998/MST/blob/main/simulation/train_code/architecture/HDNet.py
elif method == 'HDNet': 
    model = HDNet().cuda()
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True) 

# citied pytorch version by https://github.com/caiyuanhao1998/MST
elif method == 'MST': 
    model = MST().cuda()
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True) 

# citied pytorch version by https://github.com/caiyuanhao1998/MST-plus-plus
elif method == 'MST_Plus_Plus': 
    model = MST_Plus_Plus().cuda()
    flops, params = get_model_complexity_info(model, (28, 256, 256), True, True) 

# citied pytorch version by https://github.com/ucaswangls/GAP-CCoT
elif method == 'GAP_CCoT':
    model = GAP_CCoT().cuda()
    flops, params = get_model_complexity_info(model, (256, 310), True, True)

# citied pytorch version by https://github.com/caiyuanhao1998/MST/blob/main/simulation/train_code/architecture/BIRNAT.py
elif method == 'BIRNAT':
    model = BIRNAT().cuda()
    #If out of memory, the input size can be reduced to 128*182. Then, expand the FLOPs by 4 times.
    flops, params = get_model_complexity_info(model, (256, 310), True, True)
    
else:
    raise NotImplementedError

print(flops)
print(params)