# HyperReconNet
PyTorch codes for reproducing the paper: **Lizhi Wang, Tao Zhang, Ying Fu, and Hua Huang, HyperReconNet: Joint Coded Aperture Optimization and Image Reconstruction for Compressive Hyperspectral Imaging, TIP, 2019.**[[Link]](https://ieeexplore.ieee.org/document/8552450)

## Abstract
Coded aperture snapshot spectral imaging (CASSI) system encodes the 3D hyperspectral image (HSI) within a single 2D compressive image and then reconstructs the underlying HSI by employing an inverse optimization algorithm, which equips with the distinct advantage of snapshot but usually results in low reconstruction accuracy. To improve the accuracy, existing methods attempt to design either alternative coded apertures or advanced reconstruction methods, but cannot connect these two aspects via a unified framework, which limits the accuracy improvement. In this paper, we propose a convolution neural network-based end-to-end method to boost the accuracy by jointly optimizing the coded aperture and the reconstruction method. On the one hand, based on the nature of CASSI forward model, we design a repeated pattern for the coded aperture, whose entities are learned by acting as the network weights. On the other hand, we conduct the reconstruction through simultaneously exploiting intrinsic properties within HSI-the extensive correlations across the spatial and spectral dimensions. By leveraging the power of deep learning, the coded aperture design and the image reconstruction are connected and optimized via a unified framework. Experimental results show that our method outperforms the state-of-the-art methods under both comprehensive quantitative metrics and perceptive quality.

## Data
In the paper, two benchmarks are utilized for training and testing. Please check them in [Link1(ICVL)](http://icvl.cs.bgu.ac.il/hyperspectral/) and [Link2(Harvard)](http://vision.seas.harvard.edu/hyperspec/). In addition, an extra-experiment following [TSA-Net](https://link.springer.com/chapter/10.1007%2F978-3-030-58592-1_12) is implemented on [CAVE Dataset](https://www1.cs.columbia.edu/CAVE/projects/gap_camera/) and [KAIST Dataset](http://vclab.kaist.ac.kr/siggraphasia2017p1/). To start your work, make HDF5 files of the same length and place them in the correct path. The file structure is as follows:<br/>
>--data/<br/>
>>--ICVL_train/<br/>
>>>--trainset_1.h5<br/>
>>>...<br/>
>>>--trainset_n.h5<br/>
>>>--train_files.txt<br/>
>>>--validset_1.h5<br/>
>>>...<br/>
>>>--validset_n.h5<br/>
>>>--valid_files.txt<br/>

>>--ICVL_test/<br/>
>>>--test1/<br/>
>>>...<br/>
>>>--testn/<br/>

A few descriptions of datasets can be checked in [README](https://github.com/MaxtBIT/HyperReconNet/blob/main/data/readme.txt). Note that, every image for testing is saved as several 2D images according to different channels.

## Environment
Python 3.6.2<br/>
CUDA 10.0<br/>
Torch 1.7.0<br/>
OpenCV 4.5.4<br/>
h5py 3.1.0<br/>
TensorboardX 2.4<br/>
spectral 0.22.4<br/>

## Usage
1. Download this repository via git or download the [ZIP file](https://github.com/MaxtBIT/HyperReconNet/archive/refs/heads/main.zip) manually.
```
git clone https://github.com/MaxtBIT/HyperReconNet.git
```
2. Download the [pre-trained models](https://drive.google.com/file/d/1zUZyTnPl57O7iVLC1fvPAlfzOvwupqPT/view?usp=sharing) if you need.
3. Make the datasets and place them in correct paths. Then, adjust the settings in **utils.py** according to your data.
4. Run the file **main.py** to train a model.
5. Run the files **test_for_paper.py** and **test_for_kaist.py** to test models.

## Results
### 1. Reproducing Results on ICVL&Harvard Datasets
The results reproduced on [ICVL Dataset](http://icvl.cs.bgu.ac.il/hyperspectral/) and [Harvard Dataset](http://vision.seas.harvard.edu/hyperspec/). In this stage, the mask is learnable. And the size of patches is 64 * 64 * 31. In addition, only the central areas with 512 * 512 * 31 are compared in testing.
<table>
   <tr align = "center">
      <td rowspan="2">Method</td>
      <!-- <td colspan="1"></td> -->
      <td colspan="3">Performance</td>
      <td colspan="2">Complexity</td>  
      <td colspan="2">Code</td>  
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>SSIM</td>
      <td>Reference</td>
      <td>Params</td>
      <td>FLOPs</td>
      <td>PyTorch</td>
      <td>TensorFlow</td>
   </tr>
   <tr align = "center">
      <td rowspan="2">Lambda-Net</td>
      <td>28.53</td>
      <td>0.84</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Link1</a><a href = "https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html">Link2</a></td>
      <td rowspan="2">58.25</td>
      <td rowspan="2">44.59</td>
      <td rowspan="2"><a href = "https://github.com/mlplab/Lambda">Link</a></td>
      <td rowspan="2"><a href = "https://github.com/xinxinmiao/lambda-net">Link</a></td>
   </tr>
   <tr align = "center">
      <td>29.25</td>
      <td>0.89</td>
      <td>Link</td>
   </tr>
   <tr align = "center">
      <td rowspan="2">DSSP</td>
      <td>30.35</td>
      <td>0.85</td>
      <td>Link</td>
      <td rowspan="2">0.30</td>
      <td rowspan="2">20.14</td>
      <td rowspan="2"><a href = "https://github.com/mlplab/Lambda">Link</a></td>
      <td rowspan="2"><a href = "https://github.com/wang-lizhi/DSSP">Link</a></td>
   </tr>
   <tr align = "center">
      <td>28.93</td>
      <td>0.83</td>
      <td>Link</td>
   </tr>
   <tr align = "center">
      <td>DNU</td>
      <td>30.74</td>
      <td>0.86</td>
      <td>Link</td>
      <td>4.47</td>
      <td>293.90</td>
      <td><a href = "https://github.com/wang-lizhi/DeepNonlocalUnrolling">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td rowspan="2">TSA-Net</td>
      <td>30.15</td>
      <td>0.89</td>
      <td>Link</td>
      <td rowspan="2">44.25</td>
      <td rowspan="2">135.12</td>
      <td rowspan="2"><a href = "https://github.com/mengziyi64/TSA-Net">Link</a></td>
      <td rowspan="2">/</td>
   </tr>
   <tr align = "center">
      <td>31.46</td>
      <td>0.89</td>
      <td>Link</td>
   </tr>
   <tr align = "center">
      <td rowspan="2">GAP-Net</td>
      <td>32.13</td>
      <td>0.92</td>
      <td>Link</td>
      <td rowspan="2">4.27</td>
      <td rowspan="2">84.08</td>
      <td rowspan="2"><a href = "https://github.com/mengziyi64/GAP-net">Link</a></td>
      <td rowspan="2">/</td>
   </tr>
   <tr align = "center">
      <td>32.47</td>
      <td>0.93</td>
      <td>Link</td>
   </tr>
   <tr align = "center">
      <td>DGSM</td>
      <td>32.63</td>
      <td>0.92</td>
      <td>Link</td>
      <td>3.76</td>
      <td>647.80</td>
      <td><a href = "https://github.com/MaxtBIT/DGSMP">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>PnP-DIP-HSI</td>
      <td>31.30</td>
      <td>0.90</td>
      <td>Link</td>
      <td>33.85</td>
      <td>≥3000</td>
      <td><a href = "https://github.com/mengziyi64/CASSI-Self-Supervised">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>HerosNet</td>
      <td>34.45</td>
      <td>0.97</td>
      <td>Link</td>
      <td>11.75</td>
      <td>447.18</td>
      <td><a href = "https://github.com/jianzhangcs/HerosNet">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>PnP-HSI</td>
      <td>25.67</td>
      <td>0.70</td>
      <td>Link</td>
      <td>1.96</td>
      <td>≥3000</td>
      <td><a href = "https://github.com/zsm1211/PnP-CASSI">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>MST</td>
      <td>35.18</td>
      <td>0.95</td>
      <td>Link</td>
      <td>2.46</td>
      <td>31.40</td>
      <td><a href = "https://github.com/caiyuanhao1998/MST">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>CAE-SRN</td>
      <td>33.26</td>
      <td>0.93</td>
      <td>Link</td>
      <td>1.25</td>
      <td>83.06</td>
      <td><a href = "https://github.com/Jiamian-Wang/HSI_baseline">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>GAP-CCoT</td>
      <td>35.26</td>
      <td>0.95</td>
      <td>Link</td>
      <td>8.04</td>
      <td>95.60</td>
      <td><a href = "https://github.com/ucaswangls/GAP-CCoT">Link</a></td>
      <td>/</td>
   </tr>
</table>

### 2. Results of Extra-Experiments on CAVE&KAIST Datasets
The results obtained on [CAVE Dataset](https://www1.cs.columbia.edu/CAVE/projects/gap_camera/) and [KAIST Dataset](http://vclab.kaist.ac.kr/siggraphasia2017p1/). 30 scenes of CAVE are used for training, and 10 scenes of KAIST are used for testing. The fixed mask is a binary constant randomly generated in model initialization. The optimized mask is a learnable binary variable that can be optimized by the network. Note that, there is only **one binary mask** utilized in training and testing. Images with a size of 256 * 256 * 28 are used for comparison.
<table>
   <tr align = "center">
      <td></td>
      <td>Fixed Mask</td>
      <td>Optimized Mask</td>
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>33.61</td>
      <td>36.13</td>
   </tr>
   <tr align = "center">
      <td>SSIM</td>
      <td>0.915</td>
      <td>0.950</td>
   </tr>
   <tr align = "center">
      <td>SAM</td>
      <td>0.103</td>
      <td>0.075</td>
   </tr>
</table>

## Citation
```
@article{HyperReconNet,
  title={HyperReconNet: Joint Coded Aperture Optimization and Image Reconstruction for Compressive Hyperspectral Imaging},
  author={Wang, Lizhi and Zhang, Tao and Fu, Ying and Huang, Hua},
  journal={IEEE Transactions on Image Processing},
  volume={28},
  number={5},
  pages={2257-2270},
  year={2019},
}
```
