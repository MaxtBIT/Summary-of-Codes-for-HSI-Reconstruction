# Summary of Codes for HSI Reconstruction
Codes for the paper: **A survey of reconstruction algorithms for coded aperture snapshot spectral imaging, under review**.

## Environment
Python 3.6.2<br/>
CUDA 10.0<br/>
Torch 1.7.0<br/>
SciPy 1.5.4<br/>
CuPy 9.6.0<br/>
OpenCV 4.5.4<br/>
NumPy 1.19.5<br/>

## Usage
1. Download this repository via git or download the [ZIP file](https://github.com/MaxtBIT/Summary-of-Codes-for-HSI-Reconstruction/archive/refs/heads/master.zip) manually.
```
git clone https://github.com/MaxtBIT/Summary-of-Codes-for-HSI-Reconstruction.git
```
2. Create the environment and ensure the version.
3. Select a method in **cal_params_FLOPs.py**. Then, run this file to get params_count and FLOPs.

## Summary
To analyze the complexity, we have summarized the source codes of mainstream algorithms.  The code links are shown in the table, sorted as PyTorch version and TensorFlow version. In this stage, the size of reconstructed image is set as 256 * 256 * 28, following the work of Meng et al. The parameter counts and floating-point operation counts(FLOPs) are utilized to evaluate the complexity. In addition, according to different reference, the average PSNR and SSIM on 10 scenes of KAIST Dataset are listed.

<table align = "center">
   <tr align = "center">
      <td rowspan="2">Method</td>
      <!-- <td colspan="1"></td> -->
      <td colspan="3">Performance</td>
      <td colspan="2">Complexity</td>  
      <td colspan="2">Code Link</td>  
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>SSIM</td>
      <td>Reference</td>
      <td>Params</td>
      <td>FLOPs</td>
   </tr>
   <tr align = "center">
      <td rowspan="2">Lambda-Net</td>
      <td>28.53</td>
      <td>0.84</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper1  </a><a href = "https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html">  Paper2</a></td>
      <td rowspan="2">58.25</td>
      <td rowspan="2">44.59</td>
      <td rowspan="2"><a href = "https://github.com/mlplab/Lambda">Link</a></td>
      <td rowspan="2"><a href = "https://github.com/xinxinmiao/lambda-net">Link</a></td>
   </tr>
   <tr align = "center">
      <td>29.25</td>
      <td>0.89</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper1  </a><a href = "https://arxiv.org/abs/2112.06238">  Paper2</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="2">DSSP</td>
      <td>30.35</td>
      <td>0.85</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper1  </a><a href = "https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html">  Paper2</a></td>
      <td rowspan="2">0.30</td>
      <td rowspan="2">20.14</td>
      <td rowspan="2"><a href = "https://github.com/mlplab/Lambda">Link</a></td>
      <td rowspan="2"><a href = "https://github.com/wang-lizhi/DSSP">Link</a></td>
   </tr>
   <tr align = "center">
      <td>28.93</td>
      <td>0.83</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper1  </a><a href = "https://arxiv.org/abs/2112.06238">  Paper2</a></td>
   </tr>
   <tr align = "center">
      <td>DNU</td>
      <td>30.74</td>
      <td>0.86</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper1  </a><a href = "https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html">  Paper2</a></td>
      <td>4.47</td>
      <td>293.90</td>
      <td><a href = "https://github.com/wang-lizhi/DeepNonlocalUnrolling">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td rowspan="2">TSA-Net</td>
      <td>30.15</td>
      <td>0.89</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper</a></td>
      <td rowspan="2">44.25</td>
      <td rowspan="2">135.12</td>
      <td rowspan="2"><a href = "https://github.com/mengziyi64/TSA-Net">Link</a></td>
      <td rowspan="2">/</td>
   </tr>
   <tr align = "center">
      <td>31.46</td>
      <td>0.89</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper1  </a><a href = "https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html">  Paper2</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="2">GAP-Net</td>
      <td>32.13</td>
      <td>0.92</td>
      <td><a href = "https://arxiv.org/abs/2012.08364">Paper</a></td>
      <td rowspan="2">4.27</td>
      <td rowspan="2">84.08</td>
      <td rowspan="2"><a href = "https://github.com/mengziyi64/GAP-net">Link</a></td>
      <td rowspan="2">/</td>
   </tr>
   <tr align = "center">
      <td>32.47</td>
      <td>0.93</td>
      <td><a href = "https://arxiv.org/abs/2108.07739">Paper</a></td>
   </tr>
   <tr align = "center">
      <td>DGSM</td>
      <td>32.63</td>
      <td>0.92</td>
      <td><a href = "https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Deep_Gaussian_Scale_Mixture_Prior_for_Spectral_Compressive_Imaging_CVPR_2021_paper.html">Paper</a></td>
      <td>3.76</td>
      <td>647.80</td>
      <td><a href = "https://github.com/MaxtBIT/DGSMP">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>PnP-DIP-HSI</td>
      <td>31.30</td>
      <td>0.90</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9710184">Paper</a></td>
      <td>33.85</td>
      <td>≥3000</td>
      <td><a href = "https://github.com/mengziyi64/CASSI-Self-Supervised">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>HerosNet</td>
      <td>34.45</td>
      <td>0.97</td>
      <td><a href = "https://arxiv.org/abs/2112.06238">Paper</a></td>
      <td>11.75</td>
      <td>447.18</td>
      <td><a href = "https://github.com/jianzhangcs/HerosNet">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>PnP-HSI</td>
      <td>25.67</td>
      <td>0.70</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9710184">Paper</a></td>
      <td>1.96</td>
      <td>≥3000</td>
      <td><a href = "https://github.com/zsm1211/PnP-CASSI">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>MST</td>
      <td>35.18</td>
      <td>0.95</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper</a></td>
      <td>2.46</td>
      <td>31.40</td>
      <td><a href = "https://github.com/caiyuanhao1998/MST">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>CAE-SRN</td>
      <td>33.26</td>
      <td>0.93</td>
      <td><a href = "https://arxiv.org/abs/2108.07739">Paper</a></td>
      <td>1.25</td>
      <td>83.06</td>
      <td><a href = "https://github.com/Jiamian-Wang/HSI_baseline">Link</a></td>
      <td>/</td>
   </tr>
   <tr align = "center">
      <td>GAP-CCoT</td>
      <td>35.26</td>
      <td>0.95</td>
      <td><a href = "https://arxiv.org/abs/2201.05768">Paper</a></td>
      <td>8.04</td>
      <td>95.60</td>
      <td><a href = "https://github.com/ucaswangls/GAP-CCoT">Link</a></td>
      <td>/</td>
   </tr>
</table>
