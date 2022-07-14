# Summary of Codes for CASSI Reconstruction
Coded Aperture Snapshot Spectral Imaging(CASSI) is a cutting-edge technology of spectral imaging. The reconstruction algorithms of CASSI, which are devoted to solve the inverse imaging problem, determine the quality and efficiency of imaging. Given the sensing matrix **A** and the 2D CASSI measurement **y**, estimating the underlying image **x** is the fundamental task of reconstruction.

Following the experimental settings of [Meng et al.](https://github.com/mengziyi64/TSA-Net), this repository summarizes the related results and source codes. Meanwhile, the model complexity is measured by parameters counts and floating-point operations(FLOPs) counts. A [CODE](https://github.com/MaxtBIT/Summary-of-Codes-for-HSI-Reconstruction/blob/master/cal_params_FLOPs.py) is provided to calculate the complexity of learning-based algorithms.

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
To analyze the complexity, we have summarized the source codes of mainstream algorithms.  The code links are shown in the table, sorted as MATLAB, PyTorch and TensorFlow. In this stage, the size of reconstructed image is set as 256 * 256 * 28, following the work of Meng et al. The parameter counts and floating-point operation counts(FLOPs) are utilized to evaluate the complexity. Note that, the FLOPs of the iterative optimization methods are not precisely evaluated. In addition, according to different reference, the average PSNR and SSIM on 10 scenes of KAIST Dataset are listed.

<table align = "center">
   <tr align = "center">
      <td rowspan="2">Year</td>
      <td rowspan="2">Method</td>
      <!-- <td colspan="1"></td> -->
      <td colspan="3">Performance</td>
      <td colspan="2">Complexity</td>  
      <td rowspan="2">Code Link</td>  
   </tr>
   <tr align = "center">
      <td>PSNR</td>
      <td>SSIM</td>
      <td>Source</td>
      <td>Params(M)</td>
      <td>FLOPs(GMac)</td>
   </tr>
   <tr align = "center">
      <td rowspan="2">2007</td>
      <td rowspan="2"><a href = "https://ieeexplore.ieee.org/document/4358846">TwIST</a></td>
      <td>23.12</td>
      <td>0.67</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_1  </a><a href = "https://ieeexplore.ieee.org/document/9578572">  Paper_2</a></td>
      <td rowspan="2">/</td>
      <td rowspan="2">≥1000</td>
      <td rowspan="2"><a href = "https://github.com/vbisin/Image-Restoration-Algorithm-TwIST">Python</a></td>
   </tr>
   <tr align = "center">
      <td>22.44</td>
      <td>0.70</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper_3</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="2">2016</td>
      <td rowspan="2"><a href = "https://ieeexplore.ieee.org/document/7532817">GAP-TV</a></td>
      <td>24.36</td>
      <td>0.70</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_1  </a><a href = "https://ieeexplore.ieee.org/document/9578572">  Paper_2</a></td>
      <td rowspan="2">/</td>
      <td rowspan="2">≥1000</td>
      <td rowspan="2"><a href = "https://github.com/Scientific-Research-Algorithm-Toolbox/SCI-algorithms/blob/master/PnP_SCI/%5Bshared%5D/ADMM_Fastdvdnet_xinyuan/dvp_linear_inv.py">Python</a></td>
   </tr>
   <tr align = "center">
      <td>23.73</td>
      <td>0.68</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper_3  </a><a href = "https://arxiv.org/abs/2112.06238">  Paper_4</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="2">2019</td>
      <td rowspan="2"><a href = "https://ieeexplore.ieee.org/document/8481592">DeSCI</a></td>
      <td>25.27</td>
      <td>0.72</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_1  </a><a href = "https://ieeexplore.ieee.org/document/9578572">  Paper_2</a></td>
      <td rowspan="2">/</td>
      <td rowspan="2">≥1000</td>
      <td rowspan="2"><a href = "https://github.com/liuyang12/DeSCI">MATLAB</a></td>
   </tr>
   <tr align = "center">
      <td>25.86</td>
      <td>0.79</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper_3  </a><a href = "https://arxiv.org/abs/2112.06238">  Paper_4</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="2">2019</td>
      <td rowspan="2"><a href = "https://ieeexplore.ieee.org/document/9010044">λ-Net</a></td>
      <td>28.53</td>
      <td>0.84</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_1  </a><a href = "https://ieeexplore.ieee.org/document/9578572">  Paper_2</a></td>
      <td rowspan="2">58.25</td>
      <td rowspan="2">44.59</td>
      <td rowspan="2"><a href = "https://github.com/mlplab/Lambda">PyTorch </a><a href = "https://github.com/xinxinmiao/lambda-net"> TensorFlow</a></td>
   </tr>
   <tr align = "center">
      <td>29.25</td>
      <td>0.89</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper_3  </a><a href = "https://arxiv.org/abs/2112.06238">  Paper_4</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="3">2019</td>
      <td rowspan="3"><a href = "https://ieeexplore.ieee.org/document/8954038">DSSP</a></td>
      <td>30.35</td>
      <td>0.85</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_1  </a><a href = "https://ieeexplore.ieee.org/document/9578572">  Paper_2</a></td>
      <td rowspan="3">0.30</td>
      <td rowspan="3">20.14</td>
      <td rowspan="3"><a href = "https://github.com/mlplab/Lambda">PyTorch </a><a href = "https://github.com/wang-lizhi/DSSP"> TensorFlow</a></td>
   </tr>
   <tr align = "center">
      <td>28.93</td>
      <td>0.83</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper_3  </a><a href = "https://arxiv.org/abs/2112.06238">  Paper_4</a></td>
   </tr>
   <tr align = "center">
      <td>32.39</td>
      <td>0.97</td>
      <td><a href = "https://github.com/wang-lizhi/DSSP">Link_1</a></td>
   </tr>
   <tr align = "center">
      <td>2020</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9156942">DNU</a></td>
      <td>30.74</td>
      <td>0.86</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_1  </a><a href = "https://ieeexplore.ieee.org/document/9578572">  Paper_2</a></td>
      <td>4.47</td>
      <td>293.90</td>
      <td><a href = "https://github.com/wang-lizhi/DeepNonlocalUnrolling">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="2">2020</td>
      <td rowspan="2"><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">TSA-Net</a></td>
      <td>31.46</td>
      <td>0.89</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_1  </a><a href = "https://ieeexplore.ieee.org/document/9578572">  Paper_2</a></td>
      <td rowspan="2">44.25</td>
      <td rowspan="2">135.12</td>
      <td rowspan="2"><a href = "https://github.com/mengziyi64/TSA-Net">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>30.15</td>
      <td>0.89</td>
      <td><a href = "https://link.springer.com/chapter/10.1007/978-3-030-58592-1_12">Paper_3</a></td>
   </tr>
   <tr align = "center">
      <td rowspan="2">2020</td>
      <td rowspan="2"><a href = "https://arxiv.org/abs/2012.08364">GAP-Net</a></td>
      <td>32.13</td>
      <td>0.92</td>
      <td><a href = "https://arxiv.org/abs/2012.08364">Paper_5</a></td>
      <td rowspan="2">4.27</td>
      <td rowspan="2">84.08</td>
      <td rowspan="2"><a href = "https://github.com/mengziyi64/GAP-net">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>32.47</td>
      <td>0.93</td>
      <td><a href = "https://arxiv.org/abs/2108.07739">Paper_6</a></td>
   </tr>
   <tr align = "center">
      <td>2021</td>
      <td><a href = "https://opg.optica.org/prj/fulltext.cfm?uri=prj-9-2-B18&id=446778">PnP-HSI</a></td>
      <td>25.67</td>
      <td>0.70</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9710184">Paper_7</a></td>
      <td>1.96</td>
      <td>≥3000</td>
      <td><a href = "https://github.com/zsm1211/PnP-CASSI">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>2021</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9710184">PnP-DIP-HSI</a></td>
      <td>31.30</td>
      <td>0.90</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9710184">Paper_7</a></td>
      <td>33.85</td>
      <td>≥3000</td>
      <td><a href = "https://github.com/mengziyi64/CASSI-Self-Supervised">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>2021</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9578572">DGSMP</a></td>
      <td>32.63</td>
      <td>0.92</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9578572">Paper_8</a></td>
      <td>3.76</td>
      <td>647.80</td>
      <td><a href = "https://github.com/MaxtBIT/DGSMP">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>2021</td>
      <td><a href = "https://ieeexplore.ieee.org/document/9577826">DTLP</a></td>
      <td>33.88</td>
      <td>0.93</td>
      <td><a href = "https://github.com/wang-lizhi/DTLP_Pytorch">Link_2</a></td>
      <td>3.16</td>
      <td>182.98</td>
      <td><a href = "https://github.com/wang-lizhi/DTLP_Pytorch">PyTorch </a><a href = "https://github.com/zspCoder/DTLP"> TensorFlow</a></td>
   </tr>
   <tr align = "center">
      <td>2021</td>
      <td><a href = "https://arxiv.org/abs/2112.06238">HerosNet</a></td>
      <td>34.45</td>
      <td>0.97</td>
      <td><a href = "https://arxiv.org/abs/2112.06238">Paper_9</a></td>
      <td>11.75</td>
      <td>447.18</td>
      <td><a href = "https://github.com/jianzhangcs/HerosNet">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>2022</td>
      <td><a href = "https://arxiv.org/abs/2108.07739">CAE-SRN</a></td>
      <td>33.26</td>
      <td>0.93</td>
      <td><a href = "https://arxiv.org/abs/2108.07739">Paper_10</a></td>
      <td>1.25</td>
      <td>83.06</td>
      <td><a href = "https://github.com/Jiamian-Wang/HSI_baseline">PyTorch</a></td>
   </tr>
  <tr align = "center">
      <td>2022</td>
      <td><a href = "https://arxiv.org/abs/2203.02149">HDNet</a></td>
      <td>34.34</td>
      <td>0.96</td>
      <td><a href = "https://arxiv.org/abs/2203.02149">Paper_11</a></td>
      <td>2.37</td>
      <td>159.06</td>
      <td><a href = "https://github.com/caiyuanhao1998/MST/blob/main/simulation/train_code/architecture/HDNet.py">PyTorch</a></td>
   </tr>
  <tr align = "center">
      <td>2022</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">MST</a></td>
      <td>35.18</td>
      <td>0.95</td>
      <td><a href = "https://arxiv.org/abs/2111.07910">Paper_12</a></td>
      <td>2.46</td>
      <td>31.40</td>
      <td><a href = "https://github.com/caiyuanhao1998/MST">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>2022</td>
      <td><a href = "https://arxiv.org/abs/2204.07908">MST++</a></td>
      <td>35.99</td>
      <td>0.95</td>
      <td><a href = "https://github.com/caiyuanhao1998/MST">Link_3</a></td>
      <td>1.33</td>
      <td>19.47</td>
      <td><a href = "https://github.com/caiyuanhao1998/MST-plus-plus">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>2022</td>
      <td><a href = "https://arxiv.org/abs/2201.05768">GAP-CCoT</a></td>
      <td>35.26</td>
      <td>0.95</td>
      <td><a href = "https://arxiv.org/abs/2201.05768">Paper_13</a></td>
      <td>8.04</td>
      <td>95.60</td>
      <td><a href = "https://github.com/ucaswangls/GAP-CCoT">PyTorch</a></td>
   </tr>
   <tr align = "center">
      <td>2022</td>
      <td><a href = "https://ieeexplore.ieee.org/abstract/document/9741335">BIRNAT</a></td>
      <td>36.14</td>
      <td>0.97</td>
      <td><a href = "https://ieeexplore.ieee.org/abstract/document/9741335">Paper_14</a></td>
      <td>4.40</td>
      <td>3536.64</td>
      <td><a href = "https://github.com/caiyuanhao1998/MST/blob/main/simulation/train_code/architecture/BIRNAT.py">PyTorch</a></td>
   </tr>
</table>

## Welcome
If you think there is any false, omissions or other materials should be supplemented, please contact us!
