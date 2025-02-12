

<h2 align="center">
Hyperbolic Representations for Prompt Learning
</h2>

<p align="center">
  <!-- <img src="https://img.shields.io/badge/EMNLP-2023-brightgreen"> -->
  <!-- <under review><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a> -->
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
</p>

<p align="center">
Source codes and data for
* [Coling 2024 under review] [Hyperbolic Representations for Prompt Learning]
After the review period, we will open-source the code on our GitHub.
</p>


<!-- # Hyperbolic Prompt Learning


Source codes and data for
* [Coling 2024 under review] [Hyperbolic Representations for Prompt Learning] -->

### Overview
![image](overview.png)



### Setup
We conduct our experiment with Anaconda3. If you have installed Anaconda3, then create the environment for hy:

```shell
conda create -n hy python=3.8.5
conda activate hy
```

After we setup basic conda environment, install pytorch related packages via:

```shell
conda install -n hy pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

Finally, install other python packages we need:

```shell
pip install -r requirements.txt
```

### Data
For SuperGLUE and SQuAD datasets, we download them from the Huggingface Datasets APIs (embedded in our codes).

For sequence tagging datasets, we prepare a non-official packup in the data file. Please use at your own risk.

### Training
Run training scripts in [run_script](run_script) (e.g., RoBERTa for boolq):

```shell
bash run_script/run_boolq_roberta.sh
```

### Hyperbolic 

if you want to use hyperbolic, please set 
```shell
--use_hy True 
--num_c number
--prefix or --prompt
```

### Background: Hyperbolic(left) and Euclidean(right) Geometry
![image](background.png)

### Visualized Representations of Poincar´e (left) and Euclidean model (right)

<div style="text-align: center">
<img src="figures/euc.jpg" width = "25%" position = "left"/>
<img src="figures/hper.jpg" width = "26%" position = "right"/>
</div>


### To do

Next, we will conduct experiments on Dcdoeer-only models (such as GPT) in the future.

*Refer to the e-mail of "authors Response Period" (Above all, the response facility should not be used to report on new results, obtained since the submission deadline closed), we will add more results after the period of notification of acceptance.*




