# Hyperbolic Prompt Learning


Source codes and data for
* [EMNLP 2023 under review] [Hyperbolic Representations for Prompt Learning]

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

For sequence tagging (NER, SRL) datasets, we prepare a non-official packup in the data file. Please use at your own risk.

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



