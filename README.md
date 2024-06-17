# SSJE

Code for A span-sharing joint extraction framework for harvesting aspect sentiment triplets.(Knowledge-Based Systems'2022)

## Model Architecture

![img](https://gitee.com/YDLinGit/blog-img/raw/master/1-s2.0-S0950705122001381-gr2_lrg.jpg)



## Requirments

### Enviroment

```
python 3.6.1
numpy==1.17.4
tensorboardX==1.6
scikit-learn==0.25.0
torch==1.4.0
transformers
spacy==3.0.1
tpqm
```

use pip command


```bash
pip install -r requirements.txt
```

**Notes**

You can download the bert-base-cased from [here](https://huggingface.co/bert-base-cased)



### File Directory Tree

The directory tree of SSJE:

```
├─bert
│  └─base-uncased
├─data
│  ├─14lap
│  ├─14res
│  ├─15res
│  └─16res
├─layer
├─log
├─model
├─templates
├─trainer
```



## Get Started

### Datasets

download the  ASTE-Dataset-V2 dataset from  [here]( https://github.com/xuuuluuu/SemEval-Triplet-data/tree/master/ASTE-Data-V2-EMNLP2020)  or you can just use the data set that we've already processed.

### Run  

- Training and testing model effects on 2014 Restaurant

```bash
python train_triplet.py --dataset 14res --max_span_size 8
```


## HI bro listen ，
the original code just like shit !
here are some notes !

- all pip install (suggest conda install replaced)
- all tools should be latest ignoring the fucking requirements and ablove all
- run the test.py in trainer first
- use python == 11

## MORE INFO 
- windows 10 
- nvcc --version outputs is:
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
- torch v = 2.3.0+cu118
