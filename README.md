# BiPFT

This is the implementation of our AAAI2024 paper: BiPFT: Binary Pre-trained Foundation Transformer with Low-rank Estimation of Binarization Residual Polynomials.

Pretrained foundation models offer substantial benefits for a wide range of downstream tasks, which can be one of the most potential techniques to access artificial general intelligence. However, scaling up foundation transformers for maximal task-agnostic knowledge has brought about computational challenges, especially on resource-limited devices such as mobiles. This work proposes the first Binary Pretrained Foundation Transformer (BiPFT) for natural language understanding (NLU) tasks, which remarkably saves 56$\times$ operations and 28$\times$ memory. In contrast to previous task-specific binary transformers, BiPFT exhibits a substantial enhancement in the learning capabilities of binary neural networks (BNNs), promoting BNNs into the era of pre-training. 
Benefiting from extensive pretraining data, we further propose a data-driven binarization method.
Speciﬁcally, we ﬁrst analyze the binarization error in self-attention operations and derive the polynomials of binarization error.
To simulate full-precision self-attention, we define binarization error as binarization residual polynomials, and then introduce low-rank estimators to model these polynomials.
Extensive experiments validate the effectiveness of BiPFTs, surpassing task-specific baseline by 15.4\% average performance on the GLUE benchmark.
BiPFT also demonstrates improved robustness to hyperparameter changes, improved optimization efficiency, and reduced reliance on downstream distillation, which consequently generalize on various NLU tasks and simplify the downstream pipeline of BNNs.

<div align=center>
<img width=60% src="https://github.com/Xingrun-Xing/BiPFT/blob/main/fig1.png"/>
</div>

## Run

### 1. Requirements:
* We pretrain BiPFTs with a single Nvidia A800 Node (8 GPUs).
* python3, pytorch, transformers ...
    
### 2. Data:
* Prepare pretraining data the same as xxx

### 3. Steps to run:
(1) Step1: pretraining
* Change directory `cd model_zoo_exp/BCDNet/a_3f`
* run `bash run.sh`

(2) Step2:  binarizing weights and activations
* Change directory `cd model_zoo_exp/BCDNet/a_3`
* run `bash run.sh`

## Pretrained model
* [BCDNet-A](https://drive.google.com/file/d/103GiQUx422DpwGuTR0bhxfIJvVoR00M5/view?usp=sharing)

## Citation

```
@article{xing2022towards,
  title={Towards Accurate Binary Neural Networks via Modeling Contextual Dependencies},
  author={Xing, Xingrun and Li, Yangguang and Li, Wei and Ding, Wenrui and Jiang, Yalong and Wang, Yufeng and Shao, Jing and Liu, Chunlei and Liu, Xianglong},
  journal={arXiv preprint arXiv:2209.01404},
  year={2022}
}
```
