# Striped-Wrinet: Automatic Wrinkle Segmentation Based on Striped Attention Module

Official implementation of [Striped-Wrinet](https://www.sciencedirect.com/science/article/pii/S1746809423012508).
This repository contains code of model.


## Overview

Striped WriNet's central module is Striped Attention Module (SAM), which consists of Multi-Scale Striped Attention (MSA) and Global Striped Attention (GSA). The ability of the model to extract contextual information about deep folds and shallow fine lines is improved by considering the geometry of the folds to design multi-scale structures and attention mechanisms. SAM is decoupled from the network and can be adapted to any U-shape network. 

We conducted experiments on both the public and private dataset, where the images are finely labeled. For Accuracy (Acc), Dice Score (Dice), and Jaccard Similarity Index (JSI), the proposed method scores 0.8293, 0.7446, and 0.6554 in the public dataset. The model also achieves scores of 0.6888, 0.5728, and 0.4834 in the private dataset with a greater number of short and shallow fine lines.

The experiments were performed on 100 wrinkle pictures, and the Pearson coefficient of wrinkle index and the expert clinical score was 0.94, which proving the robustness of the proposed method in assessing wrinkles under different individuals.

## Quick start

```{bash}
python striped_wrinet.py
```

The output is just for clarify the dataflow inside the model, we set that number of channels is 3, number of classes is 2. if you want to use it for training, please implement the structure for training by yourself.

## Dependency

- Python 3
- [PyTorch](https://pytorch.org/)
- torchvision
- time (pip install time)
- math (pip install math)

### Note
To install the PyTorch for this project, please visit https://pytorch.org/ and install the previous version for PyTorch, for example, 1.10.2, and the corresponding torchvision version, 0.11.3.

## Citation
In case of using this source code for your research, please cite our paper.

```
@inproceedings{Yang2023Striped,
  title={Striped WriNet: Automatic Wrinkle Segmentation Based on Striped Attention Module},
  author={Mingyu Yang, Qili Shen, Detian Xu, Xiaoli Sun and Qingbin Wu},
  year={2023}
}
```



