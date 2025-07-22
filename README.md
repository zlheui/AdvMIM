# AdvMIM-FLARE25-Task3

## Overview
The repository contains the code for our paper [AdvMIM: Adversarial Masked Image Modeling for Semi-Supervised Medical Image Segmentation](https://arxiv.org/abs/2506.20563). We also provide intructions to apply our method for the unsupervised cross-modal domain adaptation task FLARE25-Task3 by generating synthetic MRI/PET data.


## Environments and Requirements
* CUDA >= 11.3
* python >= 3.9.7

To set up the environment, follow these steps:

```
conda create -n AdvMIM python=3.9.7
conda activate AdvMIM
pip install -r requirements.txt
```


## Dataset
The training Data and validation data are provided by the [FLARE25](https://www.codabench.org/competitions/2296/). In short, there are 2050 CT data (50 labeled data and 2000 pseudo-labeled data (we use the pseudo labels provided by blackbean)), 4817 unlabeled MRI data and 1000 unlabeled PET data for training.

```
|-- FLARE-Task3-DomainAdaption
|   |-- CT
|   |   |-- CT2MR_image
|   |   |-- CT2MR_label
|   |   |-- CT2PET_image
|   |   |-- CT2PET_label
|   |   |-- CT_image
|   |   `-- CT_label
|   |-- MRI
|   |   |-- PublicValidation
|   |   |   |-- MRI_imagesVal
|   |   |   |-- MRI_labelsVal
|   |   |   |-- MRI_3d
|   |   |   `-- MRI_3d_label
|   |   `-- Training
|   |       |-- MRI_image
|   |       `-- MRI_3d
|   |-- PET
|   |   |-- PublicValidation
|   |   |   |-- PET_imagesVal
|   |   |   |-- PET_labelsVal
|   |   |   |-- PET_3d
|   |   |   `-- PET_3d_label
|   |   `-- Training
|   |       |-- PET_image
|   |       `-- PET_3d
```

## Stage1: Style Translation
Please refer to ["Style_Translation/README.md"](Style_Translation/README.md) for detailed information.

## Stage2: Segmentation using Synthesized CT and Real unlabeled MRI/PET Data

### step1: preprosess
```bash
## Process CT2MR_image, CT2MR_label, MRI_3d into 2D format and generate MRI_list files with 2D slice pathes
python preprocess_slice.py
```
### step2: training
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --max_iterations 60000 --root_path ./MRI_list --exp flare25/ct-mri-new-exp-60000-full-exp-1 --num_classes 14
```

## Inference

To infer the testing cases, run this command:

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --iternum 60000 --root_path ./MRI_list --exp flare25/ct-mri-new-exp-60000-full-exp-1 --num_classes 14
```


## Evaluation

To compute the evaluation metrics, run:

```bash
CUDA_VISIBLE_DEVICES=0 python test_2D_fully_transformer.py --iternum 60000 --root_path ./MRI_list --exp flare25/ct-mri-new-exp-60000-full-exp-1 --num_classes 14
```

## Docker

To run the inference using Docker, use the following command:

> Note: This is the official inference script. When running predictions, please replace `input_dir` and `output_dir` with your own input and output directories. The input MRI or PET images must be in `.nii.gz` format.

```bash
docker run --gpus "device=0"  \
   -m 28G  \
   --rm  \
   -v  input_dir:/workspace/inputs/ \
   -v  output_dir:/workspace/outputs/ \
   omnigraft:latest /bin/bash -c "sh predict.sh MRI"

docker run --gpus "device=0"  \
   -m 28G  \
   --rm  \
   -v  input_dir:/workspace/inputs/ \
   -v  output_dir:/workspace/outputs/ \
   omnigraft:latest /bin/bash -c "sh predict.sh PET"
```

Docker Container download link [Onedrive]() 

## 📋 Results

Our method achieves the following performance on [FLARE25](https://www.codabench.org/competitions/2296/)

MRI Data
| Dataset Name       | DSC(%) | NSD(%) |
|--------------------|:------:|:------:|
| Validation Dataset | 54.74% | 57.13% |
| Test Dataset       | (?) | (?) |

PET Data
| Dataset Name       | DSC(%) | NSD(%) |
|--------------------|:------:|:------:|
| Validation Dataset | 52.62% | 32.50% |
| Test Dataset       | (?) | (?) |

## Acknowledgement

 We thank the contributors of [SSL4MIS](https://github.com/HiLab-git/SSL4MIS/tree/master/code), [FLARE25-Task3-LTUDA](https://github.com/xjiangmed/FLARE25-task3-LTUDA/tree/main), and [FLARE24-task3](https://github.com/TJUQiangChen/FLARE24-task3/tree/master).
