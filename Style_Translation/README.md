# Style Translation
The codes for the work "A 3D Unsupervised Domain Adaption Framework Combining Style Translation and Self-Training for Abdominal Organs Segmentation".

## 1. Prepare data

## Prepare Style Translation Training Samples
```bash
# sample CT, MRI and PET data for training
python sample_data.py 
# Prepare CT, MRI and PET training data
python prepare_data_for_abdominal_CT_2d.py 
```

## Prepare Style Translation Inference Samples
```bash
# Prepare inference data
python prepare_data_for_abdominal_3d_multi.py 
```

## 2. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 3. Train style translation network

- Train stage one image-to-image translation model for style transfer

```bash
python stage_1_i2i_train_multi.py --source_path "/nfs/scratch/xjiangbh/FLARE/GAN-data/CT/CT_2d/" --target_path "/nfs/scratch/xjiangbh/FLARE/GAN-data/MRI/MRI_2d/" \
python stage_1_i2i_train_multi.py --source_path "/nfs/scratch/xjiangbh/FLARE/GAN-data/CT/CT_2d/" --target_path "/nfs/scratch/xjiangbh/FLARE/GAN-data/PET/PET_2d/" \
```

```bash
python stage_1.5_i2i_inference_multi.py --ckpt_path YOUR_PATH --source_npy_dirpath SOURCE_PATH --target_npy_dirpath TARGET_PATH --save_nii_dirpath SAVE_PATH 
```


## References
* [DAR-UNet](https://github.com/Kaiseem/DAR-UNet)

## Acknowledgement

```bibtex
@article{yao2022darunet,
  title={A novel 3D unsupervised domain adaptation framework for cross-modality medical image segmentation},
  author={Yao, Kai and Su, Zixian and Huang, Kaizhu and Yang, Xi and Sun, Jie and Hussain, Amir and Coenen, Frans},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2022},
  publisher={IEEE}
}

@article{dorent2023crossmoda,
  title={CrossMoDA 2021 challenge: Benchmark of cross-modality domain adaptation techniques for vestibular schwannoma and cochlea segmentation},
  author={Dorent, Reuben and Kujawa, Aaron and Ivory, Marina and Bakas, Spyridon and Rieke, Nicola and Joutard, Samuel and Glocker, Ben and Cardoso, Jorge and Modat, Marc and Batmanghelich, Kayhan and others},
  journal={Medical Image Analysis},
  volume={83},
  pages={102628},
  year={2023},
  publisher={Elsevier}
}
```
