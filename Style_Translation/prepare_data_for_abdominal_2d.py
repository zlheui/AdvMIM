
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from typing import Tuple



def respacing_resize(
    image: sitk.Image, 
    target_xy_spacing: Tuple[float, float],
    target_xy_size: Tuple[int, int] = (512, 512)
) -> sitk.Image:
    """重采样到目标XY平面分辨率并调整尺寸"""
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # 目标参数设置
    target_spacing = (target_xy_spacing[0], target_xy_spacing[1], original_spacing[2])
    target_size = (
        int(round(original_size[0] * original_spacing[0] / target_spacing[0])),
        int(round(original_size[1] * original_spacing[1] / target_spacing[1])),
        original_size[2]  # 保持深度方向不变
    )

    # 执行重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(image)

    # 转换为NumPy数组 (顺序: [depth, height, width])
    np_image = sitk.GetArrayFromImage(resampled_image)
    
    # 获取重采样后的图像的最小值，用于填充
    min_val = np.min(np_image)

    # 中心裁剪/填充到目标尺寸
    current_height, current_width = np_image.shape[1], np_image.shape[2]
    
    # 处理高度
    if current_height < target_xy_size[1]:
        pad = (target_xy_size[1] - current_height)
        pad_before = pad // 2
        pad_after = pad - pad_before
        np_image = np.pad(np_image, ((0, 0), (pad_before, pad_after), (0, 0)), mode="constant", constant_values=min_val)
    elif current_height > target_xy_size[1]:
        start = (current_height - target_xy_size[1]) // 2
        np_image = np_image[:, start:start+target_xy_size[1], :]

    # 处理宽度
    if current_width < target_xy_size[0]:
        pad = (target_xy_size[0] - current_width)
        pad_before = pad // 2
        pad_after = pad - pad_before
        np_image = np.pad(np_image, ((0, 0), (0, 0), (pad_before, pad_after)), mode="constant", constant_values=min_val)
    elif current_width > target_xy_size[0]:
        start = (current_width - target_xy_size[0]) // 2
        np_image = np_image[:, :, start:start+target_xy_size[0]]

    # 转换回SimpleITK图像
    output_image = sitk.GetImageFromArray(np_image)
    output_image.SetSpacing((target_spacing[0], target_spacing[1], original_spacing[2]))
    return output_image

def normalize_volume(np_volume: np.ndarray) -> np.ndarray:
    """归一化到[0, 1]范围"""
    min_val = np.min(np_volume)
    max_val = np.max(np_volume)
    if max_val != min_val:
        return (np_volume - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(np_volume)

def preprocess_3d_volume(
    image_path: str,
    target_xy_spacing: Tuple[float, float],
    target_xy_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """完整预处理流程"""
    # 读取图像
    image = sitk.ReadImage(image_path)
    
    # 重采样+尺寸调整
    processed_image = respacing_resize(image, target_xy_spacing, target_xy_size)
    
    # 转换为NumPy数组并归一化
    np_volume = sitk.GetArrayFromImage(processed_image)  # [D, H, W]
    np_volume = normalize_volume(np_volume)
    
    # 提取所有深度切片
    slices = [np_volume[i, :, :] for i in range(np_volume.shape[0])]
    return np.stack(slices, axis=0)  # [N, 512, 512]

# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # step1: Respacing the source and target domain images (both should be gray) with the same XY plane resolutions, 
    # and crop/pad to the size of [512, 512, d] in terms of [width, height, depth
    # step2: Normalize each 3D images to [0, 1], and extract 2D slices from 3D volumes along depth-axis.
    # step3: Stack the list of 2D slices at zero dimension for the two domains respectively, resulting in 3D tensor with size of [N, 512, 512].
    
    # # ------------------------------------------------------------------
    # data_root = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/CT_image/"
    # save_path = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/CT_2d/"
    # count = 0
    # img_files  = os.listdir(data_root)
    # for image_id in img_files:
    #     data_id = image_id.split("_0000.nii.gz")[0]
    #     source_path = os.path.join(data_root, data_id + "_0000.nii.gz")
        
    #     # 步骤1: 统一XY平面分辨率 (例如: 1.0x1.0mm)
    #     source_slices = preprocess_3d_volume(source_path, target_xy_spacing=(1.0, 1.0))
        
    #     print(data_id, f"source shape: {source_slices.shape}")  # 输出 [N, 512, 512]
    #     print('process: ', count)
    #     N = source_slices.shape[0]
    #     for nn in range(N):
    #         # print('saving '+ str(nn) + '  '+str(source_slices[nn:nn+1, :, :].shape))
    #         # np.save(os.path.join(save_path, data_id + "_0000_slice_"+str(nn)+".npy"), source_slices[nn:nn+1, :, :])
    #         np.savez_compressed(os.path.join(save_path, data_id + "_0000_slice_"+str(nn)), image=source_slices[nn:nn+1, :, :])

    #     count += 1
    # # ------------------------------------------------------------------
    # MRI_data_root = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/MRI/MRI_image/"
    # MRI_save_path = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/MRI/MRI_2d/"
    
    # MRI_img_files  = os.listdir(MRI_data_root)
    # for image_id in MRI_img_files:
    #     data_id = image_id.split("_0000.nii.gz")[0]
    #     target_path = os.path.join(MRI_data_root, data_id + "_0000.nii.gz")
        
    #     # 步骤1: 统一XY平面分辨率 (例如: 1.0x1.0mm)
    #     target_slices = preprocess_3d_volume(target_path, target_xy_spacing=(1.0, 1.0))
    #     print(f"target shape: {target_slices.shape}")
        
    #     N = target_slices.shape[0]
    #     for nn in range(N):
    #         print('saving '+ str(nn) + '  '+str(target_slices[nn:nn+1, :, :].shape))
    #         np.save(os.path.join(MRI_save_path, data_id + "_0000_slice_"+str(nn)+".npy"), target_slices[nn:nn+1, :, :])
       
    # ------------------------------------------------------------------
    PET_data_root = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/PET_image/"
    PET_save_path = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/PET_2d/"
    
    PET_img_files  = os.listdir(PET_data_root)
    count = 0

    for image_id in PET_img_files:
        data_id = image_id.split("_0000.nii.gz")[0]
        target_path = os.path.join(PET_data_root, data_id + "_0000.nii.gz")
        
        # 步骤1: 统一XY平面分辨率 (例如: 1.0x1.0mm)
        target_slices = preprocess_3d_volume(target_path, target_xy_spacing=(1.0, 1.0))
        print(data_id, f"target shape: {target_slices.shape}")
        print('process: ', count)
        N = target_slices.shape[0]
        for nn in range(N):
            # print('saving '+ str(nn) + '  '+str(target_slices[nn:nn+1, :, :].shape))
            # np.save(os.path.join(PET_save_path, data_id + "_0000_slice_"+str(nn)+".npy"), target_slices[nn:nn+1, :, :])
            np.savez_compressed(os.path.join(PET_save_path, data_id + "_0000_slice_"+str(nn)), image=target_slices[nn:nn+1, :, :])
        count += 1

    