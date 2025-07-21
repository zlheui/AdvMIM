
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from typing import Tuple
import multiprocessing as mp
from functools import partial

from PIL import Image
from scipy.ndimage import zoom


def normalize_direction(direction):
    direction = np.array(direction).reshape(3, 3)
    for i in range(3):
        norm = np.linalg.norm(direction[:, i])
        if norm == 0:
            raise ValueError("Zero norm direction vector.")
        direction[:, i] /= norm
    return direction.flatten()

def respacing_resize(
    image: sitk.Image, 
    target_xy_spacing: Tuple[float, float],
    target_xy_size: Tuple[int, int] = (512, 512),
    interpolator=sitk.sitkLinear
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
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("LPS")
    lps_image = orienter.Execute(image)

    # print(lps_image.GetDirection())
    # exit(0)

    # image_oriented = sitk.DICOMOrient(image, 'LPS')

    # 执行重采样
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(lps_image.GetDirection())
    resampler.SetOutputOrigin(lps_image.GetOrigin())
    resampler.SetInterpolator(interpolator)
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

    # print(np_volume.shape)
    # exit(0)
    
    # 提取所有深度切片
    slices = [np_volume[i, :, :] for i in range(np_volume.shape[0])]
    return np.stack(slices, axis=0)  # [N, 512, 512]

def preprocess_3d_label(
    label_path: str,
    target_xy_spacing: Tuple[float, float],
    target_xy_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    label = sitk.ReadImage(label_path)
    processed_label = respacing_resize(label, target_xy_spacing, target_xy_size, interpolator=sitk.sitkNearestNeighbor)
    np_label = sitk.GetArrayFromImage(processed_label)
    return np_label#.astype(np.int32)  # [H, W, D], 保持整数类型

def process_single_file(data_id, params):
    """处理单个文件的完整流程"""
    try:
        # 解包参数
        data_root, save_path, label_root, label_save_path, target_spacing = params
        if os.path.exists(os.path.join(save_path, f"{data_id}.npy")):
            return True, data_id
        
        # 构建路径
        source_path = os.path.join(data_root, f"{data_id}_0000.nii.gz")
        # 处理图像
        source_slices = preprocess_3d_volume(
            source_path, 
            target_xy_spacing=target_spacing
        )

        # for i in range(source_slices.shape[0]):
        #     img = (source_slices[i] * 255).astype(np.uint8)
        #     print(img.shape)
        #     Image.fromarray(img).save(f'image_{i}.png')
        # exit(0)


        # # 保存结果
        # np.save(
        #     os.path.join(save_path, f"{data_id}.npy"),
        #     source_slices.transpose(1, 2, 0)
        # )

        # print(source_slices.transpose(1, 2, 0).shape)
        # exit(0)

        img = sitk.GetImageFromArray(source_slices)
        sitk.WriteImage(
            img,
            os.path.join(save_path, f"{data_id}.nii.gz")
        )
        
        # 处理标签
        if label_root is not None:
            label_path = os.path.join(label_root, f"{data_id}.nii.gz")
            processed_label = preprocess_3d_label(
                label_path,
                target_xy_spacing=target_spacing
            )

            label_volume = []
            for ind in range(processed_label.shape[0]):
                image_slie = processed_label[ind, :, :]
                x, y = image_slie.shape[0], image_slie.shape[1]
                image_slie = zoom(image_slie, (256 / x, 256 / y), order=0)
                label_volume.append(image_slie)

            label_volume = np.stack(label_volume, axis=0)

            # print(np.max(label_volume))
            # exit(0)

            # save by volume
            np.savez(os.path.join(label_save_path, f"{data_id}"), image=label_volume)

            # label_img = sitk.GetImageFromArray(processed_label)
            # sitk.WriteImage(
            #     label_img,
            #     os.path.join(label_save_path, f"{data_id}.nii.gz")
            # )
        return True, data_id
    except Exception as e:
        print(f"处理 {data_id} 失败: {str(e)}")
        return False, data_id


# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    
    # # 配置参数
    # config = {
    #     "data_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT_image/",
    #     "save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT_3d/",
    #     "label_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT_label/",
    #     "label_save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT2MR_label/",
    #     "target_spacing": (1.0, 1.0)
    # }
    
    # config = {
    #     "data_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/Training/MRI_image/",
    #     "save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/Training/MRI_3d/",
    #     "label_root": None,
    #     "label_save_path": None,
    #     "target_spacing": (1.0, 1.0)
    # }

    # config = {
    #     "data_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/PublicValidation/MRI_imagesVal",
    #     "save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/PublicValidation/MRI_3d/",
    #     "label_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/PublicValidation/MRI_labelsVal/",
    #     "label_save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/PublicValidation/MRI_3d_label/",
    #     "target_spacing": (1.0, 1.0)
    # }

    # config = {
    #     "data_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/Training/MRI_core_image/",
    #     "save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/Training/MRI_core_3d/",
    #     "label_root": None,
    #     "label_save_path": None,
    #     "target_spacing": (1.0, 1.0)
    # }

    
    # config = {
    #     "data_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/Training/PET_image/",
    #     "save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/Training/PET_3d/",
    #     "label_root": None,
    #     "label_save_path": None,
    #     "target_spacing": (1.0, 1.0)
    # }


    # config = {
    #     "data_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/PublicValidation/PET_imagesVal",
    #     "save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/PublicValidation/PET_3d/",
    #     "label_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/PublicValidation/PET_labelsVal/",
    #     "label_save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/PublicValidation/PET_3d_label/",
    #     "target_spacing": (1.0, 1.0)
    # }
    
    config = {
        "data_root": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/Training/PET_core_image/",
        "save_path": "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/Training/PET_core_3d/",
        "label_root": None,
        "label_save_path": None,
        "target_spacing": (1.0, 1.0)
    }

    
    # 准备任务参数
    img_files = os.listdir(config["data_root"])
    data_ids = [f.split("_0000.nii.gz")[0] for f in img_files]
    
    # 创建进程池
    num_workers = max(1, mp.cpu_count()-2)  # 留出2个核心
    pool = mp.Pool(processes=num_workers)
    
    # 包装固定参数
    worker = partial(
        process_single_file,
        params=(
            config["data_root"],
            config["save_path"],
            config["label_root"],
            config["label_save_path"],
            config["target_spacing"]
        )
    )
    
    # 执行并行处理
    results = []
    for idx, (status, data_id) in enumerate(pool.imap_unordered(worker, data_ids)):
        if status:
            print(f"已完成 {idx+1}/{len(data_ids)}: {data_id}")
        else:
            print(f"失败 {idx+1}/{len(data_ids)}: {data_id}")
        results.append(status)
    
    # 关闭进程池
    pool.close()
    pool.join()
    
    # 输出统计信息
    success = sum(results)
    print(f"\n处理完成！成功: {success}, 失败: {len(results)-success}")
    