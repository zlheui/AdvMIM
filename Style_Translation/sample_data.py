import os
import random
import shutil

random.seed(42)  # 使结果可复现


# PET data
save_dir = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/PET_image/"
PET_dir = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/Training/PET_image/"
files = os.listdir(PET_dir) # 从中随机采样500个样本
print('len(files):', len(files))
selected_files = random.sample(files, 500)
for sf in selected_files:
    src_path = os.path.join(PET_dir, sf)
    dst_path = os.path.join(save_dir, sf)
    shutil.move(src_path, dst_path)


# CT data
save_dir = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/CT_image/"
CT_dir1 = "//media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT_gt_image/"
for sf in os.listdir(CT_dir1): # 50 cases
    src_path = os.path.join(CT_dir1, sf)
    dst_path = os.path.join(save_dir, sf)
    shutil.move(src_path, dst_path)

CT_dir2 = "//media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT_pseudo_image/" 
files = os.listdir(CT_dir2) # 从中随机采样450个样本
print('len(files):', len(files))
selected_files = random.sample(files, 450)
for sf in selected_files:
    src_path = os.path.join(CT_dir2, sf)
    dst_path = os.path.join(save_dir, sf)
    shutil.move(src_path, dst_path)

    


# # MRI data
# save_dir = "/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/MRI/MRI_image/"
# files = os.listdir(save_dir)

# # type_a = "amos"
# MRI_dir1 = "/nfs/scratch/xjiangbh/FLARE/FLARE-Task3-DomainAdaption/train_MRI_unlabeled/AMOS-833/"
# files1 = os.listdir(MRI_dir1) # 833， 从中随机采样250个样本
# print('len(files1):', len(files1))
# selected_files = random.sample(files1, 250)
# for sf in selected_files:
#     src_path = os.path.join(MRI_dir1, sf)
#     dst_path = os.path.join(save_dir, sf)
#     shutil.move(src_path, dst_path)

# MRI_dir2 = "/nfs/scratch/xjiangbh/FLARE/FLARE-Task3-DomainAdaption/train_MRI_unlabeled/LLD-MMRI-3984/"
# MODALITIES = ["C+A", "C+Delay", "C+V", "C-pre", 
#              "DWI", "InPhase", "OutPhase", "T2WI"]
# files2 = os.listdir(MRI_dir2) # 3984 各个模态数量均匀，同一个case只选一个模态
# print('len(files2):', len(files2))

# def get_patient_ids(root_path):
#     """获取所有患者ID"""
#     patient_names = []
#     patient_files = {}
#     for d in os.listdir(root_path):
#         name = d.split("_")[0]
#         if name not in patient_names:
#             patient_names.append(name)
            
#         if name not in patient_files.keys():
#             patient_files[name] = []
#             patient_files[name].append(d)
#         else:
#             patient_files[name].append(d)
    

#     return patient_names, patient_files

# # 获取并验证所有患者
# all_patients, patient_files = get_patient_ids(MRI_dir2)
# print(f"总患者数: {len(all_patients)}")  # 应显示498
# SAMPLE_SIZE = 250

# # 计算各模态配额
# base = SAMPLE_SIZE // len(MODALITIES)
# remainder = SAMPLE_SIZE % len(MODALITIES)
# quotas = [base + 1 if i < remainder else base 
#             for i in range(len(MODALITIES))]

# # 打乱患者顺序
# random.shuffle(all_patients)

# # 分层抽样
# sampled = []
# idx = 0
# for modality, quota in zip(MODALITIES, quotas):
#     # 获取当前模态的患者切片
#     patients = all_patients[idx:idx+quota]
#     sampled.extend([(p, modality) for p in patients])
#     idx += quota

# # 生成文件路径
# file_paths = []
# for patient, modality in sampled:
#     p_files = patient_files[patient]
#     for p_f in p_files:
#         if modality in p_f:
#             path = os.path.join(MRI_dir2, p_f)
#             if path not in file_paths:
#                 file_paths.append(path)
# print('len(file_paths):', len(file_paths))


# for src_path in file_paths:
#     sf = src_path.split("/")[-1]
#     dst_path = os.path.join(save_dir, sf)
#     if not os.path.exists(dst_path):
#         shutil.move(src_path, dst_path)
#         