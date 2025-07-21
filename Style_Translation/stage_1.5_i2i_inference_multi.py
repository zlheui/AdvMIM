import os
import torch
import torch.multiprocessing as mp
import numpy as np
import random
import argparse
import SimpleITK as sitk
from i2i_solver import i2iSolver
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    torch.distributed.destroy_process_group()

class InferenceDataset(torch.utils.data.Dataset):
    """自定义数据集类"""
    def __init__(self, source_dir):
        self.file_list = sorted(os.listdir(source_dir))
        self.source_dir = source_dir
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        f = self.file_list[idx]

        # # load npy
        # imgs = np.load(os.path.join(self.source_dir, f))

        # load nii.gz
        imgs = sitk.ReadImage(os.path.join(self.source_dir, f))
        imgs = sitk.GetArrayFromImage(imgs)
        imgs = imgs.transpose(1, 2, 0)

        # print('shape: ', imgs.shape)
        # exit(0)

        return f, imgs

def worker(rank, world_size, opts):
    """工作进程函数"""
    # 设置设备和分布式环境
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 初始化模型
    trainer = i2iSolver(None).cuda()
    state_dict = torch.load(opts.ckpt_path, map_location=f'cuda:{rank}')
    # state_dict = torch.load(opts.ckpt_path, map_location='cpu')
    
    # 加载模型参数并转换为DDP
    trainer.enc_c.load_state_dict(state_dict["enc_c"])
    trainer.enc_s_a.load_state_dict(state_dict["enc_s_a"])
    trainer.enc_s_b.load_state_dict(state_dict["enc_s_b"])
    trainer.dec.load_state_dict(state_dict["dec"])
    
    trainer = trainer.to(device)
    trainer = DDP(trainer, device_ids=[rank])
    
    # 准备数据
    dataset = InferenceDataset(opts.source_npy_dirpath)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载目标图像集
    target_images = sorted(os.listdir(opts.target_npy_dirpath))
    
    # 推理循环
    with torch.no_grad():
        for batch_idx, (f, imgs) in enumerate(dataloader):
            # 随机选择目标图像
            idx = random.randint(0, len(target_images)-1)

            # # load npy
            # target_img = np.load(os.path.join(
            #     opts.target_npy_dirpath, 
            #     target_images[idx]
            # ))
            
            # load nii.gz
            target_img = sitk.ReadImage(os.path.join(
                opts.target_npy_dirpath, 
                target_images[idx]
            ))
            target_img = sitk.GetArrayFromImage(target_img)
            target_img = target_img.transpose(1, 2, 0)

            # print('target shape: ', target_img.shape)
            # exit(0)

            # 处理目标图像
            target_slice = target_img[:, :, int(target_img.shape[-1]/2)]
            target_tensor = torch.from_numpy((target_slice*2-1)).unsqueeze(0).unsqueeze(0).cuda().float()
            s = trainer.module.enc_s_b(target_tensor)[0].unsqueeze(0)
            
            # 处理源图像
            print('imgs.shape:', imgs.shape)
            nimgs = np.zeros((imgs.shape[1], imgs.shape[2], imgs.shape[3]), dtype=np.float32)
            for i in range(imgs.shape[-1]):
                img_tensor = (imgs[...,i]*2-1).unsqueeze(0).cuda().float()
                transfered = trainer.module.inference(img_tensor, s)
                transfered = (((transfered + 1)/2).cpu().numpy()).astype(np.float32)[0,0]
                nimgs[...,i] = transfered
                
            # 保存结果
            nimgs = nimgs.transpose(2, 0, 1)
            print('nimgs.shape:', nimgs.shape)
            img = sitk.GetImageFromArray(nimgs)  
            sitk.WriteImage(
                img, 
                # os.path.join(opts.save_nii_dirpath, f[0].replace(".npy", "_0000.nii.gz"))
                os.path.join(opts.save_nii_dirpath, f[0].replace(".nii.gz", "_0000.nii.gz"))
            )
            
            if batch_idx % 10 == 0:
                print(f"[GPU{rank}] Processed {batch_idx+1}/{len(dataloader)}")
    
    cleanup()

if __name__ == "__main__":
    # 配置参数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        # default="/media/mount/zhulei/FLARE25-task3-LTUDA/Style_Translation/CT2MRI/i2i_checkpoints/enc_0066.pt",
        default="/media/mount/zhulei/FLARE25-task3-LTUDA/Style_Translation/CT2PET/i2i_checkpoints/enc_0066.pt",
    )
    parser.add_argument(
        "--source_npy_dirpath",
        type=str,
        default="/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT_3d/",
    )
    parser.add_argument(
        "--target_npy_dirpath",
        type=str,
        # default="/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/Training/MRI_3d/",
        default="/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/PET/Training/PET_3d/",
    )
    parser.add_argument(
        "--save_nii_dirpath",
        type=str,
        # default="/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT2MR_image/",
        default="/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT2PET_image/",
    )
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    opts = parser.parse_args()
    
    # 设置CUDA设备
    if opts.gpu_ids == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    
    # 检测GPU数量
    world_size = torch.cuda.device_count()
    print(f"使用 {world_size} 个GPU进行测试")
    
    # 创建输出目录
    os.makedirs(opts.save_nii_dirpath, exist_ok=True)
    
    # 启动多进程
    world_size = torch.cuda.device_count()
    mp.spawn(
        worker,
        args=(world_size, opts),
        nprocs=world_size,
        join=True
    )