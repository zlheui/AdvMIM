from utils import I2IDataset,create_dirs
from i2i_solver import i2iSolver
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import os
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import random
import torch.nn.functional as F

import argparse
import matplotlib.pyplot as plt

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, opts):
    print(f"Running training on rank {rank}.")
    setup(rank, world_size)
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # 设置随机种子
    random.seed(opts.seed + rank)
    np.random.seed(opts.seed + rank)
    torch.manual_seed(opts.seed + rank)
    torch.cuda.manual_seed(opts.seed + rank)
    
    # 创建数据集和数据加载器
    train_dataset = I2IDataset(opts, train=True)
    # val_dataset = I2IDataset(opts, train=False)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    # validation_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=1,
    #     sampler=val_sampler,
    #     num_workers=0,
    #     pin_memory=True
    # )

    # 创建模型
    trainer = i2iSolver(opts)
    trainer = trainer.to(device)
    trainer = DDP(trainer, device_ids=[rank])
    
    iteration = 0
    for epoch in range(200):
        train_sampler.set_epoch(epoch)
        
        for train_data in train_loader:
            for k in train_data.keys():
                train_data[k] = train_data[k].to(device).detach()
            
            trainer.module.gan_forward(train_data['A_img'], train_data['B_img'])
            trainer.module.dis_update()
            trainer.module.gen_update()
            text = trainer.module.verbose()
            
            if rank == 0:  # 只在主进程打印
                sys.stdout.write(f'\r Epoch {epoch}, Iter {iteration}/ {len(train_loader)}, {text}')
            iteration += 1
           
        if rank == 0:  # 只在主进程保存
            trainer.module.save(epoch)
    
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='CT2PET')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--source_path', type=str, default='/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/CT_2d')
    parser.add_argument('--target_path', type=str, default='/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/GAN-data/PET_2d')
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    
    opts = parser.parse_args()
    
    # 设置CUDA设备
    if opts.gpu_ids == '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    
    # 检测GPU数量
    world_size = torch.cuda.device_count()
    print(f"使用 {world_size} 个GPU进行训练")
    
    # 创建必要的目录
    create_dirs(opts.name)
    
    # 启动多进程训练
    mp.spawn(
        train,
        args=(world_size, opts),
        nprocs=world_size,
        join=True
    )

