import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_ds

import pickle

import SimpleITK as sitk

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mimunet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.05,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=50,
                    help='labeled data')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')

args = parser.parse_args()


from config import get_config
from networks.vision_transformer import MiMSwinUnet as ViT_seg

parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

args = parser.parse_args()
args.patch_size = [224, 224]
config = get_config(args)


class CNNDiscriminator(nn.Module):

    def __init__(self, in_chans=4, ndf=64):
        super(CNNDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_chans, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)

        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations


    def create_model(ema=False):
        # Network definition
        model = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
        model.load_from(config)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model.load_from(config)

    def create_model2(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model2 = create_model2()

    discriminator = CNNDiscriminator(in_chans=num_classes).cuda()
    discriminator2 = CNNDiscriminator(in_chans=num_classes).cuda()

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))

    db_train_unlabeled = BaseDataSets(base_dir=args.root_path, split="train_unlabel", transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))

    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)

    trainloader_unlabeled = DataLoader(db_train_unlabeled, batch_size=batch_size, shuffle=True,
                             num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    model.train()
    model2.train()

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    optimizer_adv = optim.SGD(discriminator.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    optimizer_adv2 = optim.SGD(discriminator2.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)


    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:

        iter_labeled = iter(trainloader)
        iter_unlabeled = iter(trainloader_unlabeled)
        iterations = len(trainloader)


        for i_batch in range(iterations):

            try:
                sampled_batch = iter_labeled.next()
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            except:
                iter_labeled = iter(trainloader)
                sampled_batch = iter_labeled.next()
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            # img_itk = sitk.GetImageFromArray(volume_batch.cpu().numpy().astype(np.float32))
            # sitk.WriteImage(img_itk, "test.nii.gz")
            # print('test')

            # volume_batch = volume_batch.cpu().numpy()
            # for i in range(volume_batch.shape[0]):
            #     img = (volume_batch[i] * 255).astype(np.uint8)
            #     # print(img.shape)
            #     Image.fromarray(img[0]).save(f'image_{i}.png')



            try:
                sampled_batch_unlabeled = iter_unlabeled.next()
                volume_batch_unlabeled = sampled_batch_unlabeled['image']
            except:
                iter_unlabeled = iter(trainloader_unlabeled)
                sampled_batch_unlabeled = iter_unlabeled.next()
                volume_batch_unlabeled = sampled_batch_unlabeled['image']

            volume_batch_unlabeled = volume_batch_unlabeled.cuda()


            # volume_batch_unlabeled = volume_batch_unlabeled.cpu().numpy()
            # for i in range(volume_batch_unlabeled.shape[0]):
            #     img = (volume_batch_unlabeled[i] * 255).astype(np.uint8)
            #     # print(img.shape)
            #     Image.fromarray(img[0]).save(f'unlabeled_image_{i}.png')
            # exit(0)


            outputs, outputs_masked, reconstruct_outputs, _, _, _, _ = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            outputs2, outputs_masked2, reconstruct_outputs2, _, _ = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            outputs_masked_soft2 = torch.softmax(outputs_masked2, dim=1)

            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            loss_masked_ce = ce_loss(outputs_masked, label_batch[:].long())

            loss_ce2 = ce_loss(outputs2, label_batch[:].long())
            loss_dice2 = dice_loss(outputs_soft2, label_batch.unsqueeze(1))
            loss_masked_ce2 = ce_loss(outputs_masked2, label_batch[:].long())
            loss_masked_dice2 = dice_loss(outputs_masked_soft2, label_batch.unsqueeze(1))

            # print(volume_batch.shape)
            # print(volume_batch_unlabeled.shape)
            # exit(0)

            outputs_unlabeled, outputs_masked_unlabeled, reconstruct_outputs_unlabeled, _, x_feat_unlabeled, _, _ = model(volume_batch_unlabeled)
            outputs_soft_unlabeled = torch.softmax(outputs_unlabeled, dim=1)
            outputs_masked_soft_unlabeled = torch.softmax(outputs_masked_unlabeled, dim=1)

            outputs_unlabeled2, outputs_masked_unlabeled2, reconstruct_outputs_unlabeled2, _, x_feat_unlabeled2 = model2(volume_batch_unlabeled)
            outputs_soft_unlabeled2 = torch.softmax(outputs_unlabeled2, dim=1)
            outputs_masked_soft_unlabeled2 = torch.softmax(outputs_masked_unlabeled2, dim=1)


            pseudo_weights1, pseudo_outputs1 = torch.max(
                outputs_soft_unlabeled.detach(), dim=1, keepdim=False)
            pseudo_weights2, pseudo_outputs2 = torch.max(
                outputs_soft_unlabeled2.detach(), dim=1, keepdim=False)

            consistency_loss1 = dice_loss(
                outputs_soft_unlabeled, pseudo_outputs2.unsqueeze(1))
            consistency_loss2 = dice_loss(
                outputs_soft_unlabeled2, pseudo_outputs1.unsqueeze(1))

            consistency_masked_loss1 = torch.nn.CrossEntropyLoss(reduction='none')(outputs_masked_unlabeled, pseudo_outputs2.long())
            consistency_masked_loss1 = torch.sum(pseudo_weights2 * consistency_masked_loss1) / torch.sum(pseudo_weights2)

            consistency_masked_loss2 = torch.nn.CrossEntropyLoss(reduction='none')(outputs_masked_unlabeled2, pseudo_outputs1.long())
            consistency_masked_loss2 = torch.sum(pseudo_weights1 * consistency_masked_loss2) / torch.sum(pseudo_weights1)

            adv_output = discriminator(outputs_masked_soft_unlabeled)
            adv_loss = torch.nn.MSELoss()(adv_output, torch.ones_like(adv_output).cuda())

            adv_output2 = discriminator2(outputs_masked_soft_unlabeled2)
            adv_loss2 = torch.nn.MSELoss()(adv_output2, torch.ones_like(adv_output2).cuda())


            loss = loss_ce + loss_dice + loss_ce2 + loss_dice2 + loss_masked_ce + loss_masked_ce2 + loss_masked_dice2 + consistency_masked_loss1 + consistency_masked_loss2 + 0.1 * consistency_loss1 + 0.1 * consistency_loss2 + 0.001 * adv_loss + 0.001 * adv_loss2 

            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer_adv.zero_grad()
            optimizer_adv2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(model2.parameters(), 5.0)
            optimizer.step()
            optimizer2.step()

            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer_adv.zero_grad()
            optimizer_adv2.zero_grad()

            dis_output1 = discriminator(outputs_soft.detach())
            dis_output2 = discriminator(outputs_masked_soft_unlabeled.detach())

            dis_loss_1 = torch.nn.MSELoss()(dis_output1, torch.ones_like(dis_output1).cuda())
            dis_loss_2 = torch.nn.MSELoss()(dis_output2, torch.zeros_like(dis_output2).cuda())

            dis_output12 = discriminator2(outputs_soft2.detach())
            dis_output22 = discriminator2(outputs_masked_soft_unlabeled2.detach())

            dis_loss_12 = torch.nn.MSELoss()(dis_output12, torch.ones_like(dis_output12).cuda())
            dis_loss_22 = torch.nn.MSELoss()(dis_output22, torch.zeros_like(dis_output22).cuda())

            dis_loss = 0.1 * (dis_loss_1 + dis_loss_2 + dis_loss_12 + dis_loss_22)

            dis_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(discriminator2.parameters(), 5.0)

            optimizer_adv.step()
            optimizer_adv2.step()

            optimizer.zero_grad()
            optimizer2.zero_grad()
            optimizer_adv.zero_grad()
            optimizer_adv2.zero_grad()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_adv.param_groups:
                param_group['lr'] = lr_
            for param_group in optimizer_adv2.param_groups:
                param_group['lr'] = lr_


            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            
            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                print('\nsupervised_loss: ', loss_ce.item(), loss_dice.item(), loss_ce2.item(), loss_dice2.item())
                print('masked loss: ', loss_masked_ce.item(), loss_masked_ce2.item(), loss_masked_dice2.item())
                print('consistency loss: ', consistency_loss1.item(), consistency_loss2.item())
                print('consistency masked loss:', consistency_masked_loss1.item(), consistency_masked_loss2.item())
                print('adv_loss: ', adv_loss.item(), dis_loss_1.item(), dis_loss_2.item())
                print('adv_loss2: ', adv_loss2.item(), dis_loss_12.item(), dis_loss_22.item())
                print(outputs_soft_unlabeled.mean(dim=(0,2,3)).detach().cpu().numpy())
                print(outputs_masked_soft_unlabeled.mean(dim=(0,2,3)).detach().cpu().numpy())
                print(outputs_soft_unlabeled2.mean(dim=(0,2,3)).detach().cpu().numpy())
                print(outputs_masked_soft_unlabeled2.mean(dim=(0,2,3)).detach().cpu().numpy())

            #     model.eval()
            #     metric_list = 0.0
            #     for i_batch, sampled_batch in enumerate(valloader):
            #         metric_i = test_single_volume(
            #             sampled_batch["image"], sampled_batch["label"], model, classes=num_classes, patch_size=[224,224])
            #         metric_list += np.array(metric_i)
            #     metric_list = metric_list / len(db_val)
            #     for class_i in range(num_classes-1):
            #         writer.add_scalar('info/val_{}_dice'.format(class_i+1),
            #                           metric_list[class_i, 0], iter_num)
            #         writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
            #                           metric_list[class_i, 1], iter_num)

            #     performance = np.mean(metric_list, axis=0)[0]

            #     mean_hd95 = np.mean(metric_list, axis=0)[1]
            #     writer.add_scalar('info/val_mean_dice', performance, iter_num)
            #     writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            #     if performance > best_performance:
            #         best_performance = performance
            #         save_mode_path = os.path.join(snapshot_path,
            #                                       'iter_{}_dice_{}.pth'.format(
            #                                           iter_num, round(best_performance, 4)))
            #         save_best = os.path.join(snapshot_path,
            #                                  '{}_best_model.pth'.format(args.model))
            #         # torch.save(model.state_dict(), save_mode_path)
            #         # torch.save(model.state_dict(), save_best)

            #     logging.info(
            #         'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
            #     model.train()

            if iter_num % 3000 == 0 and iter_num > 25000:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
