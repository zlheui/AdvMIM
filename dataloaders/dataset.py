import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image

# random.seed(1337)
# np.random.seed(1337)

class BaseDataSets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.gt_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

            with open(self._base_dir + "/train_gt_slices.list", "r") as f1:
                self.gt_list = f1.readlines()
            self.gt_list = [item.replace("\n", "") for item in self.gt_list]

        elif self.split == "train_unlabel":
            with open(self._base_dir + "/train_unlabel_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            with open(self._base_dir + "/val_gt.list", "r") as f:
                self.gt_list = f.readlines()
            self.gt_list = [item.replace("\n", "") for item in self.gt_list]
        
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        label = None
        if self.split == "train":
            image = np.load(case)['image']
            label = np.load(self.gt_list[idx])['image']
        elif self.split == 'train_unlabel':
            image = np.load(case)['image']
        else:
            image = np.load(case)['image']
            label = np.load(self.gt_list[idx])['image']
            
        sample = {"image": image, "label": label, "pseudo_soft": None}
        
        if self.split == "train" or self.split == 'train_unlabel':
            if self.transform is not None:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample


def random_rot_flip(image, label=None, pseudo_soft=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    # axis = 0
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()

        if pseudo_soft is not None:
            pseudo_soft = np.rot90(pseudo_soft, k, axes=(1,2))
            pseudo_soft = np.flip(pseudo_soft, axis+1)

            return image, label, pseudo_soft, k, axis
        else:
            return image, label, pseudo_soft, k, axis
    else:
        return image, k, axis


def random_rotate(image, label, pseudo_soft=None):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    
    if label is not None:
        label = ndimage.rotate(label, angle, order=0, reshape=False)

    if pseudo_soft is not None:
        pseudo_soft = ndimage.rotate(pseudo_soft, angle, order=0, reshape=False, axes=(1,2))
    
    return image, label, pseudo_soft, angle
    
def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, pseudo_soft = sample["image"], sample["label"], sample["pseudo_soft"]

        # print('image shape: ', image.shape)

        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        sample = {}

        x, y = image.shape
        image_orig = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image_orig = torch.from_numpy(image_orig.astype(np.float32)).unsqueeze(0)

        sample['image_orig'] = image_orig
        sample['k'] = 250
        sample['axis'] = 250
        sample['angle'] = 250

        # print(image.shape)
        # print(label.shape)
        # exit(0)

        if random.random() > 1.5:
            image, label, pseudo_soft, k, axis = random_rot_flip(image, label, pseudo_soft)
            sample['k'] = k
            sample['axis'] = axis
        elif random.random() > 0.0:
            image, label, pseudo_soft, angle = random_rotate(image, label, pseudo_soft)
            sample['angle'] = angle
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

        if label is not None:
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = torch.from_numpy(label.astype(np.uint8))

        if pseudo_soft is not None:
            pseudo_soft = torch.from_numpy(pseudo_soft.astype(np.float32))
            sample['pseudo_soft'] = pseudo_soft

        # sample = {"image": image, "label": label}
        sample['image'] = image
        if label is not None:
            sample['label'] = label
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
