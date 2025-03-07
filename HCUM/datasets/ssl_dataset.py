import torch

from .data_utils import split_ssl_data, sample_labeled_data
from .dataset import BasicDataset
from collections import Counter
import torchvision
import numpy as np
from torchvision import transforms
import json
import os

import random
from .augmentation.randaugment import RandAugment

from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
from datasets.DistributedProxySampler import DistributedProxySampler

import gc
import sys
import copy
from PIL import Image

mean, std = {}, {}

mean['isic'] = [0.466, 0.471, 0.380]
mean['chest14'] = [0.485, 0.456, 0.406]
mean['rsna'] = [0.2567, 0.2584, 0.2574]


std['isic'] = [0.195, 0.194, 0.192]
std['chest14'] = [0.229, 0.224, 0.225]
std['rsna'] = [0.2791, 0.2810, 0.2789]



def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def _find_classes(dir):
    if sys.version_info >= (3, 5):
        # Faster and available in Python 3.5 and above
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    else:
        classes = [d for d in os.listdir(
            dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None, num_labels=-1):
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return x.lower().endswith(extensions)

    lb_idx = {}
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            random.shuffle(fnames)
            if num_labels != -1:
                fnames = fnames[:num_labels]
            if num_labels != -1:
                lb_idx[target_class] = fnames
            for fname in fnames:
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

    if num_labels != -1:
        with open('./sampled_label_idx.json', 'w') as f:
            json.dump(lb_idx, f)
    del lb_idx
    gc.collect()
    return instances


class ImagenetDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform, ulb, num_labels=-1, target_transform=None):
        super().__init__(root, transform)
        self.ulb = ulb
        self.target_transform = target_transform
        is_valid_file = None
        self.root = root
        self.transform = transform
        self.ulb = ulb
        self.extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                           '.pgm', '.tif', '.tiff', '.webp')
        self._find_classes = _find_classes
        self.make_dataset = make_dataset

        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(
            self.root, class_to_idx, self.extensions, is_valid_file, num_labels)

        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            msg += "Supported extensions are: {}".format(
                ",".join(self.extensions))
            raise RuntimeError(msg)

        self.loader = default_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        if self.ulb:
            self.strong_transform = copy.deepcopy(transform)
            self.strong_transform.transforms.insert(0, RandAugment(3, 5))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample_transformed = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (index, sample_transformed, target) if not self.ulb else (
            index, sample_transformed, self.strong_transform(sample))


class CropTopTwoThirds(object):
    def __call__(self, img):
        width, height = img.size
        crop_height = int(height * 2 / 3)
        img = img.crop((0, 0, width, crop_height))

        return img


class ImageNetLoader:
    def __init__(self, root_path, num_labels=-1, num_class=1000, dataset='isic'):
        self.root_path = os.path.join(root_path, dataset)
        # self.root_path = root_path
        self.num_labels = num_labels // num_class
        self.dataset = dataset

    def get_transform(self, train, ulb, dataset):
        if train:
            transform = transforms.Compose([
                # CropTopTwoThirds(),
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(mean[self.dataset], std[self.dataset])])
        else:
            transform = transforms.Compose([
                # CropTopTwoThirds(),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean[self.dataset], std[self.dataset])])
        return transform

    def get_lb_train_data(self):
        transform = self.get_transform(train=True, ulb=False, dataset=self.dataset)
        data = ImagenetDataset(root=os.path.join(self.root_path, "train"), transform=transform, ulb=False,
                               num_labels=self.num_labels)
        return data

    def get_ulb_train_data(self):
        transform = self.get_transform(train=True, ulb=True, dataset=self.dataset)
        data = ImagenetDataset(root=os.path.join(
            self.root_path, "train"), transform=transform, ulb=True)
        return data

    def get_lb_test_data(self):
        transform = self.get_transform(train=False, ulb=False, dataset=self.dataset)
        data = ImagenetDataset(root=os.path.join(
            self.root_path, "test"), transform=transform, ulb=False)
        return data


def get_transform(mean, std, crop_size, train=True):
    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomCrop(
                                       crop_size, padding=4, padding_mode='reflect'),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])


class SSL_Dataset:


    def __init__(self,
                 args,
                 alg='fixmatch',
                 name='imagenet',
                 train=True,
                 num_classes=10,
                 data_dir='./data'):

        self.args = args
        self.alg = alg
        self.name = name
        self.train = train
        self.num_classes = num_classes
        self.data_dir = data_dir
        crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
        self.transform = get_transform(mean[name], std[name], crop_size, train)

    def get_data(self, svhn_extra=True):

        dset = getattr(torchvision.datasets, self.name.upper())
        if 'CIFAR' in self.name.upper():
            dset = dset(self.data_dir, train=self.train, download=True)
            data, targets = dset.data, dset.targets
            return data, targets
        elif self.name.upper() == 'SVHN':
            if self.train:
                if svhn_extra:  # train+extra
                    dset_base = dset(self.data_dir, split='train', download=True)
                    data_b, targets_b = dset_base.data.transpose([0, 2, 3, 1]), dset_base.labels
                    dset_extra = dset(self.data_dir, split='extra', download=True)
                    data_e, targets_e = dset_extra.data.transpose([0, 2, 3, 1]), dset_extra.labels
                    data = np.concatenate([data_b, data_e])
                    targets = np.concatenate([targets_b, targets_e])
                    del data_b, data_e
                    del targets_b, targets_e
                else:  # train_only
                    dset = dset(self.data_dir, split='train', download=True)
                    data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            else:  # test
                dset = dset(self.data_dir, split='test', download=True)
                data, targets = dset.data.transpose([0, 2, 3, 1]), dset.labels
            return data, targets
        elif self.name.upper() == 'STL10':
            split = 'train' if self.train else 'test'
            dset_lb = dset(self.data_dir, split=split, download=True)
            dset_ulb = dset(self.data_dir, split='unlabeled', download=True)
            data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
            ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])
            return data, targets, ulb_data



    def get_dset(self, is_ulb=False,
                 strong_transform=None, onehot=False):


        if self.name.upper() == 'STL10':
            data, targets, _ = self.get_data()
        else:
            data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform

        return BasicDataset(self.alg, data, targets, num_classes, transform,
                            is_ulb, strong_transform, onehot)

    def get_ssl_dset(self, num_labels, index=None, include_lb_to_ulb=True,
                     strong_transform=None, onehot=False):

        # Supervised top line using all data as labeled data.
        if self.alg == 'fullysupervised':
            lb_data, lb_targets = self.get_data()
            lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                                   self.transform, False, None, onehot)
            return lb_dset, None

        if self.name.upper() == 'STL10':
            lb_data, lb_targets, ulb_data = self.get_data()
            if include_lb_to_ulb:
                ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
            lb_data, lb_targets, _ = sample_labeled_data(self.args, lb_data, lb_targets, num_labels, self.num_classes)
            ulb_targets = None
        else:
            data, targets = self.get_data()
            lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(self.args, data, targets,
                                                                        num_labels, self.num_classes,
                                                                        index, include_lb_to_ulb)
        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(self.num_classes)]
        for c in lb_targets:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(self.name) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)
        # print(Counter(ulb_targets.tolist()))
        lb_dset = BasicDataset(self.alg, lb_data, lb_targets, self.num_classes,
                               self.transform, False, None, onehot)

        ulb_dset = BasicDataset(self.alg, ulb_data, ulb_targets, self.num_classes,
                                self.transform, True, strong_transform, onehot)
        # print(lb_data.shape)
        # print(ulb_data.shape)
        return lb_dset, ulb_dset


