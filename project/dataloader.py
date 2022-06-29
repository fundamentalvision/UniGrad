import random

from PIL import ImageFilter
from PIL import Image, ImageOps
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class MultiViewDataInjector():
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transform_list]
        return output


class GaussianBlur(object):

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_transforms():
    transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    transform2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = MultiViewDataInjector([transform1, transform2])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, test_transform


def build_dataloader(cfg):
    debug = getattr(cfg, 'debug', False)

    train_transform, test_transform = build_transforms()

    train_data = datasets.ImageFolder(root=cfg.train_datadir, transform=train_transform)
    memory_data = datasets.ImageFolder(root=cfg.train_datadir, transform=test_transform)
    test_data = datasets.ImageFolder(root=cfg.test_datadir, transform=test_transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    memory_sampler = torch.utils.data.distributed.DistributedSampler(memory_data)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    batch_size = int(cfg.whole_batch_size / torch.distributed.get_world_size())
    
    train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.n_workers,
            sampler=train_sampler,
            drop_last=True,
            persistent_workers=True
        )
    memory_loader = torch.utils.data.DataLoader(
            dataset=memory_data,
            batch_size=512,
            shuffle=False,
            num_workers=cfg.n_workers,
            sampler=memory_sampler,
        )
    test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=256,
            shuffle=False,
            num_workers=cfg.n_workers,
            sampler=test_sampler
        )
    
    return train_loader, memory_loader, test_loader