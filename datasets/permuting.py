import numpy as np
import skimage.transform
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
import torch

from datasets.nonstationary import NonstationaryLoader


class PermutingData(NonstationaryLoader):
    def __init__(self, dataset, permute_period, batch_size, seed, kwargs={}):
        super(PermutingData, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.data_iter = iter(self.data_loader)

        self.permute_period = permute_period
        self.seed = seed 

    def get_data(self):
        self.current_time += 1
        try:
            images, labels = self.data_iter.next()
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            images, labels = self.data_iter.next()
        flat_images = images.view([images.shape[0],
                                   images.shape[1]*images.shape[2]*images.shape[3]])

        if self.current_time % self.permute_period == 0:
            # For easier visualization and debugging, do no permutation for first set
            if self.current_time == 0:
                self.perm = np.arange(flat_images.shape[1])
            else:
                rng = np.random.RandomState(self.seed+self.current_time)
                self.perm = rng.permutation(flat_images.shape[1])

        flat_images = flat_images[:, self.perm]
        perm_images = flat_images.view(images.shape)

        return perm_images, labels


class PermutingMNIST(PermutingData):
    def __init__(self, permute_period, batch_size, seed, train=True, kwargs={}):
        dataset = datasets.MNIST('data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        super(PermutingMNIST, self).__init__(dataset, permute_period, batch_size, seed, kwargs)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10 


class PermutingFashionMNIST(PermutingData):
    def __init__(self, permute_period, batch_size, seed, train=True, kwargs={}):
        dataset = datasets.FashionMNIST('data/fashion', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        super(PermutingFashionMNIST, self).__init__(dataset, permute_period, batch_size, seed, kwargs)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10 


class PermutingCIFAR(PermutingData):
    def __init__(self, permute_period, batch_size, seed, train=True, kwargs={}):
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('data/cifar10', train=train, download=True,
                                   transform=transform)
        super(PermutingCIFAR, self).__init__(dataset, permute_period, batch_size, seed, kwargs)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10 


class PermutingSVHN(PermutingData):
    def __init__(self, permute_period, batch_size, seed, train=True, kwargs={}):
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        if train:
            split = 'train'
        else:
            split = 'test'
        dataset = datasets.SVHN('data/svhn', split=split, download=True,
                                transform=transform)
        super(PermutingSVHN, self).__init__(dataset, permute_period, batch_size, seed, kwargs)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10 
