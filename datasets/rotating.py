import numpy as np
import skimage.transform
from torchvision import datasets, transforms
import torch

from datasets.nonstationary import NonstationaryLoader


class Rotation(object):
    """Rotate the image by angle."""
    def __init__(self, resample=False, expand=False, center=None):
        self.resample = resample
        self.expand = expand
        self.center = center
        self.angle = 0.
        
    def __call__(self, img):
        print(self.angle)
        return transforms.functional.rotate(img,
                                            self.angle,
                                            self.resample,
                                            self.expand,
                                            self.center)


def rotate_image_batch(images, angle):
    # Normalize images for skimage.transform.rotate
    norm_images = np.copy(images)
    imgs_min = norm_images.min()
    norm_images -= imgs_min 
    norm_max = norm_images.max()
    norm_images /= norm_max

    norm_images = norm_images.transpose(2,3,0,1)
    h,w,n,c = norm_images.shape
    norm_images = norm_images.reshape((h,w,n*c))
    rot_images = skimage.transform.rotate(norm_images, angle)
    rot_images = rot_images.reshape((h,w,n,c))
    rot_images = rot_images.transpose(2,3,0,1)
    # Un-normalize images
    rot_images *= norm_max
    rot_images += imgs_min

    return rot_images


class RotatingData(NonstationaryLoader):
    def __init__(self, dataset, rotate_period, batch_size, draw_and_rotate, kwargs={}):
        super(RotatingData, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.rotate_period = rotate_period
        self.draw_and_rotate = draw_and_rotate
        self.data_iter = iter(self.data_loader)
        self.sample_buffer = None
        self.draws = 0

    def get_data(self):
        self.current_time += 1
        if self.draw_and_rotate:
            try:
                self.sample_buffer = self.data_iter.next()
            except StopIteration:
                # Epoch finished, reset data loader
                self.data_iter = iter(self.data_loader)
                self.sample_buffer = self.data_iter.next()
            self.draws += 1
        else:
            # While loop handles arbitrary skips in time larger than a period
            while int(self.current_time/self.rotate_period) >= self.draws:
                try:
                    self.sample_buffer = self.data_iter.next()
                except StopIteration:
                    # Epoch finished, reset data loader
                    self.data_iter = iter(self.data_loader)
                    self.sample_buffer = self.data_iter.next()
                self.draws += 1

        images, labels = self.sample_buffer
        angle = 360.*(self.current_time/self.rotate_period)
        rot_images = rotate_image_batch(images.numpy(), angle)
        input_data = torch.from_numpy(rot_images).float()
        return input_data, labels
    

class RotatingMNIST(RotatingData):
    def __init__(self, rotate_period, batch_size, train=True, draw_and_rotate=True, kwargs={}):
        dataset = datasets.MNIST('data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        super(RotatingMNIST, self).__init__(dataset, rotate_period, batch_size, draw_and_rotate, kwargs)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10


class RocatingMNIST(RotatingData):
    def __init__(self, rotate_period, batch_size, train=True, draw_and_rotate=True, kwargs={}):
        self.dataset = datasets.MNIST('data', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        self.batch_size = batch_size
        self.train = train
        self.kwargs = kwargs
        super(RocatingMNIST, self).__init__(self.dataset, rotate_period, batch_size, draw_and_rotate, kwargs)
        
    def set_classes(self, filter_classes):
        filtered_dataset = datasets.MNIST('data', train=self.train, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
        mask = np.isin(filtered_dataset.targets, filter_classes)
        filtered_dataset.data = filtered_dataset.data[np.where(mask)[0]]
        filtered_dataset.targets = filtered_dataset.targets[np.where(mask)[0]]
        self.data_loader = torch.utils.data.DataLoader(filtered_dataset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True,
                                                       **self.kwargs)
        self.data_iter = iter(self.data_loader)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10


class RotatingFashionMNIST(RotatingData):
    def __init__(self, rotate_period, batch_size, train=True, draw_and_rotate=True, kwargs={}):
        dataset = datasets.FashionMNIST('data/fashion', train=train, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))
        super(RotatingFashionMNIST, self).__init__(dataset, rotate_period, batch_size, draw_and_rotate, kwargs)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10


class RotatingCIFAR(RotatingData):
    def __init__(self, rotate_period, batch_size, train=True, draw_and_rotate=True, kwargs={}):
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('data/cifar10', train=train, download=True,
                            transform=transform)
        super(RotatingCIFAR, self).__init__(dataset, rotate_period, batch_size, draw_and_rotate, kwargs)

    def get_dim(self):
        return self.data_loader.dataset.data.shape[1:], 10

