import numpy as np
import skimage.transform
import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from torch.utils.data.sampler import Sampler, RandomSampler, BatchSampler, SubsetRandomSampler


class NonstationaryLoader(object):
    def __init__(self):
        # Start time at -1 which means no data has been drawn 
        self.current_time = -1

    def time(self):
        return self.current_time

    def set_time(self, new_time):
        self.current_time = new_time
        
    def get_data(self):
        raise NotImplementedError

    def get_dim(self):
        raise NotImplementedError


class DisjointClasses(NonstationaryLoader):
    ''' Train on disjoint classes, switching task every 'period' number of steps.
    'classes_per_batch' determines how many classes are learned together on one training
    task. '''
    def __init__(self, dataset, period, batch_size, classes_per_batch, kwargs={}):
        super(DisjointClasses, self).__init__()
        labels = None
        if 'train_data' in dataset.__dict__.keys():
            labels = dataset.train_labels
        if 'test_data' in dataset.__dict__.keys():
            labels = dataset.test_labels
        labels = np.array(labels)
        n_classes = np.unique(labels).shape[0]
        if n_classes % classes_per_batch != 0:
            raise ValueError('classes_per_batch should evenly divide n_classes')
        n_tasks = n_classes // classes_per_batch
        data_loaders = []
        data_iters = []
        for i in range(n_tasks):
            idxes = []
            for j in range(classes_per_batch):
                idxes.append(np.where(labels == i*classes_per_batch+j)[0])
            idxes = np.concatenate(idxes)
            sampler = BatchSampler(SubsetRandomSampler(idxes), batch_size=batch_size, drop_last=False) 
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_sampler=sampler, **kwargs)
            data_loaders.append(data_loader)
            data_iter = iter(data_loader)
            data_iters.append(data_iter)
        self.period = period
        self.data_loaders = data_loaders
        self.data_iters = data_iters
        self.current_iter = 0
        self.n_classes = n_classes
        self.n_tasks = n_tasks

    def get_data(self):
        self.current_time += 1
        if self.current_time % self.period == 0:
            curr_iter = self.current_time // self.period % self.n_tasks
            self.current_iter = curr_iter
        data_iter = self.data_iters[self.current_iter]
        images, labels = data_iter.next()
        return images, labels

    def iter(self):
        return self.data_iters[self.current_iter]

    def reset_iter(self):
        for i in range(self.n_tasks):
            self.data_iters[i] = iter(self.data_loaders[i])


class BiasedData(NonstationaryLoader):
    ''' Biased batch sampler. Randomly sample 'classes_per_batch' number of
    classes and populate batch with these classes only. This data provider
    does not depend on time but still inherits from 'NonstationaryLoader' to
    be compatible with code that uses other nonstationary datasets. '''
    def __init__(self, dataset, batch_size, classes_per_batch, kwargs={}):
        super(BiasedData, self).__init__()
        labels = None
        if 'train_data' in dataset.__dict__.keys():
            labels = dataset.train_labels
        if 'test_data' in dataset.__dict__.keys():
            labels = dataset.test_labels
        sampler = BiasedBatchSampler(batch_size, labels,
                                     classes_per_batch) 
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, **kwargs)
        self.classes_per_batch = classes_per_batch
        self.data_iter = iter(self.data_loader)

    def get_data(self):
        self.current_time += 1
        images, labels = self.data_iter.next()
        return images, labels


class NormalData(NonstationaryLoader):
    ''' Dataset without any continual learning-specific things. Draws 
    iid batches from all classes. Still inherits 'NonstationaryLoader'
    to provide compatibility with code that uses other nonstationary
    datasets. '''
    def __init__(self, dataset, batch_size, kwargs={}):
        super(NormalData, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
             dataset, batch_size=batch_size, shuffle=True, **kwargs)
        self.data_iter = iter(self.data_loader)

    def get_data(self):
        images, labels = self.data_iter.next()
        return images, labels


class NormalCIFAR(NormalData):
    def __init__(self, batch_size, train=True, kwargs={}):
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('data/cifar10', train=train, download=True,
                            transform=transform)
        super(NormalCIFAR, self).__init__(dataset, batch_size, kwargs)


class BiasedCIFAR(BiasedData):
    def __init__(self, period, batch_size, classes_per_batch, train=True, kwargs={}):
        period = period
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('data/cifar10', train=train, download=True,
                            transform=transform)
        super(BiasedCIFAR, self).__init__(dataset, batch_size, classes_per_batch, kwargs)


class DisjointClassCIFAR(DisjointClasses):
    def __init__(self, period, batch_size, classes_per_batch, train=True, kwargs={}):
        period = period
        transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10('data/cifar10', train=train, download=True,
                            transform=transform)
        super(DisjointClassCIFAR, self).__init__(dataset, period, batch_size, classes_per_batch, kwargs)


class BiasedBatchSampler(Sampler):
    """
    Sampler (WITH replacement) to yield a mini-batch of non-iid indices.

    Args:
        batch_size (int): Size of mini-batch.
        labels (list): List of train/test data labels.
        classes_per_batch (int): How many classes to sample in each batch.
    """
    def __init__(self, batch_size, labels, classes_per_batch, period=None):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(classes_per_batch, int):
            raise ValueError("classes_per_batch should be an integer value, but got "
                             "classes_per_batch={}".format(classes_per_batch))
        
        # check batch_size is divisible by classes_per_batch
        if not batch_size % classes_per_batch == 0:
            raise ValueError("batch_size should be evenly divisible by classes_per_batch")
        # check n_classes isn't greater than total num classes
        n_classes = np.unique(labels).shape[0]
        if classes_per_batch > n_classes:
            raise ValueError("classes_per_batch shouldnt be larger than total num classes")

        self.n_classes = n_classes
        self.class_labels = np.arange(self.n_classes)
        self.batch_size = batch_size
        self.examples_per_class = batch_size // classes_per_batch
        self.labels = labels
        self.classes_per_batch = classes_per_batch
        # create a label -> idx mapping so we don't need to redo at each step
        # all the 0s are in the 0th idx of the list of lists, 1s in the 1st, etc.
        label_to_idxes = []
        for cls in self.class_labels:
            label_to_idxes.append(np.where(self.labels == cls)[0])
        self.label_to_idxes = label_to_idxes

    def __iter__(self):
        while True:
            np.random.shuffle(self.class_labels)
            classes_to_sample = self.class_labels[0:self.classes_per_batch]
            batch = []
            for cls in classes_to_sample:
                np.random.shuffle(self.label_to_idxes[cls])
                class_idxes = self.label_to_idxes[cls][0:self.examples_per_class]
                batch.extend(class_idxes)
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return int(np.ceil(len(self.labels) / self.batch_size))

