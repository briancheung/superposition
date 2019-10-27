import numpy as np
from torchvision import datasets, transforms
import torch

from datasets.nonstationary import NonstationaryLoader


class IncrementingCIFAR(NonstationaryLoader):
    def __init__(self, change_period, batch_size,
                 n_class=10, use_cifar10=True,
                 train=True, seed=1234, kwargs={}):
        super(IncrementingCIFAR, self).__init__()
        transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
        self.dataset_0 = datasets.CIFAR10('data/cifar10', train=train, download=True,
                                          transform=transform)
        self.dataset = datasets.CIFAR100('data/cifar100', train=train, download=True,
                                         transform=transform)
        self.change_period = change_period
        self.transform = transform
        self.batch_size = batch_size
        self.train = train
        self.rand = np.random.RandomState(seed)
        self.n_class = n_class
        self.use_cifar10 = use_cifar10

    def get_data(self):
        self.current_time += 1
        task_offset = int(self.current_time/self.change_period) % int(100/self.n_class)

        # Train initially on the bigger CIFAR10 dataset to stabilize the classifier
        # (neural networks need a lot of data to perform consistently)
        if task_offset == 0 and self.use_cifar10:
            cur_dataset = self.dataset_0
        # Switch to the smaller CIFAR100 dataset to generate new tasks 
        else:
            cur_dataset = self.dataset

        data = cur_dataset.data
        labels = np.array(cur_dataset.targets)

        # Retrieve labels within a contiguous set of class labels
        label_range = range(self.n_class*task_offset, self.n_class*(task_offset+1))
        idx = np.isin(labels, label_range)
        # Indexing like the one below only works when current_data is a numpy variable
        # and not a torch variable.
        current_data = data[idx]
        current_labels = labels[idx]

        # Retrieve a random minibatch from the current task data 
        rand_idx = self.rand.permutation(current_data.shape[0])
        mb_data = current_data[rand_idx[:self.batch_size]]
        mb_labels = current_labels[rand_idx[:self.batch_size]]

        # Preprocess input to (-1,+1) range
        mb_data = torch.Tensor(mb_data.transpose(0,3,1,2))
        mb_data = 2.*(mb_data/255.) - 1.
        
        mb_labels = torch.Tensor(mb_labels) - self.n_class*task_offset
        mb_labels = mb_labels.long()
        return mb_data, mb_labels

    def get_dim(self):
        return self.dataset.data.shape[1:], (self.n_class,)
