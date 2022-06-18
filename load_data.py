from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


class LoadData:
    def __init__(self):
        self.so2_transform = transforms.Compose([
            transforms.RandomAffine(90, translate=(0., 0.)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))]  # mean and standard deviation, respectively, for normalization
        ])
        self.se2_transform = transforms.Compose([
            transforms.RandomAffine(90, translate=(0.25, 0.25)),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))]  # mean and standard deviation, respectively, for normalization
        ])
        self.vanilla_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))])
        ])

    def get_datasets(self):
        ars = ['se2', 'se2_init']
        for ll in ars:
            if ll == 'se2_init':
                dataset = datasets.MNIST('mnist', train=True, transform=self.vanilla_transform, download=True)

            else:
                dataset = datasets.MNIST('mnist', train=True, transform=self.se2_transform, download=True)
            print(type(dataset))

        ars = ['se2', 'se2_init']
        for ll in ars:
            if ll == 'se2_init':
                dataset = datasets.MNIST('mnist', train=True, transform=self.vanilla_transform, download=True)
            else:
                dataset = datasets.MNIST('mnist', train=True, transform=self.se2_transform, download=True)

            loader = DataLoader(dataset, shuffle=False, batch_size=len(dataset))

            dataset_torch = next(iter(loader))  # since batch_size = len(dataset), this returns the entire dataset
            data_array = dataset_torch[0].numpy()
            target_array = dataset_torch[1].numpy()
            np.save(f'DAT/mnist_' + ll + '.npy', data_array)

        ars = ['se2', 'se2_init']
        for ll in ars:
            if ll == 'se2_init':
                dataset = datasets.MNIST('mnist', train=False, transform=self.vanilla_transform, download=True)
            else:
                dataset = datasets.MNIST('mnist', train=False, transform=self.se2_transform, download=True)

            loader = DataLoader(dataset, shuffle=False, batch_size=len(dataset))

            dataset_torch = next(iter(loader))  # since batch_size = len(dataset), this returns the entire dataset
            data_array_test = dataset_torch[0].numpy()
            target_array_test = dataset_torch[1].numpy()
            np.save(f'DAT/mnist_' + ll + '_test.npy', data_array_test)
            np.save(f'DAT/mnist_' + ll + '_target.npy', target_array_test)
