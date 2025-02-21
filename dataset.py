from typing import Any, Callable, Optional, Tuple
import torch
from torchvision import transforms, datasets
from PIL import Image
from util import generate_torus_point_cloud
import torch.nn.functional as F


class MNIST(datasets.MNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        data_device: str = 'cuda',
        reshape_to_vector = True,
        n_classes = 10
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.data_device = data_device
        self.reshape_to_vector = reshape_to_vector
        self.n_classes = n_classes


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)
            

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.reshape_to_vector:
            img = img.view(-1)
        
        
        target = F.one_hot(target, num_classes=self.n_classes).float()

        train_data = {'image': img, 
                      'label': target}
        return train_data

class TorusPointCloudDataset(torch.utils.data.Dataset):
    def __init__(self, dataSetConfigs):
        R = dataSetConfigs['R']
        r = dataSetConfigs['r']
        num_points = dataSetConfigs['num_points']

        """
        Args:
            R (float): Major radius of the torus.
            r (float): Minor radius of the torus.
            num_points (int): Number of points to generate.
        """
        points = generate_torus_point_cloud(R, r, num_points)
        self.points = torch.tensor(points, dtype=torch.float32)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        return {'image': self.points[idx], 
                'label': 0}


