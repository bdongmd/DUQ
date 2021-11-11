import torch
from torch.utils.data import DataLoader, Subset
from torch import Tensor
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from typing import Optional, Tuple
import yaml

# Add Gaussian noise in the training/testing dataset
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# define a gived angle rotation
class GivenAngleRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)

def get_transform(
    gaussian_std: Optional[float] = None,
    random_rotation: Optional[float] = None,
    specific_rotation: Optional[float] = None
) -> transforms.transforms.Compose:
    """Creates transform object based on input parameters
    Args:
        gaussian_std (Optional[float], optional): Std of gaussian blur. Defaults to None.
        random_rotation (Optional[float], optional): Degrees of random rotation. Defaults to None.
        specific_rotation (Optional[float], optional): Degrees of specific rotation. Defaults to None.
    Returns:
        list: Transforms to apply to data loaded
    """
    # Start with standard tensor conversion and normalization
    transform = [transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),]
    # Apply other designated transforms
    if gaussian_std:
        transform.append(AddGaussianNoise(0., gaussian_std))
    if random_rotation:
        transform.insert(0, transforms.RandomRotation(degrees=random_rotation))
    if specific_rotation:
        transform.insert(0, GivenAngleRotation(angle=specific_rotation))
    return transforms.Compose(transform)

def load_data(
    ds_start: int = 0, ds_end: int = 10000, train: bool = True,
    useGPU: bool =False, b_size: int = 64, exclude_number: bool = False,
    transforms: transforms.Compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]),
    static: bool = False
    ) -> Tuple[DataLoader, Subset]:
    """Loads MNIST dataset from torch, either train or test

    Args:
        ds_start (int, optional): Image number on which to start. Defaults to 0.
        ds_end (int, optional): Image number on which to end. Defaults to 10000.
        train (bool, optional): Training dataset - set false for testing. Defaults to True.
        useGPU (bool, optional): Enable gpu running (pin_memory). Defaults to False.
        b_size (int, optional): Batch Size. Defaults to 64.
        exclude_number (bool, optional): Exclude a number. Defaults to False.
        transforms (transforms.Compose, optional): Transform object to apply. Defaults to transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))]).
        static (bool, optional): Don't shuffle the data, useful for comparing Dropout to BNN. Defaults to False.

    Returns:
        Tuple[DataLoader, Subset]: (dataloader, dataset) pair of MNIST data
    """

    dataset = datasets.MNIST('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', download=True, train=train, transform=transforms) 
    if exclude_number and train:
        idx = dataset.targets != 9
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]

    dataset = Subset(dataset, list(range(ds_start, ds_end))) 
    kwargs = {'batch_size': b_size, 'shuffle': True}
    if static == True:
        kwargs["shuffle"] = False
    if useGPU:
        kwargs.update({'pin_memory': True})
    data_loader = DataLoader(dataset, **kwargs)
    return (data_loader, dataset)


def load_configuration_file(f: str):
    return yaml.load(open(f,"r"), Loader=yaml.FullLoader)


class Timer():
    """Class to keep track of time spent in evaluation"""

    def __init__(self) -> None:
        """Initalize object"""
        self.cpu_time = 0
        self.gpu_time = 0
        self.total_time = 0

    def __str__(self) -> str:
        """String representation of timer

        Returns:
            str: if only CPU, shows total time. If GPU exists, time on each device
        """
        if self.gpu_time > 0: 
            percent_cpu = self.cpu_time / self.total_time
            return(f"Total time: {self.total_time:.3f}.\n \
                \tCPU: {self.cpu_time} ({percent_cpu}%)\n \
                \tGPU: {self.gpu_time} ({1-percent_cpu}%)")
        else:
            return f"Total time: {self.total_time:.3f}"

    def add(self, time: float, device: str = "cpu") -> None:
        """Add time to the timer object

        Args:
            time (float): Amount of time to add
            device (str, optional): Which device to credit the time to. Defaults to "cpu".

        Raises:
            ValueError: Raised if device specified is not known
        """
        if device == "cpu":
            self.cpu_time += time
        elif device == "cuda" or device == "gpu":
            self.gpu_time += time
        else: 
            raise ValueError("Unknown device specified in timer")
        
        self.total_time = self.cpu_time + self.gpu_time

