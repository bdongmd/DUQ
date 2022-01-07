"""
Unit testing for model
Run via: pytest -v tests.py
tjb
"""
from models import network
import torch
import torchvision
from utils import load_data
import pytest

@pytest.fixture
def train_data():
    train_loader, _ = load_data(train=True)
    return train_loader

@pytest.fixture
def test_data():
    test_loader, _ = load_data(train=False)
    return test_loader

@pytest.fixture
def sample_model():
    from train import get_model
    return get_model(trainmodel="CNNWeak", dropout_rate=0.2, exclude_number=False)

### Dataset Tests ###
def test_train_data_size(train_data):
    """Test number of training images loaded by default"""
    assert len(train_data.dataset) == 10000

def test_train_image_shape(test_data):
    """Test shape of training images. Expect (1,28,28)"""
    train_loader, _ = load_data(train=True, ds_start=0, ds_end=2)
    assert train_loader.dataset[0][0].shape == torch.Size([1,28,28])
    
def test_test_data_size():
    """Test number of testing images loaded"""
    test_loader, _ = load_data(train=False)
    assert len(test_loader.dataset) == 10000

def test_test_image_shape():
    """Test shape of testing images. Expect (1,28,28)"""
    test_loader, _ = load_data(train=False, ds_start=0, ds_end=2)
    assert test_loader.dataset[0][0].shape == torch.Size([1,28,28])


### Module Tests ###
def _eval_module_shape(net):
    """Generic function to evaluate shape of module

    Args:
        net (network): Network being tested
    """
    # Evaluate on a random dataset, length 10
    x = torch.randn(10, 1, 28, 28)
    eval_output = net(x)    
    # Check against assumption, expect 10 results, 10 indices
    assert eval_output.shape == torch.Size([10,10])


def test_module_shapes():
    """Tests all defined modules"""
    from models import network
    to_test = [
        network("CNN", p=0.2),
        network("CNNWeak", p=0.2),
        network("CNNDumb", p=0.2),
        network("ReLu_Linear", p=0.2),
        network("sigmoid", p=0.2),
        network("Logistic", p=0.2),
        network("RNN", p=0.2),
        network("BNReLu", p=0.2)
    ]
    for m in to_test:
        _eval_module_shape(m)


### Utils Tests ###
def test_gaussian_noise():
    """Tests gaussian noise is as-expected"""
    from utils import AddGaussianNoise
    # Define Gaussian Noise, apply to large tensor of ones
    gn = AddGaussianNoise(mean=0.0, std=0.2)
    test_tensor = gn(torch.ones(10000))
    # Must equal 1.0 to at least one decimal place
    assert round(test_tensor.mean().item(), 1) == 1.0

def test_angle_rotation():
    """Check rotation of a tensor of 0's is still 0"""
    from utils import GivenAngleRotation
    from torchvision import transforms
    # Create AngleRotation, apply to some tensor of zeros
    rot = GivenAngleRotation(angle=30)
    test_tensor = rot(transforms.ToPILImage()(torch.zeros(64,64)))
    rotated_tensor = transforms.ToTensor()(test_tensor)
    # Rotated zeros should still be zero 
    # (could use a better check here)
    assert rotated_tensor.mean() == 0.0

def test_get_transform_type():
    """Ensure output of get_transform is correct type"""
    from utils import get_transform
    transforms = get_transform()
    assert type(transforms) == torchvision.transforms.Compose

### Train Tests
def test_model_type(sample_model):
    assert type(sample_model.clf) == network
