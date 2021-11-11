import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss
from torch.optim.optimizer import Optimizer
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from time import time
from modules import network
import sys
from utils import load_data, get_transform
from typing import NamedTuple, Optional, Union


debug = False
tryGPU = False
b_size = 64 #batch size
lr_opt = 0.001
momentum_opt = 0.9
testGaussianNoise = False 
GaussStd = 1.0
EXCLUDE_NUMBER = False 
RotateImageRandom = False
RotateImageSpecific = False
RotateAngle = (0, 360) # for random rotation, put an angle range. For fixed, put a number.
useGPU = (tryGPU and torch.cuda.is_available())
print("Running on {}.".format("GPU" if useGPU else "CPU" ))
DEVICE = torch.device("cuda" if useGPU else "cpu")

class Records:
    """Records progress and events over training procedure"""
    def __init__(self) -> None:
        """Initalize object"""
        self.train_events = [] # trained events
        self.test_events = [] # corresponding trained events
        self.train_loss_history = [] # train loss for each epoch
        self.test_loss_history = [] # test loss for each epoch
        self.test_loss_ave_history = [] # average test loss for each epoch
        self.test_acc_history = [] # acc for each epoch
        self.train_acc_history = []

    def training_update(self, train_events: list, train_acc: float, train_loss: list) -> None:
        """Update object per training epoch

        Args:
            train_events (list): Events trained over so far
            train_acc (float): Training accuracy at end of epoch
            train_loss (list): Training loss at end of epoch
        """
        self.train_events += train_events
        self.train_loss_history += train_loss
        self.train_acc_history.append(train_acc)

    def testing_update(self, test_events: int, test_acc: float, test_loss: Optional[float] = None, test_loss_ave: Optional[float] = None) -> None:
        """Update object per testing epoch

        Args:
            test_events (int): Events _trained_ over so far
            test_acc (float): Testing accuracy at end of epoch
            test_loss (Optional[float], optional): Testing loss at end of epoch. Defaults to None.
            test_loss_ave (Optional[float], optional): Average testing loss at end of epoch. Defaults to None.
        """
        self.test_events.append(test_events)
        self.test_acc_history.append(test_acc)
        if test_loss:
            self.test_loss_history.append(test_loss)
        if test_loss_ave:
            self.test_loss_ave_history.append(test_loss_ave)

    def save_to_file(self, hf: h5py.File) -> None:
        """Saves progress to an h5py file
        Args:
            hf (h5py.File): Output file
        """
        hf.create_dataset("train_events", data=self.train_events)
        hf.create_dataset("test_events", data=self.test_events)
        hf.create_dataset("train_loss_history", data=self.train_loss_history)
        hf.create_dataset("test_loss_history", data=self.test_loss_history)
        hf.create_dataset("test_loss_ave_history", data=self.test_loss_ave_history)
        hf.create_dataset("test_acc_history", data=self.test_acc_history)
        hf.create_dataset("train_acc_history", data=self.train_acc_history)


class ModelSet(NamedTuple):
    """ NamedTuple for model parameters """
    clf: network
    optimizer: Optimizer
    criterion_train: Union[CrossEntropyLoss, NLLLoss]
    criterion_test: Union[CrossEntropyLoss, NLLLoss]


def debug_data(train_loader: DataLoader) -> None:
    """Generate plot to debug data import

    Args:
        train_loader (DataLoader): training data to plot
    """
    dataiter = iter(train_loader)
    images, labels =  dataiter.next()
    print(images.shape)
    print(labels.shape)
    plt.imshow(images[0].numpy().squeeze(),cmap='gray_r')
    plt.show()
    figure = plt.figure()
    num_of_images = 60
    for index in range(1, num_of_images+1):
        plt.subplot(6, 10, index)
        plt.axis('off')
        plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
        plt.savefig('../output/plots/image/truth_label.pdf')


def get_model(trainmodel: str ="CNNWeak", dropout_rate: float = 0.2, exclude_number: bool = False) -> ModelSet:
    """Gets model from passed string, defines criterion and optimizer

    Args:
        trainmodel (str, optional): Which model defined in modules to run. Defaults to CNNWeak.
        dropout_rate (float, optional): dropout rate used. Defaults to 0.2.
        exclude_number (bool, optional): To exclude a number while training. Defaults to False.

    Returns:
        ModelSet: namedtuple with members clf, optimizer, criterion_train, criterion_test
    """    
    output_layer_size = 9 if exclude_number else 10
    clf = network(
        model_name=trainmodel,
        p=dropout_rate,
        output_layer=output_layer_size
        )        
    if trainmodel == "RNN":
        criterion_train = nn.CrossEntropyLoss()
        criterion_test = nn.CrossEntropyLoss(reduction='sum')
    else:
        criterion_train = nn.NLLLoss()
        criterion_test = nn.NLLLoss(reduction='sum')
    optimizer = optim.SGD(clf.parameters(), lr=lr_opt, momentum=momentum_opt)

    this_model = ModelSet(
        clf=clf,
        optimizer=optimizer,
        criterion_train=criterion_train,
        criterion_test=criterion_test
    )
    return this_model


# define training
def train(model_set: ModelSet, train_loader: DataLoader, records: Records, epoch: int, device: torch.device) -> None:
    """Operations performed in training procedure

    Args:
        model_set (ModelSet): Set of model, criteria, optimizer
        test_loader (DataLoader): test dataset
        records (Records): records to save output to
        epoch (int): current epoch
        device (torch.device): device to run on
    """    
    model_set.clf.train() # set model in training mode (need this because of dropout)
    train_loss = 0
    running_loss = 0
    correct_train = 0
    local_train_loss = []
    local_train_events = []
    for batch_id, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        model_set.optimizer.zero_grad()
        preds = model_set.clf(data)
        if torch.cuda.is_available():
            preds.cpu()
        train_pred = preds.argmax(dim=1, keepdim=True)
        train_loss = model_set.criterion_train(preds, target)
        train_loss.backward()
        model_set.optimizer.step()

        ### training accuracy test
        correct_train += train_pred.eq(target.view_as(train_pred)).sum()
        running_loss += train_loss.item()

        if batch_id % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.4f}%)], Loss: {}'.format(epoch, batch_id * len(data), len(train_loader.dataset), 100. * batch_id * len(data)/ len(train_loader.dataset), train_loss.item()))

    local_train_loss.append(running_loss/len(train_loader))
    local_train_events.append(epoch)
    epoch_accuracy = correct_train.item()/len(train_loader.dataset)
    records.training_update(train_events=local_train_events, train_loss=local_train_loss, train_acc=epoch_accuracy)
    print(f'Train Epoch: {epoch}, Training accuracy: {epoch_accuracy}')

# define testing
def test(model_set: ModelSet, test_loader: DataLoader, records: Records, epoch: int, device: torch.device, exclude_number: bool = False):
    """Operations performed in testing procedure

    Args:
        model_set (ModelSet): Set of model, criteria, optimizer
        test_loader (DataLoader): test dataset
        records (Records): records to save output to
        epoch (int): current epoch
        device (torch.device): device to run on
    """
    model_set.clf.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model_set.clf(data)
            if not exclude_number:
                test_loss += model_set.criterion_test(output, target).item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of max log-probability
            correct += pred.eq(target.view_as(pred)).sum().cpu()

    accuracy = 100.*correct / len(test_loader.dataset)
    events_trained_on = records.train_events[-1]
    if not exclude_number:
        test_loss_ave = test_loss / len(test_loader.dataset)
        records.testing_update(test_events=events_trained_on, test_acc=accuracy, test_loss=test_loss, test_loss_ave=test_loss_ave)
        print('\nTest set: loss: {:.4f}, ave loss:  {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, test_loss_ave, correct, len(test_loader.dataset), accuracy)
            )
    else:
        records.testing_update(test_events=events_trained_on, test_acc=accuracy)
        print(f'\nTest set accuracy: {accuracy:.4f}%\n')


def main():
    # Read args 
    train_model = sys.argv[1]
    dropout_rate = 0.1*int(sys.argv[2])
    epochs = int(sys.argv[3]) 
    ds_start = 0
    ds_end  = int(sys.argv[4])

    # Establish transforms based on input parameters
    gaussian_std = GaussStd if testGaussianNoise else None
    random_rotation = RotateAngle if RotateImageRandom else None
    specific_rotation = RotateAngle if RotateImageSpecific else None
    transforms = get_transform(gaussian_std, random_rotation, specific_rotation)
    
    # Load data using util function
    train_loader, _ = load_data(
        ds_start=ds_start, ds_end=ds_end, train=True,
        useGPU=useGPU, b_size=b_size, exclude_number=EXCLUDE_NUMBER,
        transforms=transforms
        )

    test_loader, _ = load_data(
        ds_start=ds_start, ds_end=10000, train=False,
        useGPU=useGPU, b_size=b_size, exclude_number=False,
        transforms=transforms
        )

    print('train_loader: {}; test_loader: {}'.format(len(train_loader.dataset), len(test_loader.dataset)))

    if debug:
        debug_data(train_loader)

    # Load models
    model_set = get_model(trainmodel=train_model, dropout_rate=dropout_rate, exclude_number=EXCLUDE_NUMBER)
    print(model_set.clf)

    # Instantiate Records to track training
    records = Records()

    # Train the network and do test
    time0 = time()
    for epoch in range(1, epochs+1):
        print('Epoch {}'.format(epoch))
        train(
            model_set=model_set, 
            train_loader=train_loader, 
            records=records, 
            epoch=epoch, 
            device=DEVICE
            )
        print('{} Epoch(s) training time: {} miniutes'.format(epoch, (time()-time0)/60.0))
        test(
            model_set=model_set, 
            test_loader=test_loader,
            records=records,
            epoch=epoch, 
            device=DEVICE,
            exclude_number=EXCLUDE_NUMBER
            )

    # save numbers
    hf = h5py.File('../output/forPaper/trainResult/mnist_{}_dr0p{}_ep{}_ev{}.h5'.format(train_model, int(dropout_rate*10), epochs, ds_end), 'w')
    records.save_to_file(hf)
    hf.close()

    # save the model
    if testGaussianNoise:
        torch.save(model_set.clf.state_dict(), '../output/trainModule/mnist_{}_dr0p{}_ep{}_ev{}_GaussStd{}.pt'.format(train_model, int(dropout_rate*10), epochs, ds_end, GaussStd))
        torch.save(model_set.optimizer.state_dict(), '../output/trainModule/mnist_{}_dr0p{}_ep{}_ev{}_optimizer_GaussStd{}.pt'.format(train_model, int(dropout_rate*10), epochs, ds_end, GaussStd))
    elif EXCLUDE_NUMBER:
        torch.save(model_set.clf.state_dict(), '../output/trainModule/mnist_{}_dr0p{}_ep{}_ev{}_no9.pt'.format(train_model, int(dropout_rate*10), epochs, ds_end))
        torch.save(model_set.optimizer.state_dict(), '../output/trainModule/mnist_{}_dr0p{}_ep{}_ev{}_optimizer_no9.pt'.format(train_model, int(dropout_rate*10), epochs, ds_end))
    else:
        torch.save(model_set.clf.state_dict(), '../output/forPaper/trainModule/mnist_{}_dr0p{}_ep{}_ev{}.pt'.format(train_model, int(dropout_rate*10), epochs, ds_end))
        torch.save(model_set.optimizer.state_dict(), '../output/forPaper/trainModule/mnist_{}_dr0p{}_ep{}_ev{}_optimizer.pt'.format(train_model, int(dropout_rate*10), epochs, ds_end))


if __name__ == "__main__":
    main()
