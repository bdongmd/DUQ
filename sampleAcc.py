import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

### sample accuracy with dropout enabled
def F_sample_acc_with_dropout(model, sample_loader, evalTime, device):
    correct = 0
    for i in range(evalTime):
        for data, target in sample_loader:
            model.train()
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().cpu()
    print('Observed accuracy with dropout = {}'.format(float(correct)/(len(sample_loader.dataset) * evalTime)))
    return(float(correct)/(len(sample_loader.dataset) * evalTime))

### sample accuracy with dropout disabled
def F_sample_acc_without_dropout(model, sample_loader, device):
    correct = 0
    model.eval()
    for data, target in sample_loader:
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().cpu()
    print('Observed accuracy without dropout = {}'.format(float(correct)/len(sample_loader.dataset)))
    return(float(correct)/len(sample_loader.dataset))
