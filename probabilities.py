import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


## get the probabilities of the image, dropout enabled
## maximum == maximum of the probability
## second == second of the probability
## true_max == probabilities in true category
## false_max == maximum probabilities fall into the wrong catetory
def F_probabilities(model, image_loader, device, target_range = 10, allClass = False, debug = False):
    maximum = 0.
    second = 0.
    true_max = 0.
    false_max = 0.
    prob_allclass = []
    numbers = list(range(0,10))

    tmp_false = []

    model.train()
    with torch.no_grad():
        for data, target in image_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data).cpu()
            for i in range(target_range):
                probi = torch.sort(np.exp(output[:, i]))[0]
                if debug:
                    print("i = {}, prob = {}".format(i, probi.item()))
                prob_allclass.append(probi.item())
                if i == target[0].item():
                    true_max = torch.sort(np.exp(output[:, i]))[0].item()
                else:
                    tmp_false_max = torch.sort(np.exp(output[:, i]))[0]
                    tmp_false.append(tmp_false_max.item())
    false_max = max(tmp_false)
    tmp_pro = sorted(prob_allclass)
    maximum = tmp_pro[-1]
    second = tmp_pro[-2]
    if debug:
        print("max = {}, second = {}, true_max = {}, false_max = {}".format(maximum, second, true_max, false_max))
    if allClass:
        return(maximum, second, true_max, false_max, prob_allclass)
    else:
        return(maximum, second, true_max, false_max)

## without dropout enabled !!!get the target probabities and the maximum of other categories' probabilities
## run images one by one
def F_true_false_probabilities_noDropout(model, image_loader, device, target_range = 10):
    tmp_false = []
    image_acc  = 0.
    true_maximum = 0.
    model.eval()
    with torch.no_grad():
        for data, target in image_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data).cpu()
            for j in range(target_range):
                if j == target[0].item():
                    true_maximum =  torch.sort(np.exp(output[:, j]))[0].item()
                else:
                    tmp_false_maximum =  torch.sort(np.exp(output[:, j]))[0]
                    tmp_false.append(tmp_false_maximum.item())
    if true_maximum > max(tmp_false):
        image_acc = 1.
    else:
        image_acc = 0.
    return(true_maximum, max(tmp_false), image_acc, target[0].item())
