#### this is a file to test how much evaluation we needed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import h5py
import sys
sys.path.append('../')
import significance
import probabilities
import accuracy
from modules import network

ds_start = int(sys.argv[1]) 
ds_end = int(sys.argv[2])
trainModule = "dr0p2_ep7_ev3k"
nbins = 200
evalTime = 5000
ifPlotacc = True # plot evalTime vs accuracy plot
ifPlotprb = True # plot max/second max probabilities histograms
ifPlotcumu = True # plot max/second max probabilities cumulative curve

####
def histogram_overlap(h1, h2, nbins):
    bin_width = 1./nbins
    overlap = 0.
    for i in range(nbins):
        overlap += min(h1[i],h2[i])*bin_width
    return(overlap)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
testset = datasets.MNIST('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', download=True, train=False, transform=transform)
test_image = torch.utils.data.Subset(testset, list(range(ds_start, ds_end)))
image_loader = torch.utils.data.DataLoader(test_image, batch_size=1, shuffle=True)

model = network("CNN", p=0.2)
model.load_state_dict(torch.load('../output/trainModule/mnist_cnn_{}.pt'.format(trainModule)))
print(model)

#### plot image
dataiter = iter(image_loader)
images, labels = dataiter.next()
plt.imshow(images[0].numpy().squeeze(),cmap='gray_r')
#plt.show()
plt.savefig('../output/plots/image_acc_test/image{}/image.pdf'.format(ds_end))

if ifPlotacc:
    iTime = []
    acc = []
    correct = 0
    tmp_acc = 0.

    for i in range(1, 10000+1):
        for data, target in image_loader:
            model.train()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum()
            tmp_acc = 100*float(correct)/float(i)
            #print('i = {}, correct = {}, acc = {}'.format(i, correct, tmp_acc))
            iTime.append(i)
            acc.append(tmp_acc)

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    ax.get_yaxis().get_major_formatter().set_useOffset(False)
    plt.plot(iTime, acc, color='blue')
    plt.xlabel('# of evaluation times')
    plt.ylabel('image accuracy (%)')
    plt.xscale('log')
    plt.savefig('../output/plots/image_acc_test/image{}/accuracy.pdf'.format(ds_end))
    #plt.show()

image_acc_method1 = accuracy.F_image_ave_acc(model, image_loader, evalTime)

true_maximum, false_maximum = probabilities.F_true_false_probabilities(model, image_loader, evalTime)
nodp_true_maximum, nodp_false_maximum = probabilities.F_true_false_probabilities_noDropout(model, image_loader)

hist_true_max, bin_edges_true = np.histogram(true_maximum, bins=nbins, range=(0,1))
hist_false_max, bin_edges_false = np.histogram(false_maximum, bins=nbins, range=(0,1))
overlap = histogram_overlap(hist_true_max, hist_false_max, nbins)/(1./nbins*evalTime)

if ifPlotprb:
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    ax.text(0.1, 0.95, 'overlap = {}%'.format(overlap*100), transform=ax.transAxes)
    plt.plot(nodp_true_maximum[0], 20, 'ro', c='#1f77b4')
    plt.plot(nodp_false_maximum[0], 20, 'ro', color='darkorange')
    plt.hist(true_maximum, bins=nbins, range=(0,1), histtype='step', density=True)
    plt.hist(false_maximum, bins=nbins, range=(0,1), histtype='step', density=True)
    plt.legend(['w/o dropout - correct', 'w/o dropout - wrong', 'w/ dropout - correct', 'w/ dropout - wrong'], loc='upper right')
    plt.xlabel('probability')
    plt.ylabel('P.D.F')
    #plt.show()
    plt.savefig('../output/plots/image_acc_test/image{}/pdf.pdf'.format(ds_end))

if ifPlotcumu:
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    hist1, bin1, patches1 = plt.hist(true_maximum, bins=nbins, range=(0,1), histtype='step',density=True, cumulative=True)
    hist2, bin2, patches2 = plt.hist(false_maximum, bins=nbins, range=(0,1), histtype='step',density=True, cumulative=True)
    plt.legend(['w/ dropout - correct', 'w/ dropout - wrong'], loc='upper right')
    plt.xlabel('probability')
    plt.savefig('../output/plots/image_acc_test/image{}/prob_cumu.pdf'.format(ds_end))

print(histogram_overlap(hist1, hist2, nbins))

print('image_acc = {}'.format(image_acc_method1))
print('overlap = {}'.format(overlap))
