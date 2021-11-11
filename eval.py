from modules import network
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import h5py
import sys
from time import time

### evaluation on the whole set of samples

trainmodel = 'CNN'
evalTime = 3000 
dropout_rate = float(sys.argv[1])*0.1
trainModule = "dr0p2_ep5_ev3k"

# load test dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
test_dataset = datasets.MNIST('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# load trained model
model = network(
    model_name=trainmodel,
    p=dropout_rate
    )

model.load_state_dict(torch.load('../output/trainModule/mnist_cnn_{}.pt'.format(trainModule)))
print(model)

time0 = time()
loss = []
accuracy = []
for i in range(evalTime):
	if i%100 == 0:
		print('{} evaluation time: {} minutes.'.format(i, (time()-time0)/60.0))
	test_loss = 0
	correct = 0
	model.eval()
	#with torch.no_grad():
	for data, target in test_loader:
		if trainmode == 'dropout':
			model.train()
		data, target = Variable(data), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(output, target, reduction='sum').item()
		pred = output.argmax(dim=1, keepdim=True) # get the index of max log-probability
		correct += pred.eq(target.view_as(pred)).sum()
	loss.append(test_loss/len(test_loader.dataset))
	accuracy.append(100.*correct/len(test_loader.dataset))
	print('loss: {}, accuracy: {}'.format(test_loss/len(test_loader.dataset), 100.*correct/len(test_loader.dataset)))

hf = h5py.File('../output/testResult/Train{}_Test_acc_dr0p{}.h5'.format(trainModule, sys.argv[1]), 'w')
hf.create_dataset('test_loss', data=loss)
hf.create_dataset('test_acc', data=accuracy)
hf.close()
