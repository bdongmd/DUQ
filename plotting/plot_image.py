import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from modules import network

# load test dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),])
test_loader = torch.utils.data.DataLoader(datasets.MNIST('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', download=True, train=False, transform=transform), batch_size=64, shuffle=True)

# load trained model
model = network("CNN", p=0.2)
model.load_state_dict(torch.load('../output/trainModule/mnist_cnn_2Ldrop2_funcdr0_ep500.pt'))
print(model)

model.eval()
# plotting
nomatch_target= []
nomatch_pred = []
test_data_nomatch = []
match_target= []
match_pred = []
test_data_match = []
with torch.no_grad():
	for batch_id, (test_data, test_target) in enumerate(test_loader):
		data, target = Variable(test_data), Variable(test_target)
		output = model(data)
		pred = output.argmax(dim=1, keepdim=True)
		for i in range(len(target)):
			if target[i].item() == pred[i].item():
				test_data_match.append(data[:][:][:][i])
				match_target.append(target[i])
				match_pred.append(pred[i].item())
			else:
				test_data_nomatch.append(data[:][:][:][i])
				nomatch_target.append(target[i])
				nomatch_pred.append(pred[i].item())

for j in range(7):
	fig = plt.figure()
	for i in range(9):
		plt.subplot(3,3,i+1)
		plt.axis('off')
		plt.imshow(test_data_nomatch[9*j+i][0], cmap = 'gray', interpolation='none')
		plt.title('truth: {}, labeled: {}'.format(nomatch_target[9*j+i], nomatch_pred[9*j+i]))
	#plt.show()
	plt.savefig('../output/plots/image/labeled_label_notmatch{}.pdf'.format(j+1))
