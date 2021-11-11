import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

#f = h5py.File('../../output/forPaper/trained_60k_nonrotated/trainResult/mnist_CNNWeak_dr0p2_ep100_ev60000.h5', 'r')
f = h5py.File(sys.argv[1], 'r')
train_loss = f['train_loss_history'][()]
test_loss = f['test_loss_ave_history'][()]
train_acc = f['train_acc_history'][()]
test_acc = f['test_acc_history'][()]
train_events = f['train_events'][()]
test_events = f['test_events'][()]
f.close()

fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
plt.plot(train_events, train_loss, color='blue')
plt.plot(train_events, test_loss, color='red')
plt.legend(['train', 'test'], loc='upper right')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.yscale('log')
plt.xscale('log')
plt.show()
#plt.savefig('plots/loss_{}_epoch{}.pdf'.format(filename, epochs))

fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
plt.yscale('log')
plt.yticks(np.arange(91,102,step=1))
#plt.ylim(bottom=90)
#plt.ylim(top=105)
plt.plot(train_events, train_acc*100, color='blue')
plt.plot(train_events, test_acc*100, color='red')
plt.legend(['train', 'test'], loc='upper right')
plt.xlabel('trained events')
plt.ylabel('accuracy(%)')
plt.xscale('log')
plt.show()
#plt.savefig('plots/acc_{}_epoch{}_log.pdf'.format(filename, epochs))

