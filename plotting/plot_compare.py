import h5py
import numpy as np
import matplotlib.pyplot as plt

epochs = 500
filename1 = 'dr0p0_funcdr0'
label1 = 'w/o dropout'
filename2 = 'dr0p2_funcdr0'
label2 = 'w/ dropout, p=0.2'
filename3=False
label3=False

comVar = 'train_loss' # choose between 'train_loss', 'test_loss', 'test_acc'
if comVar == 'test_loss':
    eventsVar = 'test_events'
    ylabel = 'loss'
    labelLocation = 'upper'
    saveFile = 'loss'
    textlabel = 'test'
    y = 2500
    x = 100000
elif comVar == 'train_loss':
    eventsVar = "train_events"
    ylabel = 'loss'
    labelLocation = 'upper'
    saveFile = 'loss'
    textlabel = 'train'
    y = 0.3
    x = 200000
elif comVar == 'test_acc':
    eventsVar = 'test_events'
    ylabel = 'accuracy(%)'
    labelLocation = 'lower'
    saveFile = 'acc'
    textlabel = 'test'
    y = 99
    x = 100000

f1 = h5py.File('../output/trainResult/loss_{}_ep{}.h5'.format(filename1, epochs), 'r')
Var1  = f1['{}'.format(comVar)][()]
events = f1['{}'.format(eventsVar)][()]
print(Var1)
print(events)
f1.close()

f2 = h5py.File('../output/trainResult/loss_{}_ep{}.h5'.format(filename2, epochs), 'r')
Var2  = f2['{}'.format(comVar)][()]
f2.close()

if filename3:
    f3 = h5py.File('../output/trainResult/loss_{}_ep{}.h5'.format(filename3, epochs), 'r')
    Var3  = f3['{}'.format(comVar)][()]
    f3.close()

fig = plt.figure()
ax1 = fig.add_axes([0.15, 0.4, 0.8, 0.5])
if textlabel == 'train':
    plt.plot(events[::15], Var1[::15], color='black')
    plt.plot(events[::15], Var2[::15], color='blue')
    if filename3:
        plt.plot(events[0:0:10], Var3[::10], color='red')
else:
    plt.plot(events, Var1, color='black')
    plt.plot(events, Var2, color='blue')
    if filename3:
        plt.plot(events, Var3, color='red')

if filename3:
    plt.legend([label1,label2,label3], loc='{} right'.format(labelLocation))
else:
    plt.legend([label1,label2 ], loc='{} right'.format(labelLocation))

plt.xscale('log')
if comVar == 'test_acc': 
    plt.ylim(91,100)
    plt.yticks(np.arange(91,100,1.))
    plt.grid(True, axis='both', linestyle='--')
else:
    plt.yscale('log')
plt.ylabel('{}'.format(ylabel))
ax1.text(x, y , '{} results'.format(textlabel), fontsize = 14)
ax1 = fig.add_axes([0.15, 0.1, 0.8, 0.3])
plt.plot(events, Var2/Var1, color='blue')
if filename3:
    plt.plot(events, Var3/Var1, color='red')
plt.xlabel('trained events')
plt.xscale('log')
#plt.grid(True, linestyle='--')
plt.ylabel('other/black')
#plt.show()
plt.savefig('../output/plots/trainResultCompare/compare_{}_epoch{}_dr0vs0p2_{}.pdf'.format(saveFile, epochs, textlabel))
