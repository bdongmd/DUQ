import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from iminuit import Minuit

def gau(x, a, b, c):
	return a*np.exp(-0.5*(x-b)**2/c**2)
	#return (a/(c*np.sqrt(2.*np.pi)))*np.exp(-0.5*((x-b)/(c))**2)
def loglikelihoodg(a,b,c):
	#return n*(np.log(2*np.pi))+2*np.sum((y-gau(x,a,b,c))**2/(2*s**2))
	return np.sum((gau(x,a,b,c)-y)**2)

f = h5py.File('output/TrainingDr0p2ep500_test_acc_dr0p2.h5', 'r')
acc = f['test_acc'][()]
f.close()

'''
# unbinned fitting
def gau(x, mu, sigma):
	return np.exp(-0.5*(x-mu)**2/sigma**2)/(np.sqrt(2*np.pi)*sigma)

#plt.hist(acc, bins=50, histtype='step')
ulh = UnbinnedLH(gau, acc)
m = Minuit(ulh, mu=99.1, sigma=0.15)
m.migrad()
ulh.show(m)
'''

# binned fitting
xmin = 98.8
xmax = 99.5
nbins = 70
fig = plt.figure()
ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
y, x_border, patches = plt.hist(acc, bins=nbins, range=(xmin, xmax), facecolor='blue')
x = x_border[:-1]+np.diff(x_border)/2.
n=len(x)

#xLow = np.linspace(xmin, xmax, nbins)
m = Minuit(loglikelihoodg, a=100, b=99.1, c=0.05)
m.migrad()
#m.hesse()
#m.minos()

plt.plot(x, gau(x, *m.values.values()))
plt.legend(['fit'], loc='upper right', fontsize=14)
plt.text(98.8, 200, 'mean = {:.3f}\nsigma = {:.3f}'.format(m.values.values()[1], m.values.values()[2]), fontsize=12, horizontalalignment='left', verticalalignment='top')
plt.xlabel('accuracy(%)')
plt.ylabel('counts')
#plt.show()
plt.savefig('plots/acc_dis/acc_dis_2L0p2_ep500.pdf')
