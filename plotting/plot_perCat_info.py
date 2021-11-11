import h5py
from scipy import stats
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import sys

#cali_output = 'CNNWeak_dr0p2_ep100_ev3000_image1_5000'
'''
cali_output = 'CNNWeak_dr0p2_ep300_ev10000_image1_5000'
f = h5py.File('../../output/testResult/uncertainty/cali_{}.h5'.format(cali_output), 'r')
image_true_mu = f['image_true_mu'][()]
image_p_true_noDropout = f['image_acc_noDropout'][()]
image_true_std = f['image_true_std'][()]
image_target = f['image_target'][()]
image_acc = f['image_acc'][()]
image_probability = f['image_probability'][()]
'''
cali_output = 'CNNWeak_dr0p4_ep300_ev10000_image1_5000'
f = h5py.File('../../output/testResult/uncertainty/cali_{}.h5'.format(cali_output), 'r')
image_true_mu = f['image_true_mu'][()]
image_p_true_noDropout = f['NoDro_image_true_probs'][()]
image_true_std = f['true_image_std'][()]
image_target = f['image_target'][()]
image_acc = f['image_acc'][()]
image_probability = f['image_probability'][()]

### save plots to pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("../../output/plots/caliplot_{}.pdf".format(cali_output))
'''
fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.plot(image_probability, image_acc - image_probability, 'o')
ax.set_xlabel('Calculated Probability')
ax.set_ylabel('Calculated Probability - Image Accuracy')
pdf.savefig()
fig.clear()
plt.close(fig)
'''

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist((image_probability-image_true_mu)/image_true_std, bins=1000, range=[-10, 10], color='blue',density=True, alpha=0.7)
ax.set_ylabel("Frequency")
ax.set_xlabel("pull3")
pdf.savefig()
fig.clear()
plt.close(fig)


fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist(image_p_true_noDropout - image_probability, bins=200, range=[-1, 1], color='blue',density=True, alpha=0.7)
ax.set_ylabel("Frequency")
ax.set_xlabel("image_p_true_noDropout - image_probability")
pdf.savefig()
fig.clear()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist(image_probability - image_acc, bins=200, range=[-0.5, 0.5], color='blue',density=True, alpha=0.7)
ax.set_ylabel("Frequency")
ax.set_xlabel("Calculated Probability - Image Accuracy")
pdf.savefig()
fig.clear()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
#loc,sca=stats.norm.fit(image_p_true_noDropout - image_true_mu)
ax.hist(image_p_true_noDropout - image_true_mu, bins=100, range=[-1, 1], color='blue',density=True, alpha=0.7)
#ax.plot(np.linspace(-1,1,100),stats.norm.pdf(np.linspace(-1,1,100), loc=loc,scale=sca), linewidth=2,color='green',label="Gaussian Fit loc: %3.2f scale: %3.2f"%(loc,sca))
ax.set_ylabel("Frequency")
#ax.legend()
ax.set_xlabel("image_p_true_noDropout - image_true_mu")
plt.close(fig)

image_pull = (image_p_true_noDropout - image_true_mu)/image_true_std
fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist(image_pull, bins=200, range=[-5, 5], color='blue',density=True, alpha=0.7)
loc,sca=stats.norm.fit(image_pull)
ax.plot(np.linspace(-5,5,200),stats.norm.pdf(np.linspace(-5,5,200), loc=loc,scale=sca), linewidth=2,color='green',label="Gaussian Fit loc: %3.2f scale: %3.2f"%(loc,sca))
ax.set_ylabel("Probability Density")
ax.set_xlabel("pull")
ax.legend()
pdf.savefig()
fig.clear()
plt.close(fig)

image_pull2 = (image_p_true_noDropout - image_probability)/image_true_std
fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist(image_pull2, bins=200, range=[-5, 5], color='blue',density=True, alpha=0.7)
loc,sca=stats.norm.fit(image_pull2)
ax.plot(np.linspace(-5,5,200),stats.norm.pdf(np.linspace(-5,5,200), loc=loc,scale=sca), linewidth=2,color='green',label="Gaussian Fit loc: %3.2f scale: %3.2f"%(loc,sca))
ax.set_ylabel("Probability Density")
ax.set_xlabel("pull2")
ax.legend()
pdf.savefig()
fig.clear()
plt.close(fig)

for i in range(10):
    idx = image_target == i
    tmp_image_acc = image_acc[idx]
    tmp_diff = image_probability[idx] - image_acc[idx]
    tmp_std = image_true_mu[idx]
    tmp_pull = (image_p_true_noDropout[idx] - image_true_mu[idx])/image_true_std[idx]
    tmp_pull2 = (image_p_true_noDropout[idx] - image_probability[idx]) / image_true_std[idx]
    #tmp_pull3 = (image_acc_noDropout[i] - image_probability[idx]) / image_true_std[idx]

    fig, ax = plt.subplots(figsize=(10,10), ncols=2, nrows=3)
    ax[0,0].hist(tmp_image_acc, bins=100, range=[-1,1],alpha=0.7, label="image {}".format(i), density=True)
    ax[0,0].set_xlabel("Image Accuracy")
    ax[0,0].set_ylabel("Frequency")
    ax[0,0].legend()
    ax[1,0].hist(tmp_std,bins=100,range=[0,1], alpha=0.7, label='image {}'.format(i), density=True)
    ax[1,0].set_xlabel("Image Uncertainty")
    ax[1,0].set_ylabel("Frequency")
    ax[1,0].legend()
    ax[0,1].hist(tmp_diff, bins=100, range=[-0.5, 0.5], alpha=0.7, label="image {}".format(i), density=True)
    ax[0,1].set_xlabel("Calculated Probability - Observed Probaility")
    ax[0,1].set_ylabel("Frequency")
    ax[0,1].legend()
    ax[1,1].hist(tmp_pull, bins=500, range=[-5, 5], alpha=0.7, label='image {}'.format(i), density=True)
    ax[1,1].set_xlabel("Pull")
    ax[1,1].set_ylabel("Frequency")
    ax[1,1].legend()
    ax[2,0].hist(image_p_true_noDropout - image_true_mu, bins=100, range=[-1, 1], alpha=0.7, label='image {}'.format(i), density=True)
    ax[2,0].set_xlabel("image_p_true_noDropout - image_true_mu")
    ax[2,0].set_ylabel("Frequency")
    ax[2,0].legend()
    ax[2,1].hist(tmp_pull2, bins=500, range=[-5, 5], alpha=0.7, label='image {}'.format(i), density=True)
    ax[2,1].set_xlabel("Pull - 2")
    ax[2,1].set_ylabel("Frequency")
    ax[2,1].legend()
    pdf.savefig()
    fig.clear()
    plt.close(fig)

pdf.close()
