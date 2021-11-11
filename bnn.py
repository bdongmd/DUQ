"""Creat Bayesian Neural Network"""
from typing import Union
from numpy.core.fromnumeric import shape
from pyro.primitives import deterministic
import torch
import torch.nn.functional as F
from torch import nn

import h5py

from utils import load_data, load_configuration_file

import pyro
from pyro.distributions import Normal, Categorical, Uniform
from pyro.nn.module import PyroSample, PyroModule
from pyro.infer import Predictive

from pyro.infer.autoguide import AutoDiagonalNormal
from scipy import stats
import numpy as np
from modules import network
import sys
from collections import OrderedDict
import arviz as az
import matplotlib.pyplot as plt

import os
if sys.platform == "darwin":
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

## pytroch random seed seeting 
torch.manual_seed(0)

#########################
### Define models #######
#########################

class BCNNWeak(PyroModule):
    def __init__(self, p: float, prior_std: float = 1.0) -> None:
        """Bayesian version of CNNWeak

        Args:
            p (float): dropout rate
            prior_std (float, optional): standard deviation for *all* priors. Defaults to 1.0.
        """
        super().__init__()

        prior = Normal(0, prior_std)

        self.conv1 = PyroModule[nn.Conv2d](1, 1, kernel_size=11)
        self.conv1.weight = PyroSample(prior.expand([1, 1, 11, 11]).to_event(2))
        self.conv1.bias = PyroSample(prior.expand([1]).to_event(1))

        self.dropout = nn.Dropout(p)
        
        self.fc1 = PyroModule[torch.nn.Linear](1*9*9, 25)
        self.fc1.weight = PyroSample(prior.expand([25, 1*9*9]).to_event(2))
        self.fc1.bias = PyroSample(prior.expand([25]).to_event(1))

        self.fc2 = PyroModule[torch.nn.Linear](25, 10)
        self.fc2.weight = PyroSample(prior.expand([10, 25]).to_event(2))
        self.fc2.bias = PyroSample(prior.expand([10]).to_event(1))


                
    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> float:
        """Define a forward pass through network
        For pyro network using SVI, forward is envoked in svi.step 


        Args:
            x (torch.Tensor): Input data
            y (torch.Tensor, optional): True class. Defaults to None.

        Returns:
            [type]: [description]
        """

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        with pyro.plate("data", size=x.shape[0]):
            # Probs approach
            probs = deterministic("probs", torch.exp(F.log_softmax(x, dim=1)))
            obs = pyro.sample("obs", Categorical(probs=probs), obs=y)

        return probs


#########################
### Plotting Utils ######
#########################

@torch.no_grad()
def plot_param_samples(model: Union[BCNNWeak, pyro.infer.autoguide.guides.AutoDiagonalNormal], nsamples) -> None:
    """Plot the weight and bias for one parameter set per layer

    Args:
        model (Union[BCNNWeak, pyro.infer.autoguide.guides.AutoDiagonalNormal]): Either a pyro model (prior) or guide (posterior)
        nsamples ([type]): number of samples to draw
    """

    # For prior predictive
    if type(model) == BCNNWeak:
        layer_dict ={
            "fc1" : model.fc1,
            "fc2" : model.fc2
        }
        def weight_accessor(layer):
            return layer_dict[layer].weight
        def bias_accessor(layer):
            return layer_dict[layer].bias
    # For posterior predictive
    elif type(model) == pyro.infer.autoguide.guides.AutoDiagonalNormal:
        def weight_accessor(layer):
            return model.forward()[f"{layer}.weight"]
        def bias_accessor(layer):
            return model.forward()[f"{layer}.bias"]

    layers = ["fc1","fc2"]

    # setup plot
    nlayers = len(layers)
    fig, ax_enum = plt.subplots(nrows=nlayers, ncols=2)

    # plot 1 posterior for all layers
    # sample weights and biases (for all nodes)
    weights = [[],[]]
    biases = [[],[]]

    for i, layer in enumerate(layers):

        for _ in range(nsamples):
            weights[i].append(torch.flatten(weight_accessor(layer)).numpy())
            biases[i].append(torch.flatten(bias_accessor(layer)).numpy())

        # Plot weight
        plt.sca(ax_enum[i,0])
        az.plot_kde(weights[i],ax=ax_enum[i,0])
        ax_enum[i,0].get_yaxis().set_visible(False)
        ax_enum[i,0].set_title("all nodes - weight")

        # Plot bias
        plt.sca(ax_enum[i,1])
        az.plot_kde(biases[i], ax=ax_enum[i,1])
        ax_enum[i,1].get_yaxis().set_visible(False)
        ax_enum[i,1].set_title("all nodes - bias")
    plt.tight_layout()
    return(weights,biases)


def plot_loss(train_loss: np.array, test_loss: np.array) -> None:

    fig = plt.figure(figsize=(8,8))
    marker_size = 4
    # Training Loss    
    plt.plot(
        np.arange(1, len(train_loss)+1),
        train_loss,
        markersize=marker_size,
        color="orange",
        marker="o",
        label="Training Loss",
        alpha=0.8    
        )

    plt.plot(
        np.arange(1, len(test_loss)+1),
        test_loss,
        markersize=marker_size,
        color="dodgerblue",
        marker="o",
        label="Testing Loss",
        alpha=0.8
        )
    plt.yscale("log")
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.legend(frameon=False)


if __name__ == "__main__":

    cfg = load_configuration_file(sys.argv[1])
    EPOCHS=cfg["epochs"]

    if not os.path.exists(f"{cfg['plotDirectory']}/train/"):
        os.makedirs(f"{cfg['plotDirectory']}/train/")

    train_loader, _ = load_data(
        ds_start=0, ds_end=60000, train=True,
        useGPU=cfg["useGPU"], b_size=cfg["batch_size"], exclude_number=cfg["trainExcludeOneNumber"]
        )

    test_loader, _ = load_data(
        ds_start=0, ds_end=10000, train=False,
        useGPU=cfg["useGPU"], b_size=cfg["batch_size"], exclude_number=cfg["testExcludeOneNumber"]
        )
    
    pyro.enable_validation(True)
    pyro.clear_param_store()
    # Create Bayesian Convolutional Neural Net
    model = BCNNWeak(p=0.2, prior_std=1)
    
    # Create Guide - approximation to posterior
    # A parameterized distribution of variational parameters
    guide = AutoDiagonalNormal(model)

    # Create SVI object
    # Use trace elbo as loss - "Evidence lower bound"
    # Maximizing elbo minimizes KL divergence
    svi = pyro.infer.SVI(
        model,
        guide,
        optim=pyro.optim.ClippedAdam({'lr':1e-2}),
        loss=pyro.infer.Trace_ELBO()
    )

    # Plot prior distributions
    print("Plotting prior samples")
    plot_param_samples(model, 5000)
    plt.savefig(f"{cfg['plotDirectory']}/train/prior_predictive.png")

    
    epoch_loss = np.zeros(EPOCHS)
    test_loss = np.zeros(EPOCHS)
    for k in range(len(epoch_loss)):
        if k%5 ==0: print(f"epoch {k}")
        loss = 0
        
        # Do training step
        for x, y in train_loader:
            # SVI.step -> take a single gradient step, return loss estimate
            loss += svi.step(x, y)
        epoch_loss[k] = loss
        
        # Do testing evaluation
        loss = 0
        for x, y in test_loader:
            # SVI.evaluate_loss -> get loss estimate without gradient step
            loss += svi.evaluate_loss(x, y)
        test_loss[k] = loss

    plot_loss(train_loss=epoch_loss/len(train_loader.dataset), test_loss=test_loss/len(test_loader.dataset))
    plt.savefig(f"{cfg['plotDirectory']}/train/bnn_loss.png")

    # Create Predictive model
    predictive = Predictive(
        model, guide=guide, num_samples=cfg["samples"])
    print(f"Final Loss {epoch_loss[-1]}")

    # Plot posterior distributions
    allweights, allbiases = plot_param_samples(guide, 10000)
    hf = h5py.File(f"{cfg['plotDirectory']}/train/posterior_predictive.h5", 'w')
    hf.create_dataset('train_loss', data = epoch_loss/len(train_loader.dataset))
    hf.create_dataset('test_loss', data = test_loss/len(test_loader.dataset))
    hf.create_dataset('bias_layer0',data = allbiases[0])
    hf.create_dataset('bias_layer1',data = allbiases[1])
    hf.close()
    plt.savefig(f"{cfg['plotDirectory']}/train/posterior_predictive.png")


    # Save the bcnn
    torch.save({
        cfg["model_name"]: predictive,
        cfg["state_dict"]: predictive.state_dict(),
        #"bcnn_guide": guide, # TODO: Drop if not useful in the future
        #"bcnn_svi" : svi,
        #"bcnn_model": model
    }, cfg["model_file"])


    # We'll get majority logic accuracy
    correct_arr = []
    y_hat = np.zeros(shape=(len(test_loader.dataset)))
    y_true = np.zeros(shape=(len(test_loader.dataset)))
    print("Starting to evaluate majority logic accuracy (abort if this isn't useful to you.)")
    for i, (x, y) in enumerate(test_loader):
        print(f"Progress - {round(100*i/len(test_loader))}% done")
        preds = predictive(x)
        max_indices = torch.sort(torch.exp(preds["probs"]), dim=2, descending=True).indices[:,:,0]        

        block_start = i * test_loader.batch_size
        block_end = block_start + len(y)
        y_hat[block_start:block_end] = stats.mode(max_indices, 0).mode[0,:]
        y_true[block_start:block_end] = y

    acc = sum(np.array(y_hat) == np.array(y_true)) / len(y_true)
    print(f"Majority logic accuracy = {acc}")    


