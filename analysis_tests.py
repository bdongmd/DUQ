"""File to perform analysis comparisons on single posteriors"""
from typing import Dict, Union
from plotting.dropout_evaluations import DropoutEvaluations
import numpy as np
import torch

import matplotlib.pyplot as plt

import h5py
import sys

from scipy import stats

PLOT_DIR="../output/plots/"

def closure_test(true_images: torch.Tensor, pred_prob: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> None:
    # Plot the predicted classed using highest probability 

    integer_bins = np.arange(0,11)
    acc = 0.11

    true_hist, true_binEdges = np.histogram(true_images.numpy(),bins=integer_bins, density=True)
    true_bin_centers = 0.5*(true_binEdges[1:] + true_binEdges[:-1])

    true_hist_err = true_hist * acc
    fig = plt.figure(figsize=(8,8))

    plt.errorbar(
        true_bin_centers,
        true_hist,
        yerr = true_hist_err,
        xerr = 0.5,
        #marker = '.',
        drawstyle = 'steps-mid',
        color="darkorange",
        label="True Class"        
    )


    if type(pred_prob) == torch.Tensor:
        pred_prob = {"Predicted Class": pred_prob}

    for i, (l, preds) in enumerate(pred_prob.items()):

        predicted_class = torch.sort(preds, dim=2, descending=True).indices[:,:,0]
        predicted_class = predicted_class.flatten()

        predicted_class_hist, pred_binEdges = np.histogram(predicted_class.numpy(),bins=integer_bins, density=True)
        pred_bin_centers = 0.5*(pred_binEdges[1:] + pred_binEdges[:-1])

        fr = i/len(pred_prob.items())
        color = plt.cm.brg(fr) if len(pred_prob.items()) == 0 else "dodgerblue"
        plt.errorbar(
            pred_bin_centers,
            predicted_class_hist,
            xerr = 0.5,
            #marker = '.',
            drawstyle = 'steps-mid',
            color=color,
            alpha=0.8,
            label=l
        )

    plt.xlabel("Class")
    plt.ylabel("Density")
    plt.xlim(0,10)
    plt.ylim(bottom=0, top=(1.3 * max(true_hist)))
    plt.gca().set_xticks(integer_bins)
    plt.legend(frameon=False)
        

def h5_to_tensor(dataset: h5py._hl.dataset.Dataset):
    return(torch.tensor(np.array(dataset)))


def calibration_curve(pyro_ppc):

    target_probs =  stats.mode(np.around(pyro_ppc["target_probs"], 2)).mode[0,:]
    not_target_probs = stats.mode(np.around(pyro_ppc["not_target_probs"], 2)).mode[0,:].flatten()
    #pdb.set_trace()
    joint_probs = np.concatenate([target_probs,not_target_probs])
    targs = np.concatenate([np.ones(len(target_probs)), np.zeros(len(not_target_probs))])

    # bin probabilities
    bin_size = 0.10
    thebins = np.arange(0,1,bin_size)
    percent_true = []
    errors = []
    for bin_edge in thebins:
        selector = np.logical_and(joint_probs > bin_edge, joint_probs < bin_edge+bin_size)
        i_probabilities = joint_probs[selector]
        instances = targs[selector]
        target_total = instances.sum()
        instance_total = len(instances)
        percent_true.append(target_total/instance_total)
        errors.append( np.sqrt(target_total) / instance_total )

    plt.bar(
        thebins + bin_size/2,
        percent_true,
        width=bin_size,
        color="firebrick",
        alpha=0.7,
        yerr=errors
    )
    ref = np.arange(0,1,0.01)
    plt.plot(ref,ref, color="grey")
    plt.xlabel("Confidence (Assigned Probability)")
    plt.ylabel("Accuracy (True Target / Total)")
    plt.ylim(0,1)
    plt.title("Modal Estimator")
    plt.savefig(PLOT_DIR+"calibration_bar.png")
    plt.close()

    plt.errorbar(
        thebins+bin_size/2,
        percent_true,
        yerr=errors
        )
    plt.ylim(0,1)
    ref = np.arange(0,1,0.01)
    plt.plot(ref,ref, color="grey")
    plt.savefig(PLOT_DIR+"calibration_scatter.png")



if __name__ == "__main__":
    
    # Load pyro PPC
    pyro_file = h5py.File(sys.argv[1], mode="r")
    pyro_ppc_true = pyro_file["image_true_probs"]

    # Calibration study
    calibration_curve(pyro_file)


    closure_test(h5_to_tensor(pyro_file["image_target"]), h5_to_tensor(pyro_file["all_probabilities"]))
    plt.savefig(f"{PLOT_DIR}/closure-bnn")

    # Load dropout
    dropout_file = DropoutEvaluations(sys.argv[2])
    closure_test(h5_to_tensor(dropout_file.target), h5_to_tensor(dropout_file.all_probs))
    plt.savefig(f"{PLOT_DIR}/closure-do")

    #model_dict = 
    closure_test(
        h5_to_tensor(dropout_file.target), 
        {
            "Dropout" : h5_to_tensor(dropout_file.all_probs),
            "BNN" : h5_to_tensor(pyro_file["all_probabilities"])
        }
        )

    plt.savefig(f"{PLOT_DIR}/closure-multi")

