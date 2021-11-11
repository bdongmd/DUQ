"""Perform analysis on comparisons of posteriors"""
from plotting.eval_plot_library import SAVEDIR
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py

import pdb
from scipy.stats import ks_2samp
from plotting.dropout_evaluations import DropoutEvaluations

from torch.utils import data

PLOT_DIR="../output/plots/"
# todo - generate posteriors dir


def check_size(arrays: list) -> None:
    """Checks on shape of arrays

    Args:
        arrays (list): list of arrays to check

    Raises:
        ValueError: Not enough arrays passed (need at least 1 pair for 2 total)
        ValueError: Dimensionality mismatch along index of number of images
    """


    if len(arrays) < 2:
        raise ValueError(f"Not enough arrays passed. Got {len(arrays)}, expect at least 2")
    
    first_shape = arrays[0].shape[1] # get n_image index
    for i in arrays:
        if i.shape[1] != first_shape:
            raise ValueError(f"Array shape mismatch. First: {first_shape}; Error {i.shape}")


def plot_posteriors(
    pyro_ppc,
    dropout_evaluations: List[DropoutEvaluations], 
    dataset: str,
    num_images: int
    ):
    # only supported for true/false

    # Get Distributions
    pyro_dist = pyro_ppc[dataset]
    pyro_targets = pyro_ppc["image_target"]
    # TODO : extend support to multiple files

    dropout_evaluation_file = dropout_evaluations[0]
    dropout_dist = dropout_evaluation_file.label_to_method(dataset)
    check_size([pyro_dist] + [dropout_dist])
    dropout_targets = dropout_evaluation_file.target

    bins = np.arange(0, 1, 0.04)
    for i in range(num_images):
        plt.figure()
        hist_kwargs = {
            "histtype": "step",
            "linewidth" : 2,
            "bins" : bins
        }
        
        plt.hist(pyro_dist[:,i], color="firebrick", label=f"BNN (target = {pyro_targets[i]})", **hist_kwargs)
        plt.hist(dropout_dist[:,i], color="dodgerblue", label=f"Dropout (target = {dropout_targets[i]})", **hist_kwargs)
        plt.xlabel("Probability")
        plt.ylabel("Evaluations")
        plt.legend(frameon=False, loc="upper center")
        plt.savefig(PLOT_DIR+f"/posteriors/{dataset}_posterior_{i}")



def compare_statistic(ks_pvalues, datasets):
    # Evaluate K-S Test Between predictive distributions
    # For each image. WARNING - this method is very slow now
    # todo: add multithreading here? Not positive if vecotization is possible
    evals_to_do = len(list(ks_pvalues.values())[0])
    #evals_to_do = 100
    for i in range(evals_to_do):
        # Print status
        if 100*(i/evals_to_do) % 10 == 0:
            print(f"{100*(i/evals_to_do)} percent done.")

        # check each dataset type
        for ds in datasets:

            # For each dropout case
            for j in range(len(dropout_evaluations_ds_list[ds])):  
                if ds != "all_probabilities":
                    try:
                        ks_eval = ks_2samp(pyro_ppc_ds[ds][:,i], dropout_evaluations_ds_list[ds][j][:,i])
                        ks_pvalues[ds][i,j] = ks_eval.pvalue
                        ks_statistics[ds][i,j] = ks_eval.statistic
                    except:
                        pdb.set_trace()

                else:
                    for c in range(0,10):
                        try:
                            ks_eval = ks_2samp(pyro_ppc_ds[ds][:,i,c], dropout_evaluations_ds_list[ds][j][:,i,c])
                            ks_pvalues[ds][i,j] = ks_eval.pvalue
                            ks_statistics[ds][i,j] = ks_eval.statistic

                        except:
                            pdb.set_trace()

    # Win
    for ds in ks_pvalues.keys():
        ks_pvalues[ds] = ks_pvalues[ds][:evals_to_do]
        ks_statistics[ds] = ks_statistics[ds][:evals_to_do]



def ks_comparison(pyro_ppc, dropout_evaluations: List[DropoutEvaluations], datasets: List[str]):
    # TODO - add dropout rate for multiple plotting

    pyro_ppc_ds = {d: pyro_ppc[d] for d in datasets}
    
    dropout_evaluations_ds_list = {}
    for d in datasets:
        dropout_evaluations_ds_list[d] = [e.label_to_method(d) for e in dropout_evaluations]
        check_size([pyro_ppc_ds[d]] + dropout_evaluations_ds_list[d])

    # Generate empty arrays; size nImages
    ks_pvalues = {
        d: np.zeros((list(pyro_ppc_ds.values())[0].shape[1], len(dropout_evaluations_ds_list[d]))) for d in datasets
    }
    ks_statistics = {
     d : np.zeros((list(pyro_ppc_ds.values())[0].shape[1], len(dropout_evaluations_ds_list[d]))) for d in datasets
    }

    # Evaluate K-S Test Between predictive distributions
    # For each image
    evals_to_do = len(list(ks_pvalues.values())[0])
    #evals_to_do = 100
    for i in range(evals_to_do):
        # Print status
        if 100*(i/evals_to_do) % 10 == 0:
            print(f"{100*(i/evals_to_do)} percent done.")

        # check each dataset type
        for ds in datasets:

            # For each dropout case
            for j in range(len(dropout_evaluations_ds_list[ds])):  
                if ds != "all_probabilities":
                    try:
                        ks_eval = ks_2samp(pyro_ppc_ds[ds][:,i], dropout_evaluations_ds_list[ds][j][:,i])
                        ks_pvalues[ds][i,j] = ks_eval.pvalue
                        ks_statistics[ds][i,j] = ks_eval.statistic
                    except:
                        pdb.set_trace()

                else:
                    for c in range(0,10):
                        try:
                            ks_eval = ks_2samp(pyro_ppc_ds[ds][:,i,c], dropout_evaluations_ds_list[ds][j][:,i,c])
                            ks_pvalues[ds][i,j] = ks_eval.pvalue
                            ks_statistics[ds][i,j] = ks_eval.statistic

                        except:
                            pdb.set_trace()

    # Win
    for ds in ks_pvalues.keys():
        ks_pvalues[ds] = ks_pvalues[ds][:evals_to_do]
        ks_statistics[ds] = ks_statistics[ds][:evals_to_do]

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(10,10),ncols=1, nrows=1)
        plt.hist(ks_pvalues[ds][:,0], bins=100, density=False,color="#BD3039", alpha=0.7)
        plt.xlabel("p-value")
        plt.ylabel("Count (10k images)")
        plt.savefig(f"{PLOT_DIR}/{ds}_ks-pvals.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(10,10),ncols=1, nrows=1)
        plt.hist(ks_statistics[ds][:,0], bins=100, density=False,color="#BD3039", alpha=0.7)
        plt.xlabel("KS Statistic")
        plt.ylabel("Count (10k images)")
        plt.savefig(f"{PLOT_DIR}/{ds}_ks-statistic.png")
        plt.close()



def dict_entry_from_label(pyro_file: h5py.File, dropout_files: List[h5py.File], dataset: str) -> Tuple:
    """Generates tuple of form (bnn_dataset, dropout_dataset_1, dropout_dataset_2, ...) from a given label

    Args:
        pyro_file (h5py.File): bnn ppc output 
        dropout_files (List[h5py.File]): list of dropout files
        dataset (str): name of dataset from h5py file to obtain

    Returns:
        Tuple: tuple of all datasets for the label in form (bnn_dataset, dropout_dataset_1, dropout_dataset_2, ...)
    """

    # Merge bnn output to list of dropouts
    dataset_list =  [pyro_file[dataset]]  + [f[dataset] for f in dropout_files]
    # Return Tuple
    return tuple(dataset_list)



def main():

    # Load pyro PPC
    pyro_file = h5py.File(sys.argv[1], mode="r")
    # Load dropout evaluations
    dropout_files = [DropoutEvaluations(f) for f in sys.argv[2:]]

    # Generate dictionary of quantities to check
    labels = ["image_true_probs", "image_false_probs", "all_probabilities"]

    # Run KS 
    #ks_comparison(pyro_file, dropout_files, labels)
    plot_posteriors(pyro_file, dropout_files, "image_true_probs", 6)
    plot_posteriors(pyro_file, dropout_files, "image_false_probs", 6)

if __name__ == "__main__":
    main()

