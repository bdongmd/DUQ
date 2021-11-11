"""Library of evaluation plots to perform in-time on calibration.py"""
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.pyplot import fignum_exists, savefig
import numpy as np
from numpy.core.records import record
from torch import floor
from calibrate import EvaluationRecords
from typing import Optional
from scipy import stats

import pdb
SAVEDIR = "../output/testResult/uncertainty/"


def plot_style(x_label: str, y_label: str) -> None:
    """General function to apply style formats to all plots

    Args:
        x_label (str): x-axis label
        y_label (str): y-axis label
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    _, labels = plt.gca().get_legend_handles_labels()
    if len(labels) >0:
        plt.legend(frameon=False) 

def save_fig(title: Optional[str] = None, pdf: Optional[PdfPages] = None) -> None:
    """Save figure, either as single png or to the active PdfPages object

    Args:
        title (Optional[str], optional): Title of file (only used for single-png save). Defaults to None.
        pdf (Optional[PdfPages], optional): PdfPages object to save to (if applicable). Defaults to None.

    Raises:
        ValueError: If niether pdf or title are passed, desired output is ambiguous, so raise error
    """
    if pdf:
        pdf.savefig()
    elif title:
        # Check valid extension
        if not ".pdf" in title or not ".png" in title:
            title += ".png"
        # Add save directory to title
        if SAVEDIR not in title:
            title = SAVEDIR + title
        plt.savefig(title)
    else:
        raise ValueError("Must pass either a plot title or PdfPages object")


#### plot significance comparison
def plot_significance(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots significance histogram

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10,10),ncols=1, nrows=1)
    plt.hist(np.array(records.significance), bins=100, range=[-5,5],density=True, label="Dropout Calculated", alpha=0.7)
    plt.hist(stats.norm.ppf(records.image_acc),bins=100, range=[-5,5],color='orange',density=True, label="Dropout Observed", alpha=0.7)
    plot_style(x_label="Frequency", y_label="Classification Significance")
    save_fig(title="significance", pdf=pdf)
    plt.close()

#### plot probability comparison
def plot_probability(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots probability histogram

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """    
    fig,ax = plt.subplots(figsize=(10,10),ncols=1, nrows=1)
    plt.hist(np.array(records.probability), bins=200,range=[0,1],density=True, label="Dropout calculated", alpha=0.7)
    plt.hist(np.array(records.image_acc),bins=200,range=[0,1],density=True, label="Dropout Observed", alpha=0.7)
    plot_style(x_label="Frequency", y_label="Classification Probability")
    save_fig(title="probability", pdf=pdf)
    plt.close()

#### plot significance vs probability 
def plot_sig_v_prob(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots probability vs significance distribution

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """    
    fig,ax = plt.subplots(figsize=(10,10),ncols=1, nrows=1)
    plt.plot(records.significance, records.probability,'o')
    plot_style("Classification Significance", "Classification Probability")
    save_fig("sig_v_prob", pdf=pdf)
    plt.close()

#### plot calculated probability vs observed and pull
def plot_prob_v_pull(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots significance vs accuracy and pulls, both symmetric and asymmetric

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """    
    fig, ax = plt.subplots(figsize=(10,10),ncols=2, nrows=2)
    plt.sca(ax[0,0])
    plt.plot(records.image_acc, records.probability, 'o')
    plot_style("Image accuracy", "Calculated Probability")

    plt.sca(ax[0,1])
    plt.plot(records.image_acc, records.probability_asy, 'o')
    plot_style("Image accuracy", "Calculated Probability (asy)")

    plt.sca(ax[1,0])
    plt.hist(records.image_pull,bins=300,range=[-5,5], alpha=0.7, label="Pull from symmetric uncertainty", density=True)
    plot_style("Pull", "Probability Density")

    plt.sca(ax[1,1])
    plt.hist(records.image_pull_asy,bins=300,range=[-5,5], alpha=0.7, label="Pull from asymmetric uncertainty", density=True)
    plot_style("Pull", "Probability Density")


    save_fig("prob_v_pull", pdf=pdf)
    plt.close()

#### plot probability - accuracy
def plot_prob_m_acc(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots probability minus accuracy

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """    
    fig,ax = plt.subplots(figsize=(10,10),ncols=2,nrows=1)
    plt.sca(ax[0])
    plt.hist(records.probability - records.image_acc.numpy(), bins=200, range=[-1,1], density=True, alpha=0.7)
    plot_style("Calculated Probability - Image Accuracy", "Frequency")

    plt.sca(ax[1])
    plt.hist(records.probability_asy - records.image_acc.numpy(), bins=200,range=[-1,1],density=True, alpha=0.7)
    plot_style("Calculated Probability(asy) - Image Accuracy", "Frequency")

    save_fig("prob_m_acc", pdf)
    plt.close()

#### score difference
def plot_score_diff(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots dropout disabled minus enabled score

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """    
    fig,ax = plt.subplots(figsize=(10,10),ncols=1,nrows=1)
    plt.hist(records.NoDro_true_probs.numpy() - records.true_mu.numpy(), bins=200,range=[-1,1],density=True, alpha=0.7)
    plot_style("dropout disabled score - dropout enabled average score", "Frequency")
    save_fig("dropout_score_diff", pdf=pdf)
    plt.close()

#### plot uncalibrated probability vs accuracy
def plot_mu_v_acc(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots output probability vs image accuracy

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """   

    fig,ax = plt.subplots(figsize=(10,10),ncols=1, nrows=1)
    plt.plot(records.image_acc, records.true_mu, 'o')
    plot_style("Image accuracy", "Softmax output probability")
    plt.ylim([0,1])
    save_fig("mu_v_acc", pdf=pdf)
    plt.close(fig)

### plot sample accuracy and consistency with predicted.
def plot_accuracies(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots accuracy lines

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """   

    fig,ax = plt.subplots(figsize=(10,10),ncols=1,nrows=1)
    ax.axvline(
        records.probability.sum()/len(records.probability), ls='--', color='blue', 
        label="Dropout Calculated Accuracy: %3.3f"%(records.probability.sum()/len(records.probability))
        )
    ax.axvline(
        records.probability_asy.sum()/len(records.probability_asy), ls='--', color='green', 
        label="Dropout Calculated (from asy uncer) Accuracy: %3.3f"%(records.probability_asy.sum()/len(records.probability_asy))
        )
    ax.axvline(
        sum(records.image_acc)/len(records.image_acc), ls='--',color='red',
        label="Observed Accuracy: %3.3f"%(sum(records.image_acc)/len(records.image_acc))
        )
    plot_style("Accuracy","")
    save_fig("accuracies", pdf=pdf)
    plt.close()

#### plot pull test
def plot_pull_test(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots test version of pull distribution

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """      
    fig,ax = plt.subplots(figsize=(10,10),ncols=1,nrows=1)
    loc,sca=stats.norm.fit(records.image_pull)
    plt.hist(records.image_pull_test,bins=1000,range=[-5,5], alpha=0.7, label="Image to Sample Pull", density=True)
    plt.plot(np.linspace(-5,5,1000), stats.norm.pdf(np.linspace(-5,5,1000), loc=loc,scale=sca), linewidth=2,color='green',label="Gaussian Fit loc: %3.2f scale: %3.2f"%(loc,sca))
    plot_style("Pull", "Probability Density")
    save_fig("pull_test", pdf=pdf)
    plt.close()


def plot_posterior_comparison(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """ Plots pull derived from posteriors using mode estimator and mean esimator

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """

    fig = plt.figure()

    # MLE true class probability distribution
    plt.hist(records.image_pull, bins=np.arange(-2,2,0.1), label="Mean Estimator (CI)", color="firebrick", alpha=0.6)
    plt.hist(records.image_pull_mode[0,:], bins=np.arange(-2,2,0.1), label="Mode Estimator (HPDI)", color="cornflowerblue", alpha=0.6)
    plt.legend(frameon=False)
    plt.xlabel("Asymmetric Pull Distribution")
    save_fig("posterior_comparison", pdf=pdf)
    

def plot_single_true_distributions(
    records: EvaluationRecords, pdf: Optional[PdfPages] = None, 
    number=20, draw_mean: bool = True, draw_mode: bool = True, draw_median: bool = False
    ) -> None:
    """ Plots posterior distribution for [number] images, with indicated summary statistics

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
        number (int, optional): Number of images to draw predictions for. Defaults to 20. TODO generalize
        draw_mean (bool, optional): Draw mean indicator. Defaults to True.
        draw_mode (bool, optional): Draw mode indicator. Defaults to True.
        draw_median (bool, optional): Draw median indicator. Defaults to False.
    """

    col = 5
    row = math.ceil(number/col)

    fig, ax = plt.subplots(nrows=row, ncols=col)
    i=0
    for r in range(row):
        for c in range(col):
            plt.sca(ax[r,c])
            bins, _, _ = plt.hist(records.true_probs[:,i].numpy(), bins=np.arange(0,1,0.01))
            if draw_mean:
                mean = records.true_probs[:,i].mean()
                plt.axvline(mean, ymin=0, ymax=max(bins), color="firebrick", linestyle="--", linewidth=1)
            if draw_mode:
                plt.axvline(records.max_true_posterior[:,i], ymin=0, ymax=max(bins), color="forestgreen", linestyle="--", linewidth=1)
            if draw_median:
                median_val = np.median(records.true_probs.numpy(), axis=0)
                plt.axvline(median_val, ymin=0, ymax=max(bins), color="violet", linestyle="--", linewidth=1)

            plt.title(f"Image {i}", fontsize=10)
            i+=1
    plt.tight_layout()
    save_fig("image_true_distributions", pdf=pdf)


def plot_single_distributions_with_output(records: EvaluationRecords, pdf: Optional[PdfPages] = None, number=5) -> None:
    """Plots single image distributions with dropout disabled score

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
        number (int, optional): number of images to draw. Defaults to 5. TODO generalize
    """
    
    col = number
    row = 1

    fig, ax = plt.subplots(nrows=row, ncols=col, figsize=(12,4))
    i=0
    for r in range(row):
        for c in range(col):
            if row !=1:
                plt.sca(ax[r,c])
            else:
                plt.sca(ax[c])

            mean = records.true_probs[:,i].mean()
            bins, _, _ = plt.hist(records.true_probs[:,i].numpy(), bins=np.arange(0,1,0.01))
            plt.axvline(mean, ymin=0, ymax=max(bins), color="firebrick", linestyle="--", linewidth=1)
            plt.axvline(records.max_true_posterior[:,i], ymin=0, ymax=max(bins), color="forestgreen", linestyle="--", linewidth=1)
            plt.axvline(records.NoDro_true_probs[i], ymin=0, ymax=max(bins), color="violet", linestyle="--", linewidth=1)
            plt.title(f"Image {i}", fontsize=10)
            i+=1
    plt.tight_layout()
    save_fig("image_with_prediction", pdf=pdf)


#### plot calculated probability vs observed and pull
def plot_probability_comparison(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Plots significance vs accuracy and pulls, both symmetric and asymmetric

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
        TODO - normalize colorbar, if plots are worth keeping
    """   

    fig, ax = plt.subplots(figsize=(10,8),ncols=2, nrows=1)
    plt.sca(ax[0])
    ibins = np.arange(0,1,0.02)
    h=plt.hist2d(records.image_acc.numpy(), records.probability_asy, bins=[ibins,ibins],cmap="Blues")
    plot_style("Image accuracy", "Mean Probability")

    plt.sca(ax[1])
    h=plt.hist2d(records.image_acc.numpy(),records.modal_probability_asy, bins=[ibins,ibins],cmap="Blues")
    plot_style("Image accuracy", "Modal Probability") 
    save_fig("probability_comparison", pdf=pdf)   

#### plot significance comparison
def plot_significance_comparison(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Comparison of significance from mode estimator vs mean estimator

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """
    fig, ax = plt.subplots(figsize=(10,10),ncols=1, nrows=1)
    plt.hist(np.array(records.significance_asy), bins=100, range=[-5,5],density=True,color="cornflowerblue", label="Mean Estimator", alpha=0.7)
    plt.hist(np.array(records.modal_significance_asy),bins=100, range=[-5,5],color='firebrick',density=True, label="Modal Estimator", alpha=0.7)
    plot_style(x_label="Frequency", y_label="Classification Significance")
    save_fig(title="significance_comp", pdf=pdf)
    plt.close()


def plot_distance_metrics(records: EvaluationRecords, pdf: Optional[PdfPages] = None) -> None:
    """Draw the distance of the classical ML prediction to various posterior summary statistics

    Args:
        records (EvaluationRecords): saved evaluation parameters 
        pdf (Optional[PdfPages], optional): Pass desired PdfPages object, if none passed, saves as png. Defaults to None.
    """

    #Calculate distance from various summary statistics

    mean_val = np.mean(records.true_probs.numpy(), axis=0)
    d_mean = records.NoDro_true_probs.numpy() - mean_val
    median_val = np.median(records.true_probs.numpy(), axis=0)
    d_median = records.NoDro_true_probs.numpy() - median_val
    mode_val = stats.mode(np.around(records.true_probs, 2)).mode
    d_mode = records.NoDro_true_probs.numpy() - mode_val[0,:]
    d_mode_mean = mode_val - mean_val
    #pdb.set_trace()

    fig, ax = plt.subplots(nrows=3, figsize=(5,8))

    plt.sca(ax[0])
    mean_std = round(float(d_mean.std()),3)
    plt.hist(d_mean, bins=100, label=f"$\sigma$ = {mean_std}")
    plt.legend(frameon=False)
    plt.xlabel("Classical Point Est. - Posterior Mean")
    
    plt.sca(ax[1])
    median_std = round(float(d_median.std()),3)
    plt.hist(d_median, bins=100, label=f"$\sigma$ = {median_std}")
    plt.legend(frameon=False)
    plt.xlabel("Classical Point Est. - Posterior Median")

    plt.sca(ax[2])
    mode_std = round(float(d_mode.std()),3)
    plt.hist(d_mode, bins=100, label=f"$\sigma$ = {mode_std}")
    plt.legend(frameon=False)
    plt.xlabel("Classical Point Est. - Posterior Mode")

    plt.tight_layout()
    save_fig("distance_metrics",pdf=pdf)
    #plt.title(f"Image {i}", fontsize=10)
