""" Study to analyze varied dropout rates
--- Sept 2020 | tjb ---
"""
# utils
import sys
from collections import namedtuple

# Analysis Tools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Local Imports
from dropout_evaluations import DropoutEvaluations

doHist, doScatter= True, False

def evaluate_posterior_parameters(evaluation):
    """Generate summary statistics from each image posterior 
    Args:
        evaluation (DropoutEvaluations): readout of file from calibrate.py
    """

    # what is posterior probability dataset?
    posterior_mean = np.mean(evaluation.true_probs) # mean of each posterior
    posterior_std = np.std(evaluation.true_probs)

    return posterior_mean, posterior_std


def evaluate_pull_parameters(evaluation):
    """Generate summary statistics from pull distribution
    Args:
        evaluation (DropoutEvaluations): readout of file from calibrate.py
    """

    pull_mean = np.mean(evaluation.pull)
    pull_std = np.std(evaluation.pull)
    return pull_mean, pull_std


def style():
    """Add consistent labels to plots
    """
    plt.xlabel("Dropout Rate")
    plt.ylabel("Pull")


def pull_aggregation_plot(all_evaluations):
    """Make plot comparing pull distributions across dropout rates
    Args:
        all_evaluations (list(DropoutEvaluations)): All parsed files loaded through utility class
    """
    # Generate data constant with dropout on X, pull distribution on y
    one_sigma = []
    two_sigma = []
    do_rates = []

    # Aggregate all data (for hist)
    x = np.array([])
    y = np.array([])

    for evaluation in all_evaluations:

        # pull plot
        y_data = evaluation.pull

        # X just needs to be the dropout rate over and over
        x_data = np.ones(len(evaluation.pull)) * evaluation.test_dropout_rate

        # Dropout rates
        do_rates.append(evaluation.test_dropout_rate)

        #  Get quantiles
        one_sigma.append(np.quantile(y_data, [.15865, .84135]))
        two_sigma.append(np.quantile(y_data, [.02275, .97725]))

        # Add data
        x = np.append(x, x_data)
        y = np.append(y, y_data)

    lower_one, upper_one = [x[0] for x in one_sigma], [x[1] for x in one_sigma]
    lower_two, upper_two = [x[0] for x in two_sigma], [x[1] for x in two_sigma]

    if doHist:
        # Plot histogram
        fig = plt.figure(dpi=200)
        x_binwidth = 0.02
        xbins = np.arange(0,1, x_binwidth)
        ybins = np.linspace(-3, 3, 100)
        hist=plt.hist2d(x, y, bins=[xbins, ybins], cmap="Blues")
        
        # Generate Maximum a posteriori line
        nonzero_bin_y = [ybins[j] for j in [np.argmax(i) for i in hist[0]] if j !=0]
        plt.plot(do_rates, nonzero_bin_y, color="black",label="MAP")

        # Deviations
        plt.plot(do_rates, upper_two, color="goldenrod",label="$2\sigma$")
        plt.plot(do_rates, lower_two, color="goldenrod")
        plt.plot(do_rates, upper_one, color="green",label="$1\sigma$")
        plt.plot(do_rates, lower_one, color="green")

        # Style and save
        plt.legend(frameon=False)
        style()
        plt.savefig("../output/plots/varied_do/pulls_hist.png")
        plt.close()

    if doScatter:
        ### Old code, shoudn't use these plots
        ### Was hoping the low-alpha overlay would make a sort of "density" plot, but ended up just a blue line.

        # Plot quantiles - kind of ugly, probably consider a better way
        fig = plt.figure(dpi=200)


        # Adjust do rates to cover marker width
        do_rates[0] = do_rates[0]-0.01
        do_rates[-1] = do_rates[-1]+0.01

        # Start with plotting 1 and 2 sigma bands
        plt.fill_between(x=do_rates, y1=upper_two, y2=lower_two, color="goldenrod",alpha=0.3, label="$2\sigma$")
        plt.fill_between(x=do_rates, y1=upper_one, y2=lower_one, color="green",alpha=0.3, label="$1\sigma$")

        # Plot pull distributions
        for i, evaluation in enumerate(all_evaluations):
            x_data = np.ones(len(evaluation.pull)) * evaluation.test_dropout_rate
            y_data = evaluation.pull
            plt.scatter(x_data, y_data, color="cornflowerblue", alpha=0.002, s=4)
        
        # Spoof marker color for legend (otherwise would be alpha=0.01)
        plt.scatter([],[], color="cornflowerblue", label="Pull Distribution")
        plt.legend(frameon=False)
        plt.ylim(
            bottom=(min(lower_two)-0.5*abs(min(lower_two))),
            top=(max(upper_two)+0.5*abs(max(upper_two)))
        )
        # Finalize
        style()
        plt.savefig("../output/plots/varied_do/pulls_scatter.png")
        plt.close()

if __name__ == "__main__":

    file_list = sys.argv[1:]
    evaluations = [DropoutEvaluations(f) for f in file_list]
    pull_aggregation_plot(evaluations)
