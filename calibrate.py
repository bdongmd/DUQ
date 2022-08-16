from torch.nn.modules import utils
from models import network
import sampleAcc 

import h5py
import torch
from torch.utils.data import DataLoader
import sys
import timeit
from utils import load_data, get_transform
from plotting.eval_plot_library import *

import arviz as az
sample_batch_size = 512
image_batch_size = 1
debug = False
tryGPU = True
testGaussianNoise = False
GaussStd = 0.8
testExcludeOneNumber = False
RotateImageRandom = False
RotateImageSpecific = False
RotateAngle = 1.0
PLOT_PAGES = True # True -> plot PdfPages object, else individual plots

useGPU = (tryGPU and torch.cuda.is_available())
DEVICE = torch.device("cuda" if useGPU else "cpu")
print("Running on {}.".format("GPU" if useGPU else "CPU" ))



class Timer():
    """Class to keep track of time spent in evaluation"""

    def __init__(self) -> None:
        """Initalize object"""
        self.cpu_time = 0
        self.gpu_time = 0
        self.total_time = 0

    def __str__(self) -> str:
        """String representation of timer

        Returns:
            str: if only CPU, shows total time. If GPU exists, time on each device
        """
        if self.gpu_time > 0: 
            percent_cpu = self.cpu_time / self.total_time
            return(f"Total time: {self.total_time:.3f}.\n \
                \tCPU: {self.cpu_time} ({percent_cpu}%)\n \
                \tGPU: {self.gpu_time} ({1-percent_cpu}%)")
        else:
            return f"Total time: {self.total_time:.3f}"

    def add(self, time: float, device: str = "cpu") -> None:
        """Add time to the timer object

        Args:
            time (float): Amount of time to add
            device (str, optional): Which device to credit the time to. Defaults to "cpu".

        Raises:
            ValueError: Raised if device specified is not known
        """
        if device == "cpu":
            self.cpu_time += time
        elif device == "cuda" or device == "gpu":
            self.gpu_time += time
        else: 
            raise ValueError("Unknown device specified in timer")
        
        self.total_time = self.cpu_time + self.gpu_time


def load_model(f_state_dict: str, trainmodel: str = "CNNWeak", dropout_rate: float = 0.2) -> network:
    """Load trained model  

    Args:
        f_state_dict (str): string designating location of state dictionary saved in training
        trainmodel (str, optional): Model type to load. Defaults to "CNNWeak".
        dropout_rate (float, optional): dropout rate to apply. Defaults to 0.2.

    Returns:
        network: trained model object
    """    


    output_layer_size = 9 if testExcludeOneNumber else 10
    model = network(
        model_name=trainmodel,
        p=dropout_rate,
        output_layer=output_layer_size
        )
    model.load_state_dict(torch.load(f_state_dict))
    return model


### Probabilities 
def evaluation(model: network, test_loader: DataLoader, records: EvaluationRecords, n_predictions: int, timer: Timer) -> None:
    """ Core evaluation of this code, evaluating model many times with dropout enabled

    Args:
        model (network): Trained network
        test_loader (DataLoader): Test dataset
        records (EvaluationRecords): Object to record outputs
        n_predictions (int): Number of evaluations to perform
        timer (Timer): Timer object to keep track of where time is spent
    """

    with torch.no_grad():
        total_evals = len(test_loader)
        model.to(DEVICE)
        for i, (data, target) in enumerate(test_loader):

            # Consider reducing print frequency for small batch size or many evals
            print(f"Progress -- {round(100*i/total_evals)}% done")

            init = timeit.default_timer()
            data, target = data.to(DEVICE), target.to(DEVICE)
            final = timeit.default_timer()
            data_moving_time = (final-init)

            init = timeit.default_timer()
            # Evaluate dropout predictions
            # Gives tensor along axis [n_predictions, dataloader size, mnist digits]
            model.train() # Turn on "training" so dropout is invoked (never call backwards, so no network updating)
            dropout_output = torch.stack([model(data) for _ in range(n_predictions)])

            # Evaluate non-dropout predictions
            model.eval() # Shuts off dropout
            noDropout_output = model(data)
            final = timeit.default_timer()
            model_eval_time = final-init


            init = timeit.default_timer()
            # Convert to probabilities
            probabilities = torch.exp(dropout_output)
            noDropout_probabilities = torch.exp(noDropout_output)

            # Get most likely labels and probabilities of those        
            ## Dropout
            max_evaluated_labels = torch.sort(probabilities, dim=2, descending=True).indices[:,:,0]
            max_evaluated_probabilities = torch.sort(probabilities, dim=2, descending=True).values[:,:,0]
            sec_evaluated_labels = torch.sort(probabilities, dim=2, descending=True).indices[:,:,1]
            sec_evaluated_probabilities = torch.sort(probabilities, dim=2, descending=True).values[:,:,1]
            ## No dropout
            noDO_max_evaluated_labels = torch.sort(noDropout_probabilities, descending=True).indices[:,0]
            noDO_max_evaluated_probabilities = torch.sort(noDropout_probabilities, descending=True).values[:,0]
            noDO_sec_evaluated_labels = torch.sort(noDropout_probabilities, descending=True).indices[:,1]
            noDO_sec_evaluated_probabilities = torch.sort(noDropout_probabilities, descending=True).values[:,1]

            # Make boolean slice for when the max prob == true
            ## Dropout
            dim_corrected_target = target.repeat(n_predictions, 1)
            model_match_bool = max_evaluated_labels == dim_corrected_target
            ## No Dropout
            noDO_model_match_bool = noDO_max_evaluated_labels == target

            # Fill them in the appropriate indices
            block_start_index = i * test_loader.batch_size
            block_end_index = block_start_index + test_loader.batch_size
            
            ## Dropout
            records.probs[:, block_start_index:block_end_index] = max_evaluated_probabilities
            records.matched_bool[:, block_start_index:block_end_index] = model_match_bool
            ## No Dropout
            records.noDO_probs[block_start_index:block_end_index] = noDO_max_evaluated_probabilities
            records.noDO_matched_bool[block_start_index:block_end_index] = noDO_model_match_bool

            # Save targets and probabilities to record
            records.targets[block_start_index:block_end_index] = target
            records.all_probabilities[:, block_start_index:block_end_index,:] = probabilities
            records.all_noDropout_probabilities[block_start_index:block_end_index,:] = noDropout_probabilities
            records.all_max_evaluated_probabilities[:,  block_start_index:block_end_index] = max_evaluated_probabilities
            records.all_sec_evaluated_probabilities[:,  block_start_index:block_end_index] = sec_evaluated_probabilities
            records.all_noDO_max_evaluated_probabilities[block_start_index:block_end_index] = noDO_max_evaluated_probabilities
            records.all_noDO_sec_evaluated_probabilities[block_start_index:block_end_index] = noDO_sec_evaluated_probabilities
            final = timeit.default_timer()
            calc_time = (final-init)
        
            # Update Timer
            timer.add(time=data_moving_time, device="cpu")
            timer.add(time=model_eval_time, device="gpu" if useGPU else "cpu")
            timer.add(time=calc_time, device="cpu") # TODO: make sure this is on CPU
        

def main():
    # Read command line args
    trainmodel = sys.argv[1] 
    trainpara = sys.argv[2]
    dropout_rate = 0.1*int(sys.argv[3])
    predictions =  int(sys.argv[4])
    range_min = int(sys.argv[5])
    range_max = int(sys.argv[6])
    print(f"Predictions: {predictions}")

    # Load model
    state_dict_path = f'../output/trainModule/mnist_{trainmodel}_{trainpara}.pt'
    model = load_model(state_dict_path, trainmodel=trainmodel, dropout_rate=dropout_rate)
    print(model)

    # load dataset
    gaussian_std = GaussStd if testGaussianNoise else None
    random_rotation = RotateAngle if RotateImageRandom else None
    specific_rotation = RotateAngle if RotateImageSpecific else None
    transforms = get_transform(gaussian_std, random_rotation, specific_rotation)
    test_loader, selected_images = load_data(
        ds_start=range_min, ds_end=range_max, train=False,
        useGPU=useGPU, b_size=sample_batch_size, exclude_number=testExcludeOneNumber,
        transforms=transforms, static=True
    )

    # Calculate sample accuracy without dropout
    sample_acc = sampleAcc.F_sample_acc_without_dropout(model, test_loader, DEVICE)

    # Initalize evaluation records and timer
    records = EvaluationRecords(predictions, len(selected_images), range_min, range_max)
    timer = Timer()

    evaluation(model=model, test_loader=test_loader, records=records, n_predictions=predictions, timer=timer)
    # Print out timer summary
    print(timer)

    # Calculate post-evaluation quantities
    records.finalize()
    print("Calcualted sample accuracy = {}".format(sum(records.probability)/len(records.probability)))
    print("Calcualted sample accuracy (with asy uncertainty) = {}".format(sum(records.probability_asy)/len(records.probability_asy)))
    # Save evaluation file
    if testGaussianNoise:
        hf = h5py.File(f'../output/testResult/uncertainty/cali_{trainmodel}_{trainpara}_test{int(dropout_rate*10)}_image{range_min+1}_{range_max}_GaussStd{GaussStd}.h5',"w")
    else:
        print("Filename: "+ f'../output/testResult/uncertainty/cali_{trainmodel}_{trainpara}_test{int(dropout_rate*10)}_image{range_min+1}_{range_max}.h5')
        print("Pdfname: "+ f'../output/testResult/uncertainty/cali_{trainmodel}_{trainpara}_test{int(dropout_rate*10)}_image{range_min+1}_{range_max}.pdf')
        hf = h5py.File(f'../output/testResult/uncertainty/cali_{trainmodel}_{trainpara}_test{int(dropout_rate*10)}_image{range_min+1}_{range_max}.h5', "w")
    records.save_to_file(hf=hf)
    hf.close()

    # Generate plotting pdf
    if PLOT_PAGES:
        if testGaussianNoise:
            pdf = PdfPages(f"../output/testResult/uncertainty/cali_{trainmodel}_{trainpara}_test{int(dropout_rate*10)}_image{range_min+1}_{range_max}_GaussStd{GaussStd}.pdf")
        else:
            pdf = PdfPages(f"../output/testResult/uncertainty/cali_{trainmodel}_{trainpara}_test{int(dropout_rate*10)}_image{range_min+1}_{range_max}.pdf")
    else:
        pdf = None

    # Plot all methods
    for plot_method in [
        plot_significance,
        plot_probability, 
        plot_sig_v_prob,
        plot_prob_v_pull,
        plot_prob_m_acc,
        plot_score_diff,
        plot_mu_v_acc,
        plot_accuracies,
        plot_pull_test,
        plot_posterior_comparison,
        plot_single_true_distributions,
        plot_single_distributions_with_output,
        plot_probability_comparison,        
        plot_significance_comparison,
        plot_distance_metrics,
        ]:
        plot_method(records=records, pdf=pdf)

    if pdf:
        pdf.close()


if __name__ == "__main__":
    main()
