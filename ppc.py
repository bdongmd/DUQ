"""Create generic EvaluationRecords object for a network"""

import sys
import timeit
import h5py
import torch
from bnn import BCNNWeak

# Utils
import argparse
from EvaluationRecords import EvaluationRecords
from utils import *

# Thing for typing
from typing import Union
from scipy.stats.stats import mode
from torch.serialization import load

from torch.utils.data.dataloader import DataLoader
from modules import network
from pyro.infer import Predictive

tryGPU = False
testGaussianNoise = False
GaussStd = 0.8
testExcludeOneNumber = False
RotateImageRandom = False
RotateImageSpecific = False
RotateAngle = 1.0

useGPU = (tryGPU and torch.cuda.is_available())
DEVICE = torch.device("cuda" if useGPU else "cpu")
print("Running on {}.".format("GPU" if useGPU else "CPU" ))


### Probabilities 
@torch.no_grad() 
def evaluation_bnn(
    model: Predictive, 
    test_loader: DataLoader, 
    records: EvaluationRecords, 
    n_predictions: int, 
    timer: Timer
    ) -> None:
    """Do evaluation for BNN
    TODO: THis code is very similar to evaluation for Dropout, possibly a way to merge

    Args:
        model (Predictive): Pyro Predictive
        test_loader (DataLoader): Test data set dataloader
        records (EvaluationRecords): Evaluation Records object to store output
        n_predictions (int): Number of posterior predictions to do
        timer (Timer): Timer object for counting time on each processor
    """


    total_evals = len(test_loader)
    model.eval()
    model.to(DEVICE)
    for i, (data, target) in enumerate(test_loader):

        # Consider reducing print frequency for small batch size or many evals
        print(f"Progress -- {round(100*i/total_evals)}% done")
        init = timeit.default_timer()
        data, target = data.to(DEVICE), target.to(DEVICE)
        final = timeit.default_timer()
        data_moving_time = (final-init)

        init = timeit.default_timer()
        # Evaluate probabilities
        # Gives tensor along axis [n_predictions, dataloader size, mnist digits]
        probabilities = model(data)
        final = timeit.default_timer()
        model_eval_time = final-init


        init = timeit.default_timer()
        # Convert to probabilities
        probabilities = probabilities["probs"]

        # Get most likely labels and probabilities of those        
        max_evaluated_labels = torch.sort(probabilities, dim=2, descending=True).indices[:,:,0]
        max_evaluated_probabilities = torch.sort(probabilities, dim=2, descending=True).values[:,:,0]
        sec_evaluated_labels = torch.sort(probabilities, dim=2, descending=True).indices[:,:,1]
        sec_evaluated_probabilities = torch.sort(probabilities, dim=2, descending=True).values[:,:,1]

        # Make boolean slice for when the max prob == true
        dim_corrected_target = target.repeat(n_predictions, 1)
        model_match_bool = max_evaluated_labels == dim_corrected_target

        # Fill them in the appropriate indices
        block_start_index = i * test_loader.batch_size
        block_end_index = block_start_index + test_loader.batch_size
        
        ## Dropout
        records.probs[:, block_start_index:block_end_index] = max_evaluated_probabilities
        records.matched_bool[:, block_start_index:block_end_index] = model_match_bool

        # Save targets and probabilities to record
        records.targets[block_start_index:block_end_index] = target
        records.all_probabilities[:, block_start_index:block_end_index,:] = probabilities
        records.all_max_evaluated_probabilities[:,  block_start_index:block_end_index] = max_evaluated_probabilities
        records.all_sec_evaluated_probabilities[:,  block_start_index:block_end_index] = sec_evaluated_probabilities
        final = timeit.default_timer()
        calc_time = (final-init)
    
        # Update Timer
        timer.add(time=data_moving_time, device="cpu")
        timer.add(time=model_eval_time, device="gpu" if useGPU else "cpu")
        timer.add(time=calc_time, device="cpu")
    

def main():
    # Validate a configuration file is passed
    if len(sys.argv) < 2:
        raise ValueError("Pass a configuration file (YAML) as an argument.")
    cfg = load_configuration_file(sys.argv[1])

    # Load dataset
    gaussian_std = GaussStd if testGaussianNoise else None
    random_rotation = RotateAngle if RotateImageRandom else None
    specific_rotation = RotateAngle if RotateImageSpecific else None
    transforms = get_transform(gaussian_std, random_rotation, specific_rotation)
    test_loader, selected_images = load_data(
        ds_start=cfg["range_min"], ds_end=cfg["range_max"], train=False,
        useGPU=cfg["useGPU"], b_size=cfg["batch_size"], exclude_number=cfg["testExcludeOneNumber"],
        transforms=transforms, static=True
    )

    # Create Evaluation Records object
    records = EvaluationRecords(cfg["samples"], len(selected_images), cfg["range_min"], cfg["range_max"])
    # Load all saved model artifacts
    model_artifacts = torch.load(cfg["model_file"])
    # Specifically get Predictive object
    model = model_artifacts[cfg["model_name"]]    

    # Instantiate timer
    timer = Timer()

    # Run evaluation
    evaluation_bnn(
        model=model, 
        test_loader=test_loader,
        records=records,
        n_predictions=cfg["samples"],
        timer=timer
        )

    # Calculate end-of-evaluation parameters on samples
    records.finalize()
    # Save to File
    hf = h5py.File(cfg["posterior_file"], mode="w")    
    records.save_to_file(hf)


if __name__ == "__main__":
    main()