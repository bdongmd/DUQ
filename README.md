Dropout
=================

code for Dropout studies using MNIST dataset  
MNIST dataset download from: http://yann.lecun.com/exdb/mnist/


# Setup

The original package dependency is as below:

1. python3
2. numpy
3. matplotlib
4. torch
5. torchvision
6. h5py
7. scipy
8. cuda if you want to run with GPU

This has expanded with the Bayesian Neural Network, which requires [Pyro](http://pyro.ai/). Unfortunately, it does not exist on conda, so you'll have to install via pip. Run: `pip3 install pyro-ppl` - note the ppl! Crossing conda and pip makes things tangle easily, which is a pain.

To make this a bit easier, an `environment.yaml` file is included for easy installation: 

```
conda env create -f environment.yaml
````



# Models

The following following instructions outline how to train each network, then run it on the testing data. These are also drivers within the `Makefile` to make the procedure less involved. The general design flow is to train each network, run on testing data and produce one uniform data housing object `EvaluationRecords`, containing useful output quantities for either network.

## Dropout Network
---

Makefile approach:

```
make dropout_workflow
```

### Training:

```
python3 train.py $(model_name) $(droupout_rate) $(epochs) $(n_images) 
```

The `model_name` parameter must be defined in `modules.py`, generally you'll want `CNNWeak`. This produces a file of format:

```
../output/trainModule/mnist_{model_name}_dr0p{dropout_rate*10}_ep{epochs}_ev{n_images}.pt
```

### Evaluate on testing data:
```
python3 calibrate.py $(model_name) $(training_string) $(dropout_rate) $(posterior_samples) $(start_image) $(end_image)
```

Where `training_string` is the core string of the output file mentioned above, something like: `dr0p2_ep300_ev10000` for a dropout rate of 0.2, 300 epochs, and 10000 images.

This will produce the `EvaluationRecords` object, which will be saved at 

```
../output/testResult/uncertainty/cali_{model_name}_{training_string}_test{dropout_rate*10}_image{range_min+1}_{range_max}.h5
```

Additionally, it runs all of the plotting functions within the file `plotting/eval_plot_library.py`, which are saved within `../output/testResult/uncertainty/` (this directory is a global in that script, may be switched).


## Bayesian Neural Network
---

Makefile approach: 
```
make bnn_workflow
```

### Training

The Bayesian network is driven by a configuration file to set all the options. A sample can be found at `example_opts.yaml`.

For training, run:

```
python bnn.py options.yaml
```

This uses SVI to do the training and outputs a file with the Predictive object and state dictionary saved the name given in the configuration as `model_file` (`bcnn.pyro` in the example)

### Posterior Predictive Check

Next we want to use the model to make posterior predictions. The script takes the same options file:

```
python ppc.py options.yaml
```

This is the BNN equivalent of `calibrate.py`, and the output object is the same, the finalize method of `EvaluationRecords`.


# Analysis

A utility script to load out output of the `EvaluationRecords` object is found in `plotting/dropout_evaluations.py`.



Two primary analysis scripts live in this directory, `analysis_tests.py` is intended to perform analysis on a single model. 


To compare two (todo: or more) models, use the `model_comparison.py` script.

```
python model_comparison.py bnn_output.h5 dropout_1.h5 dropout_2.h5 dropout_3.h5 ....
```

Given the output of `calibrate.py` (the dropout posterior) and of `ppc.py` (the bnn posterior), we want to make comparisons. The following script (**WIP**) is built to do the image-wise KS test:


## Possible future upkeep todos
1. Create specific `EvaluationRecords` objects for the BNN and dropout network independently, which inherit from a parent class.
2. Make testing suite more rigorous. Current coverage is quite small, would be nice to expand this.


## Full Directory Contents

### Code

General

- `analysis_tests.py` - Analysis on single model
- `EvaluationRecords.py` - Class for recording evaluation on testing data
- `modules.py` - Defined NN models
- `mutualinfo.py` - Functions to calculate mutual information (KL divergence, Jensen-Shannon divergence) 
- `smoothing.py` - Class to smooth 2D array along one axis
- `utils.py` - General utilities for loading data, rotating images, adding noise

Dropout Network

- `train.py` - Training routine for dropout network
- `calibrate.py` - Evaluate on testing data

BNN
- `bnn.py` - Training script for BNN
- `ppc.py` - Evaluation script for running on testing data

### Examples, utilities

- `example_opts.yaml` - example options for BNN workflow
- `Makefile` - makefile that includes general workflows
- `run.sh` - Binbin's original run script, allows running by passing model as arg
- `tests.py` - Testing suite for functions using `pytest`

### Deprecated
- `eval.py` - (todo: deprecated?)
- `probabilities.py` - Calculate probabilities of dropout network, replaced by vectorized code
- `sampleAcc.py` - Calculate sample accuracy, replaced by vectorized code


## Resources:

[SVI introduction](http://pyro.ai/examples/svi_part_i.html)

### Literature

1. [Dropout: a simple way to prevent neural networks from overfitting](http://jmlr.org/papers/v15/srivastava14a.html)
2. [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf)
3. [Well-calibrated Model Uncertainty with Temperature Scaling for Dropout Variational Inference](http://bayesiandeeplearning.org/2019/papers/77.pdf)
4. [Everything you always wanted to know about pulls](http://physics.rockefeller.edu/luc/technical_reports/cdf5776_pulls.pdf)
5. [Evidential Deep Learning to Quantify Classification Uncertianty](https://arxiv.org/pdf/1806.01768.pdf)
6. [Work in ALCF to use a Stochastic Spiking NN to approximate Dropout as a means for UQ](https://ornlcda.github.io/icons2019/presentations/wycoff_icons2019_prez.pdf)
7. [Calibration of Model Uncertainty for Dropout Variational Inference](https://arxiv.org/pdf/2006.11584.pdf)
8. [Qualitative Analysis of Monte Carlo Dropout](https://arxiv.org/abs/2007.01720v1) (unpublished but interesting)
9. [Uncertainty Quantification in Deep Learning](https://www.inovex.de/blog/uncertainty-quantification-deep-learning/)
10. [Peculiar Velocity Estimation from Kinetic SZ Effect using Deep Neural Networks](https://arxiv.org/abs/2010.03762)


### Meetings

- Aug 11, 2020: https://indico.cern.ch/event/945285/  
- Aug 18, 2020: https://indico.cern.ch/event/947387/  

Updated: Jan 7, 2020 (tjb)