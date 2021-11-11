MODEL = CNNWeak
DO_RATE = 2 # Default used in paper
EPOCHS = 300
IMAGES = 10000


setup:
	mkdir -p ../output/trainModule
	mkdir -p ../output/testModule
	mkdir -p ../output/testResult/uncertainty/

# --  General Workflow --
# -----------------------
dropout_workflow:
	python3 train.py $(MODEL) $(DO_RATE) $(EPOCHS) $(IMAGES)
	# Note - uses same dropout rate in training and testing
	python3 calibrate.py $(MODEL) dr0p$(DO_RATE)_ep$(EPOCHS)_ev$(IMAGES) $(DO_RATE) 0 $(IMAGES)

# This assumes that the names in example_opts.yaml are used
bnn_workflow: bnn.py ppc.py 
	python3 bnn.py example_opts.yaml
	python3 ppc.py example_opts.yaml


# --  Single Steps --
# -----------------------
train_dropout: train.py
	python3 train.py $(MODEL) $(DO_RATE) $(EPOCHS) $(IMAGES)
evaluate_dropout: calibrate.py
	python3 calibrate.py $(MODEL) dr0p$(DO_RATE)_ep$(EPOCHS)_ev$(IMAGES) $(DO_RATE) 0 $(IMAGES)

train_bnn: bnn.py
	python3 bnn.py example_opts.yaml
evaluate_bnn: bnn.py bcnn.pyro
	python3 ppc.py example_opts.yaml



# --  Varied Dropout Rate Studies --
# ----------------------------------

DROPOUTS = 1 2 3 4 5 6 7 8 # Used for a suite of training

## Train dropout = test dropout
full_varied_dropout_study: plotting/varied_rates.py
	$(foreach val, $(DROPOUTS), python3 train.py $(MODEL) $(val) 300 10000;)
	$(foreach val, $(DROPOUTS), python3 calibrate.py $(MODEL) dr0p$(val)_ep300_ev10000 $(val) 10000 0 10000;)
	python3 $< $(foreach val, $(DROPOUTS), ../output/testResult/uncertainty/cali_$(MODEL)_dr0p$(val)_ep300_ev10000_test$(val)_image1_10000.h5)

## Train dropout = 0; test dropout varies
zero_train_study: plotting/varied_rates.py
	python3 train.py $(MODEL) 0 300 10000;
	$(foreach val, $(DROPOUTS), python3 calibrate.py $(MODEL) dr0p0_ep300_ev10000 $(val) 10000 0 10000;)
	python3 $< $(foreach val, $(DROPOUTS), ../output/testResult/uncertainty/cali_$(MODEL)_dr0p0_ep300_ev10000_test$(val)_image1_10000.h5)

## Train dropout = 0.20 ; test dropout varies
const_train_study: plotting/varied_rates.py
	$(foreach val, $(DROPOUTS), python3 calibrate.py $(MODEL) dr0p2_ep300_ev10000 $(val) 10000 0 10000;)
	python3 $< $(foreach val, $(DROPOUTS), ../output/testResult/uncertainty/cali_$(MODEL)_dr0p2_ep300_ev10000_test$(val)_image1_10000.h5)



## --- Individual steps ---
varied_train: train.py
	$(foreach val, $(DROPOUTS), python3 train.py $(MODEL) $(val) 300 10000;)

varied_calibrate: calibrate.py
	$(foreach val, $(DROPOUTS), python3 calibrate.py $(MODEL) dr0p$(val)_ep300_ev10000 $(val) 10000 0 10000;)

varied_plots: plotting/varied_rates.py
	python3 $< $(foreach val, $(DROPOUTS), ../output/testResult/uncertainty/cali_$(MODEL)_dr0p$(val)_ep300_ev10000_image1_10000.h5)

varied_dropout: plotting/varied_rates.py plotting/dropout_evaluations.py
	python3 $< ../output/testResult/uncertainty/cali_CNNWeak_dr0p*_ep300_ev10000_image1_10000.h5


