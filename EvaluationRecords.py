import torch
import numpy as np
from scipy import stats
import h5py
from utils import Timer
import arviz as az

class EvaluationRecords():
    """
    Class that records output of evaluation. Includes methods for final probability calculations.
    """

    def __init__(self, n_predictions: int, n_images: int, range_min: int, range_max: int) -> None:
        """Initalize object. Only sets parameters which are calculated in evaluation.

        Args:
            n_predictions (int): Number of dropout evaluations to perform (e.g. posterior samples)
            n_images (int): Number of images iterating over
        """
        self.n_predictions = n_predictions
        self.n_images = n_images
        self.probs = torch.zeros((n_predictions, n_images)) # Array saving probabilities
        self.matched_bool = torch.zeros((n_predictions, n_images), dtype=torch.bool) # Array saving true/false


        self.range_min = range_min
        self.range_max = range_max

        self.noDO_probs = torch.zeros(n_images) # Array saving probabilities
        self.noDO_matched_bool = torch.zeros(n_images, dtype=torch.bool)# Array saving true/false
        
        self.targets = torch.zeros(n_images, dtype=torch.int64)
        self.all_probabilities = torch.zeros(n_predictions, n_images, 10)
        self.all_noDropout_probabilities = torch.zeros(n_images, 10)
        self.all_max_evaluated_probabilities = torch.zeros(n_predictions, n_images)
        self.all_sec_evaluated_probabilities = torch.zeros(n_predictions, n_images)
        self.all_noDO_max_evaluated_probabilities = torch.zeros(n_images)
        self.all_noDO_sec_evaluated_probabilities = torch.zeros(n_images)

    def finalize(self) -> None:
        """Calculates and sets the remainder of parameters once evaluated.

        Raises:
            ValueError: Raised if ran before evaluations are recorded
        """


        if self.all_probabilities.mean() == 0.0:
            raise ValueError("Run evaluation on records before finalizing!")

        # Set probabilities 
        ## True probabilities - where the target and prediction match  
        self.true_probs = torch.gather(
            self.all_probabilities, 
            2,  
            self.targets.long().unsqueeze(-1).unsqueeze(0).repeat(self.n_predictions,1,1)
            ).squeeze(2) # appropriate dim
        
        ## False probabilities
        self.false_probs = torch.zeros(self.n_predictions, self.n_images)
        self.false_probs += self.matched_bool * self.all_sec_evaluated_probabilities
        self.false_probs += torch.logical_not(self.matched_bool) * self.probs

        target_bool = (torch.arange(0,10).repeat(self.n_images).reshape((self.n_images,10)) == self.targets.view(self.n_images,1))
        self.target_probs = self.all_probabilities[:,target_bool]
        self.not_target_probs = self.all_probabilities[:, ~target_bool].reshape((self.n_predictions, self.n_images, 9))

        # Do Same for no-Dropout, single evaluation case
        # NOTE - this won't load for BNN
        self.NoDro_true_probs =  torch.gather(
            self.all_noDropout_probabilities, 
            1, 
            self.targets.long().unsqueeze(-1)
            ).squeeze(1)
        self.NoDro_false_probs = torch.zeros(self.n_images)
        self.NoDro_false_probs += self.noDO_matched_bool * self.all_noDO_sec_evaluated_probabilities
        self.NoDro_false_probs += torch.logical_not(self.noDO_matched_bool) * self.noDO_probs

        # Evaluate final plotting quantities
        # - Each Tensor here is of length selected_images (dim=0 -> n_predictions)
        self.image_acc = torch.sum(self.matched_bool,dim=0) / float(self.n_predictions)
        self.image_acc_noDropout = self.noDO_matched_bool*1 # convert to 1/0 for true/false

        self.true_mu = torch.sum(self.true_probs, dim=0) / self.n_predictions
        self.false_mu = torch.sum(self.false_probs, dim=0) / self.n_predictions

        self.true_std = torch.std(self.true_probs, dim=0)
        self.false_std = torch.std(self.false_probs, dim=0)

        ### Using Mean Estimator
        ########################
        self.significance = ((self.true_mu - self.false_mu)/ torch.sqrt(( torch.std(self.true_probs, dim=0))**2 + (torch.std(self.false_probs, dim=0))**2))
        
        self.probability = stats.norm.cdf(self.significance)
        
        ####### for asymmetric errors
        lbound=0.158655254
        ubound=0.841344746
        self.true_CI = np.quantile(self.true_probs, [lbound, ubound], axis=0)
        self.false_CI = np.quantile(self.false_probs, [lbound, ubound], axis=0)

        true_std_asy = []
        significance_asy = []
        for i in range(self.range_max-self.range_min):
            if self.true_mu[i] > self.false_mu[i]:
                significance_asy+=[((self.true_mu[i]-self.false_mu[i]) / np.sqrt((self.true_mu[i]-self.true_CI[0,i])**2 + (self.false_mu[i]-self.false_CI[1,i])**2)).item()]
                true_std_asy += [(self.true_mu[i]-self.true_CI[0,i]).item()]
            else:
                significance_asy+=[((self.true_mu[i]-self.false_mu[i]) / np.sqrt((self.true_mu[i]-self.true_CI[1,i])**2 + (self.false_mu[i]-self.false_CI[0,i])**2)).item()]
                true_std_asy+=[(self.true_CI[1,i]-self.true_mu[i]).item()]

        self.true_std_asy = torch.tensor(true_std_asy)
        self.significance_asy = torch.tensor(significance_asy)
        self.probability_asy = stats.norm.cdf(significance_asy)
        self.image_pull=(self.NoDro_true_probs.numpy() - self.true_mu.numpy())/(self.true_std.numpy())
        self.image_pull_asy=(self.NoDro_true_probs.numpy() - self.true_mu.numpy())/(self.true_std_asy.numpy())
        self.image_pull_test=(self.NoDro_true_probs.numpy()-self.probability)/(self.true_std.numpy())
        
        ### Using Mode Estimator
        ########################
        self.max_true_posterior = stats.mode(np.around(self.true_probs, 2)).mode
        self.max_false_posterior = stats.mode(np.around(self.false_probs, 2)).mode
        self.modal_HPDI = az.hdi(self.true_probs.numpy(), hdi_prob=.68).T
        self.false_modal_HPDI = az.hdi(self.false_probs.numpy(), hdi_prob=.68).T
        ## TODO clean this 

        true_std_asy = []
        significance_asy = []
        true_std_asy_hpdi =[]
        for i in range(self.range_max-self.range_min):

            if self.max_true_posterior[:,i] > self.max_false_posterior[:,i]:
                significance_asy+=[((self.max_true_posterior[:,i]-self.max_false_posterior[:,i]) / np.sqrt((self.max_true_posterior[:,i]-self.modal_HPDI[0,i])**2 + (self.max_false_posterior[:,i]-self.false_modal_HPDI[1,i])**2)).item()]
                true_std_asy += [(self.max_true_posterior[:,i]-self.modal_HPDI[0,i]).item()]
            else:
                significance_asy+=[((self.max_true_posterior[:,i]-self.max_false_posterior[:,i]) / np.sqrt((self.max_true_posterior[:,i]-self.modal_HPDI[1,i])**2 + (self.max_false_posterior[:,i]-self.false_modal_HPDI[0,i])**2)).item()]
                true_std_asy+=[(self.modal_HPDI[1,i]-self.max_true_posterior[:,i]).item()]


        adj_std = np.std(self.max_true_posterior)
        self.image_pull_mode=(self.NoDro_true_probs.numpy() - self.max_true_posterior)/(true_std_asy)

        self.modal_significance = ((self.max_true_posterior - self.max_false_posterior)/ np.sqrt(( np.std(self.max_true_posterior))**2 + (np.std(self.max_false_posterior))**2))[0,:]
        self.modal_probability = stats.norm.cdf(self.modal_significance)
        
        self.modal_significance_asy = torch.tensor(significance_asy)
        self.modal_probability_asy = stats.norm.cdf(significance_asy)
        

    def save_to_file(self, hf: h5py.File) -> None:
        """Saves progress to an h5py file
        
        Args:
            hf (h5py.File): Output file
        """

        hf.create_dataset('image_acc', data=self.image_acc)
        hf.create_dataset('image_significance', data=self.significance)
        hf.create_dataset('image_probability',data=self.probability)
        hf.create_dataset('all_probabilities',data=self.all_probabilities, shape=self.all_probabilities.shape)
        hf.create_dataset('image_probability_asy',data=self.probability_asy)
        hf.create_dataset('image_true_probs', data=self.true_probs)
        hf.create_dataset('image_false_probs', data=self.false_probs)
        hf.create_dataset('true_image_std', data=self.true_std)
        hf.create_dataset('image_false_std', data=self.false_std)
        hf.create_dataset('image_true_mu', data=self.true_mu)
        hf.create_dataset('image_false_mu', data=self.false_mu)
        hf.create_dataset('image_target', data=self.targets)
        hf.create_dataset('image_max_prob', data=self.all_max_evaluated_probabilities)
        hf.create_dataset('image_sec_prob', data=self.all_sec_evaluated_probabilities)
        hf.create_dataset('NoDro_image_acc', data=self.image_acc_noDropout)
        hf.create_dataset('NoDro_image_max_prob', data=self.all_noDO_max_evaluated_probabilities)
        hf.create_dataset('NoDro_image_sec_prob', data=self.all_noDO_sec_evaluated_probabilities)
        hf.create_dataset('NoDro_image_true_probs', data=self.NoDro_true_probs)
        hf.create_dataset('NoDro_image_false_probs', data=self.NoDro_false_probs)
        hf.create_dataset('image_pull_1', data=self.image_pull)#no dropout score with dropout enabled average score
        hf.create_dataset('image_pull_2', data=self.image_pull_test)# no dropout score with probability 
        hf.create_dataset('image_pull_asy', data=self.image_pull_asy)
        hf.create_dataset('modal_probability', data=self.modal_probability)
        hf.create_dataset('target_probs', data=self.target_probs)
        hf.create_dataset('not_target_probs', data=self.not_target_probs)


