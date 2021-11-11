""" Utility class for loading files out of calibrate.py
--- Sept 2020 | tjb ---
"""
import h5py
import numpy as np

class DropoutEvaluations:
    """ Load files output by calibrate.py
    - Gets h5 saved data
    - Parses name string to get other relevant info
    """

    def __init__(self, filename):

        # Pull from the dataset
        try:
            with h5py.File(filename, "r") as source:
                self.acc = np.array(source["image_acc"])
                self.significance = np.array(source["image_significance"])
                self.probability = np.array(source["image_probability"])
                self.true_image_std = np.array(source["true_image_std"])
                self.false_std = np.array(source["image_false_std"])
                self.true_mu = np.array(source["image_true_mu"])
                self.false_mu = np.array(source["image_false_mu"])
                self.acc_noDropout = np.array(source["NoDro_image_acc"])
                self.target = np.array(source["image_target"])
                #self.pull =  np.array(source["image_pull"])
                self.true_probs = np.array(source["image_true_probs"])
                self.false_probs = np.array(source["image_false_probs"])
                self.all_probs = np.array(source["all_probabilities"])
        except KeyError as err:
            print(f"Unable to access file contents - failed on {err}")
            print("No longer trying to load from file! correct input.")

        # Scrape useful information from the filename
        unique_string = filename.split("/")[-1].split(".h5")[0]
        component_list = unique_string.split("_")
        self.trainmodel = component_list[1]
        self.trainpara = component_list[2]
        self.train_dropout_rate = round(int(component_list[2].split("dr0p")[1]) * 0.1, 2)
        self.epochs = int(component_list[3].split("ep")[1])
        self.posterior_evaluations = int(component_list[4].split("ev")[1])
        self.test_dropout_rate = round(int(component_list[5].split("test")[1]) * 0.1, 2)
        self.range_min = int(component_list[6].split("image")[1])
        self.range_max = int(component_list[7])
        self.evaluated_images = (self.range_max - self.range_min) + 1

    def label_to_method(self, label: str):
        the_map ={
            "image_true_probs" : self.true_probs,
            "image_false_probs" : self.false_probs,
            "all_probabilities" : self.all_probs
        }
        return the_map[label]
