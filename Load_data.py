import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
from scipy.io import loadmat
import os
from PIL import Image
from scipy.interpolate import interp1d
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import json
import neurokit2 as nk

database_dir = r"H:\Codes\PPG_to_BP\Extracted_data\Extracted_Data_Final\MIMIC\small_data_50_subjects\\2_variance_filter_on_bp_filtered_data\Fold_02"
        
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    

def load_preprocessed_data(name, file=None, save=True):
    dir = database_dir + "\\" + name + ".json"
    if save:
        with open(dir, "w") as outfile:
            json.dump(file, outfile, cls=NumpyEncoder)
    else:
        with open(dir, "r") as outfile:
            file = json.load(outfile)
            return file