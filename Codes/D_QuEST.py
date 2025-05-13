import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import joblib
import json
import pywt
from scipy.signal import cwt, morlet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from PPG_Net.Feature_extractor import create_ppg_dataset

ml1 = RandomForestRegressor(n_estimators=100, random_state=42)
ml2 = RandomForestRegressor(n_estimators=100, random_state=42)

def ml_based_error(mlf, dlf, yt):
    ml1.fit(mlf, yt)
    ml2.fit(dlf, yt)
    
    yp_ml = ml1.predict(mlf).reshape(2, -1)
    yp_dl = ml2.predict(dlf).reshape(2, -1)
    
    err = mean_squared_error(yp_ml, yp_dl)
    return err

def supervised_loss(ppg_signals, y_true, dl_features):
    mlf, yt = create_ppg_dataset(ppg_signals, y_true[:, 0], y_true[:, 1], sampling_rate=125, clean=False)
    
    mlf = np.array(mlf)
    mlf = mlf / mlf.max(axis=0)
    yt = np.array(yt)
    
    err = ml_based_error(mlf, dl_features, yt)
    return err
