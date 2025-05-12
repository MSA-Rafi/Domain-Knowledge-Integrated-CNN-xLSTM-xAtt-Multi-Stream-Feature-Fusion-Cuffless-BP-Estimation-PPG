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


def ppg_to_scalogram(ppg, downsample_factor=4, scales=np.arange(1, 128), wavelet='morl', target_size=(250, 250)):
    ppg_batch = ppg[:, ::downsample_factor]
    # Ensure the input is a 2D tensor (batch_size, sequence_length)
    if ppg_batch.dim() > 2:
        ppg_batch = ppg_batch.view(ppg_batch.size(0), -1)
    
    # Convert to NumPy for easier manipulation
    ppg_batch_np = ppg_batch.cpu().numpy()

    batch_size, sequence_length = ppg_batch_np.shape
    scalogram_images = []

    for ppg in ppg_batch_np:
        # Compute the continuous wavelet transform (CWT)
        coefficients, _ = pywt.cwt(ppg, scales, wavelet)
        
        # Normalize the coefficients
        scalogram = np.abs(coefficients)
        scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-6)

        # Convert the scalogram to a 2D image
        scalogram_image = torch.tensor(scalogram, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Resize the scalogram image to the target size
        scalogram_image_resized = F.interpolate(scalogram_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        # Repeat the single-channel image to create 3-channel RGB images
        scalogram_image_rgb = scalogram_image_resized.repeat(3, 1, 1)  # Shape: (3, height, width)

        scalogram_images.append(scalogram_image_rgb)

    # Stack the Scalogram images to create a batch
    scalogram_images = torch.stack(scalogram_images)

    return scalogram_images


def ppg_to_mtf(ppg, n_bins=10, downsample_factor=4, target_size=(250, 250)):
    ppg_batch = ppg[:, ::downsample_factor]
    # Ensure the input is a 2D tensor (batch_size, sequence_length)
    if ppg_batch.dim() > 2:
        ppg_batch = ppg_batch.view(ppg_batch.size(0), -1)
    
    # Convert to NumPy for easier manipulation
    ppg_batch_np = ppg_batch.cpu().numpy()

    batch_size, sequence_length = ppg_batch_np.shape
    mtf_images = []

    for ppg in ppg_batch_np:
        # Quantize the PPG signal into discrete bins
        quantized_ppg = np.digitize(ppg, np.linspace(ppg.min(), ppg.max(), n_bins))

        # Compute the transition matrix
        transition_matrix = np.zeros((n_bins, n_bins))

        for i in range(sequence_length - 1):
            current_state = quantized_ppg[i]
            next_state = quantized_ppg[i + 1]
            transition_matrix[current_state - 1, next_state - 1] += 1

        # Normalize the transition matrix to get probabilities
        transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True) + 1e-6

        # Convert the transition matrix to a 2D image
        mtf_image = torch.tensor(transition_matrix, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        # Resize the MTF image to the target size
        mtf_image_resized = F.interpolate(mtf_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        # Repeat the single-channel image to create 3-channel RGB images
        mtf_image_rgb = mtf_image_resized.repeat(3, 1, 1)  # Shape: (3, height, width)

        mtf_images.append(mtf_image_rgb)

    # Stack the MTF images to create a batch
    mtf_images = torch.stack(mtf_images)

    return mtf_images


def ppg_to_image(ppg_batch, downsample_factor=4, target_size=(250, 250)):
    # Downsample the PPG signals
    downsampled_ppg_batch = ppg_batch[:, ::downsample_factor]
    
    # Normalize the PPG signals to range [0, 1]
    #downsampled_ppg_batch = (downsampled_ppg_batch - downsampled_ppg_batch.min(dim=1, keepdim=True)[0]) / (downsampled_ppg_batch.max(dim=1, keepdim=True)[0] - downsampled_ppg_batch.min(dim=1, keepdim=True)[0] + 1e-6)

    # Reshape the PPG signals to create a 2D grid representation
    ppg_images = downsampled_ppg_batch.unsqueeze(1)  # Add channel dimension
    ppg_images = F.interpolate(ppg_images, size=target_size, mode='bilinear', align_corners=False)
    
    # Repeat the single-channel image to create 3-channel RGB images
    ppg_images = ppg_images.repeat(1, 3, 1, 1)  # Shape: (batch_size, 3, height, width)

    return ppg_images
