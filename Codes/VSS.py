# BP Filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def Variance_Filter(data, threshold_distance_variance = 10.0, threshold_height_variance = 0.0025):

    threshold_height = 0.45

    # Find systolic peaks
    signal = np.array(data[125:1125])
    signal = signal.reshape(1000)
    signal = signal / max(signal)
    peaks, _ = find_peaks(signal, height=threshold_height)
    flag = True

    # Check if all peaks meet the height threshold
    if not np.all(signal[peaks] >= 0.7):
        flag = False

    # Calculate heights of peaks
    peak_heights = signal[peaks]

    # Calculate variance of peak heights
    if len(peak_heights) > 1:
        height_variance = np.var(peak_heights)
    else:
        height_variance = 0  # If there's only one peak, variance is 0

    # Calculate distances between consecutive peaks
    distances = []
    for p in range(len(peaks) - 1):
        d = abs(peaks[p] - peaks[p+1])
        distances.append(d)
    
    # Calculate variance of distances
    if len(distances) > 1:
        distance_variance = np.var(distances)
    else:
        distance_variance = 0  # If there's only one distance, variance is 0

    # print(f"height variance: {height_variance: .4f} and distance variance: {distance_variance: .4f}")

    # Check if variance is within thresholds
    if height_variance <= threshold_height_variance and distance_variance <= threshold_distance_variance and np.all(signal[peaks] >= 0.75):
        #quality = nk.ppg_quality(signal, sampling_rate=125, method="templatematch")
        #return np.mean(quality)
        return True
    else:
        return False
