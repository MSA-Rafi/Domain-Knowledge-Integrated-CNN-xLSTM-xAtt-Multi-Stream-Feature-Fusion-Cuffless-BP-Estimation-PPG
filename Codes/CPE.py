# BP Filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

def conv_moving_average(ppg_signal, window_size=75, show_plot=False):
    # Apply moving average filter to estimate baseline
    baseline = np.convolve(ppg_signal, np.ones(window_size)/window_size, mode='same')
    # Subtract the baseline from the original signal
    ppg_signal_corrected = ppg_signal - baseline
    
    if show_plot:
        # Plotting the original and corrected signals
        plt.figure(figsize=(10, 5))
        plt.plot(ppg_signal, label='Original Signal', alpha=0.7)
        plt.plot(ppg_signal_corrected, label='Corrected Signal')
        plt.title(f'convolutional moving average')
        plt.xlabel('Sample Index')
        plt.ylabel('Signal Amplitude')
        plt.legend()
        plt.show()

    ppg_signal_corrected = savgol_filter(ppg_signal_corrected, window_length=51, polyorder=4)

    min_value = np.min(ppg_signal_corrected)
    ppg_signal_corrected = ppg_signal_corrected + abs(min_value)
    ppg_signal_corrected = ppg_signal_corrected / max(ppg_signal_corrected)
    return ppg_signal_corrected