import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import neurokit2 as nk
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

def mean_distance_between_systolic_peaks(ppg_signal, distance=5, prominence=0.025):
    systolic_peaks, _ = find_peaks(ppg_signal, distance=distance, prominence=prominence, height=0.7)
    
    # Calculate the differences between consecutive diastolic peak indices
    systolic_distances = np.diff(systolic_peaks)
    
    # Compute the mean of the distances
    mean_distance = np.mean(systolic_distances)
    return mean_distance

def find_diastolic_peaks(ppg_signal, distance=5, prominence=0.025):
    # Step 1: Detect the systolic peaks of the PPG signal
    systolic_peaks, _ = find_peaks(ppg_signal, distance=distance, prominence=prominence, height=0.7)
    
    # Step 2: Find the indices of the lowest points between systolic peaks
    lowest_points_indices = np.array([np.argmin(ppg_signal[systolic_peaks[i]:systolic_peaks[i+1]]) + systolic_peaks[i] 
                                      for i in range(len(systolic_peaks) - 1)])
    
    diastolic_peaks_indices = []
    diastolic_peaks_values = []

    # Step 3: Define the diastolic search zone between each systolic peak and the following lowest point
    for i in range(len(systolic_peaks) - 1):
        start_idx = systolic_peaks[i] + 1
        end_idx = lowest_points_indices[i]
        
        if start_idx < end_idx:  # Ensure the search zone is valid
            # Step 4: Find the peaks within the diastolic search zone
            zone_peaks, _ = find_peaks(ppg_signal[start_idx:end_idx], distance=distance//4, prominence=prominence/5)
            
            # Step 5: Determine the diastolic peak
            if len(zone_peaks) > 0:
                # Find the highest peak within the search zone
                zone_peak_values = ppg_signal[start_idx:end_idx][zone_peaks]
                max_peak_idx = zone_peaks[np.argmax(zone_peak_values)]
                diastolic_peaks_indices.append(start_idx + max_peak_idx)
                diastolic_peaks_values.append(ppg_signal[start_idx + max_peak_idx])
    
    # Convert to numpy arrays for easier handling
    diastolic_peaks_indices = np.array(diastolic_peaks_indices)
    diastolic_peaks_values = np.array(diastolic_peaks_values)
    
    return diastolic_peaks_indices, diastolic_peaks_values

def mean_distance_between_diastolic_peaks(ppg_signal):
    # Find diastolic peaks
    diastolic_peaks_indices, _ = find_diastolic_peaks(ppg_signal)
    
    # Calculate the differences between consecutive diastolic peak indices
    diastolic_distances = np.diff(diastolic_peaks_indices)
    
    # Compute the mean of the distances
    mean_distance = np.mean(diastolic_distances)
    
    return mean_distance

def find_ascending_and_descending_branch_times(ppg_signal, distance=5, prominence=0.025):
    # Find systolic peaks
    systolic_peaks, _ = find_peaks(ppg_signal, distance=distance, prominence=prominence, height=0.7)
    
    # Find diastolic peaks using your provided function
    diastolic_peaks_indices, diastolic_peaks_values = find_diastolic_peaks(ppg_signal, distance, prominence)
    
    # Find lowest points between systolic peaks
    lowest_points_indices = np.array([np.argmin(ppg_signal[systolic_peaks[i]:systolic_peaks[i+1]]) + systolic_peaks[i] 
                                      for i in range(len(systolic_peaks) - 1)])
    
    ABT = []
    DBT = []
    
    # Iterate through systolic peaks and corresponding lowest points
    for i in range(len(systolic_peaks) - 1):
        # For each pair of systolic peaks and lowest points
        systolic_peak = systolic_peaks[i+1]
        lowest_point = lowest_points_indices[i]
        
        # Find corresponding diastolic peak (assumption: diastolic peaks occur between systolic peaks)
        if i < len(diastolic_peaks_indices):
            diastolic_peak = diastolic_peaks_indices[i]
        else:
            continue
        
        # Calculate ABT and DBT
        abt = systolic_peak - lowest_point
        dbt = lowest_point - diastolic_peak
        
        ABT.append(abs(abt))
        DBT.append(abs(dbt))
    
    ABT = np.array(ABT)
    DBT = np.array(DBT)
    
    return np.mean(ABT), np.mean(DBT)


def get_ppg_to_hr(ppg_signal, sampling_rate=125):
    # Process the PPG signal
    df, info = nk.ppg_process(ppg_signal, sampling_rate=sampling_rate)
    
    # Calculate the mean heart rate from the processed data
    heart_rate_bpm = np.mean(df["PPG_Rate"])

    return heart_rate_bpm


def create_ppg_dataset(ppg_signals, sbp_values, dbp_values, sampling_rate=125, clean=True, progress=False):
    # Initialize an empty list to store the results
    data = []

    # Loop through each PPG signal
    for i, ppg in enumerate(ppg_signals):

        if progress:
            interval = int(len(ppg_signals) / 10)

            if i > 0 and (i % interval == 0 or i == (len(ppg_signals) - 1)):
                print(f"Processing signal {int((i+1)/len(ppg_signals) * 100)} %")

        ppg_signal = np.array(ppg).reshape(1000)

        # Extract features using the get_ppg_signal_features function
        heart_rate_bpm = get_ppg_to_hr(ppg_signal, sampling_rate=sampling_rate)
        
        # Extract the mean distance between diastolic peaks
        diastolic_peak_interval = mean_distance_between_diastolic_peaks(ppg_signal)
        mean_systolic_interval = mean_distance_between_systolic_peaks(ppg_signal)

        abt, dbt = find_ascending_and_descending_branch_times(ppg_signal)
        
        # Get the corresponding SBP and DBP values
        sbp = sbp_values[i]
        dbp = dbp_values[i]
        
        # Append the results to the data list
        data.append([heart_rate_bpm, mean_systolic_interval, diastolic_peak_interval, abt, dbt, sbp, dbp])

    # Create a DataFrame from the data list
    df = pd.DataFrame(data, columns=["HR", "MSI", "MDI", "ABT", "DBT", "SBP", "DBP"])

    if clean:
        df.dropna(axis=0, inplace=True)
        df = df[df["MSI"] <= 180]
        df = df[df["DBT"] <= 80]

    else:
        df.fillna(0, inplace=True)

    # Separate features (X) and labels (y)
    X = df.drop(columns=["SBP", "DBP"])
    y = df[["SBP", "DBP"]]

    return X, y


dir = r"D:\Rafi_Thesis\Final_Codes"

def manage_data(X=None, y=None, save_dir=None, flag=True):
    if flag:
        # Ensure that the indices of X and y align correctly for merging
        if X is None or y is None:
            raise ValueError("Both X and y must be provided to save the data.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of rows in X and y must be the same for merging.")
        
        # Merge DataFrames along columns
        merged_df = pd.concat([X, y], axis=1)
        
        # Save the merged DataFrame to a CSV file
        if save_dir is None:
            raise ValueError("Output file name must be provided when saving.")
        merged_df.to_csv(save_dir + "\\" + "ML_features_data.csv", index=False)
        print("DataFrame saved")
        
    else:
        # Load DataFrame from CSV file
        if save_dir is None:
            raise ValueError("Input file name must be provided when loading.")
        loaded_df = pd.read_csv(save_dir + "\\" + "ML_features_data.csv")
        
        # Split the loaded DataFrame into X and y
        # Assumes that the last column is y and the rest are X
        X = loaded_df.drop(columns=["SBP", "DBP"])
        y = loaded_df[["SBP", "DBP"]]
        
        return X, y