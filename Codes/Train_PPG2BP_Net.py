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


from PPG_Net.PPG2BP_Net import PPGNet
from PPG_Net.utils import pearson_corr, HuberLoss
from PPG_Net.D_QuEST import supervised_loss
from PPG_Net.Image_transform import ppg_to_scalogram, ppg_to_mtf, ppg_to_image
from PPG_Net.Load_data import load_preprocessed_data

###########################################################################################################################################
# Load Data
# Data Saving or Loading

ppg_train = load_preprocessed_data("pre_processed_ppg_train", save=False)
abp_train = load_preprocessed_data("pre_processed_abp_train", save=False)
y_train = load_preprocessed_data("pre_processed_y_train", save=False)
# age_train = save_load_json("pre_processed_age_train", save=False)
# gender_train = save_load_json("pre_processed_gender_train", save=False)

ppg_test = load_preprocessed_data("pre_processed_ppg_test", save=False)
abp_test = load_preprocessed_data("pre_processed_abp_test", save=False)
y_test = load_preprocessed_data("pre_processed_y_test", save=False)
# age_test = save_load_json("pre_processed_age_test", save=False)
# gender_test = save_load_json("pre_processed_gender_test", save=False)

ppg_train = np.array(ppg_train).reshape(-1, 1000, 1)
abp_train = np.array(abp_train).reshape(-1, 1000, 1)
y_train = np.array(y_train).reshape(-1, 2)
# age_train = np.array(age_train).reshape(-1, 1)
# gender_train = np.array(gender_train).reshape(-1, 1)

ppg_test = np.array(ppg_test).reshape(-1, 1000, 1)
abp_test = np.array(abp_test).reshape(-1, 1000, 1)
y_test = np.array(y_test).reshape(-1, 2)
# age_test = np.array(age_test).reshape(-1, 1)
# gender_test = np.array(gender_test).reshape(-1, 1)

###########################################################################################################################################
# Ignore all warnings
warnings.filterwarnings("ignore")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PPGNet().to(device)
ch_num = 2  # Assuming y_train is a 2D array
criterion_mse = nn.MSELoss()
criterion_huber = HuberLoss()
criterion_mae = nn.L1Loss()  # MAE loss


# Assuming ppg_train and y_train are available
gamma = nn.Parameter(torch.tensor(0.3, device=device, requires_grad=True))
optimizer = optim.Adam(list(model.parameters()) + [gamma], lr=0.001)  # Set an initial learning rate

# optimizer = optim.Adam(model.parameters() + [gamma], lr=0.001)

# Add a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

epochs = 35
patience = 7  # Number of epochs with no improvement after which training will be stopped
best_val_loss = float('inf')
no_improvement = 0
# gamma = 0.2
dl_loss = []
ml_loss = []

# File path to save the best model
best_model_path = "detailed_err_weights.pth"

# Move data to GPU
ppg_train_tensor = torch.Tensor(ppg_train).unsqueeze(1).squeeze(-1).to(device)  # Adjust the input shape
abp_train_tensor = torch.Tensor(abp_train).unsqueeze(1).squeeze(-1).to(device)
y_train_tensor = torch.Tensor(y_train).to(device)
dataset = TensorDataset(ppg_train_tensor, abp_train_tensor, y_train_tensor)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Training loop
for epoch in range(epochs):
    # gamma = gamma * 10 ** (-epoch / 35)
    g = gamma.item()
    print(f'Epoch: {epoch+1}/{epochs}', end=" ")
    print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}', end=" ")
    print(f'Coupling Factor: {gamma.item()}')
    # Training phase
    model.train()  # Set the model to training mode
    train_mse, train_mae, train_pearson, train_ml = 0.0, 0.0, 0.0, 0.0
    
    i = 1
    for data in train_loader:       
        print("-", end="")
        # g = gamma.cpu().detach().numpy()
        input_ppg, input_abp, targets = data
        
        #print("Getting image Data")
        inp_img = ppg_to_image(input_ppg)
        inp_mtf = ppg_to_mtf(input_ppg)
        inp_scale = ppg_to_scalogram(input_ppg)

        inp_img = inp_img.to(device)
        inp_mtf = inp_mtf.to(device)
        inp_scale = inp_scale.to(device)
        input_abp, targets = input_abp.to(device), targets.to(device)  # Move data to GPU

        outputs, sup = model(inp_img, inp_mtf, inp_scale, input_abp)
        
        loss_mse = criterion_mse(outputs, targets)
        loss_huber = criterion_huber(outputs, targets)
        loss_mae = criterion_mae(outputs, targets)

        qpl = supervised_loss(input_ppg.cpu().numpy(), targets.cpu().numpy(), sup.cpu().detach().numpy())
        # loss = loss_mse + qpl * gamma.cpu().detach().numpy()
        

        if i == len(train_loader):
            loss = loss_huber + qpl * gamma
        else:
            loss = loss_huber + qpl * g
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            gamma.clamp_(min=1e-6)

        # Calculate Pearson correlation for this batch
        pearson = pearson_corr(outputs, targets)
        
        # Accumulate the losses and Pearson correlation
        train_mse += loss_mse.item()
        train_mae += loss_mae.item()
        train_pearson += pearson.item()
        train_ml += qpl.item() * gamma.item()

        if i % 100 == 0:
            print()
            print(f"        batch {i}: CF: {gamma.item()}, Avg ML: {train_ml/i} and Avg mse: {train_mse/i}")
        i += 1
    
    # Calculate average training metrics for the epoch
    train_mse /= len(train_loader)
    train_mae /= len(train_loader)
    train_pearson /= len(train_loader)
    train_ml /= len(train_loader)

    dl_loss.append(train_mse)
    ml_loss.append(train_ml)

    print()
    print(f'Train - MSE: {train_mse:.8f}, MAE: {train_mae:.8f}, Pearson: {train_pearson:.8f}, ML: {train_ml:.8f}')

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_mse, val_mae, val_pearson = 0.0, 0.0, 0.0

    with torch.no_grad():  # Disable gradient calculation for validation
        for data in val_loader:
            input_ppg, input_abp, targets = data

            #print("Getting image Data")
            inp_img = ppg_to_image(input_ppg)
            inp_mtf = ppg_to_mtf(input_ppg)
            inp_scale = ppg_to_scalogram(input_ppg)
            #print(f"Image data loaded with shape inp2 = {inp2.shape}, input = {inputs.shape}, target = {targets.shape}")

            inp_img = inp_img.to(device)
            inp_mtf = inp_mtf.to(device)
            inp_scale = inp_scale.to(device)
            input_abp, targets = input_abp.to(device), targets.to(device)  # Move data to GPU

            outputs, _ = model(inp_img, inp_mtf, inp_scale, input_abp)
            
            loss_mse = criterion_mse(outputs, targets)
            loss_mae = criterion_mae(outputs, targets)
            
            # Calculate Pearson correlation for this batch
            pearson = pearson_corr(outputs, targets)
            
            # Accumulate the losses and Pearson correlation
            val_mse += loss_mse.item()
            val_mae += loss_mae.item()
            val_pearson += pearson.item()

    # Calculate average validation metrics for the epoch
    val_mse /= len(val_loader)
    val_mae /= len(val_loader)
    val_pearson /= len(val_loader)
    
    print(f'Val   - MSE: {val_mse:.8f}, MAE: {val_mae:.8f}, Pearson: {val_pearson:.8f}')

    # Save the model if it has the best validation loss so far
    if val_mse < best_val_loss:
        best_val_loss = val_mse
        no_improvement = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved with validation MSE: {best_val_loss:.8f}")
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print("Early stopping triggered.")
            break

    # Step the scheduler after each epoch
    scheduler.step()

# plt.plot(dl_loss)
# plt.show()
# plt.plot(ml_loss)
# plt.show()

# After training, load the best model
model.load_state_dict(torch.load(best_model_path))
print(f"Best model loaded from {best_model_path}")

# Define the test dataset and loader (assuming test data is available)
ppg_test_tensor = torch.Tensor(ppg_test).unsqueeze(1).squeeze(-1).to(device)  # Adjust the input shape
abp_test_tensor = torch.Tensor(abp_test).unsqueeze(1).squeeze(-1).to(device)
y_test_tensor = torch.Tensor(y_test).to(device)
test_dataset = TensorDataset(ppg_test_tensor, abp_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Function for evaluating the model on test data
def evaluate_model_on_test_data(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    test_mse, test_mae, test_pearson = 0.0, 0.0, 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():  # Disable gradient calculation
        for data in test_loader:
            input_ppg, input_abp, targets = data

            #print("Getting image Data")
            inp_img = ppg_to_image(input_ppg)
            inp_mtf = ppg_to_mtf(input_ppg)
            inp_scale = ppg_to_scalogram(input_ppg)
            #print(f"Image data loaded with shape inp2 = {inp2.shape}, input = {inputs.shape}, target = {targets.shape}")

            inp_img = inp_img.to(device)
            inp_mtf = inp_mtf.to(device)
            inp_scale = inp_scale.to(device)
            input_abp, targets = input_abp.to(device), targets.to(device)  # Move data to GPU

            outputs, _ = model(inp_img, inp_mtf, inp_scale, input_abp)

            loss_mse = mean_squared_error(targets.cpu().numpy(), outputs.cpu().numpy())
            loss_mae = mean_absolute_error(targets.cpu().numpy(), outputs.cpu().numpy())

            # Calculate Pearson correlation
            pearson = pearson_corr(outputs, targets)

            # Accumulate the losses and Pearson correlation
            test_mse += loss_mse
            test_mae += loss_mae
            test_pearson += pearson.item()

            # Store predictions and targets for further analysis if needed
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Calculate average metrics
    test_mse /= len(test_loader)
    test_mae /= len(test_loader)
    test_pearson /= len(test_loader)
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Save performance results
    results = {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_pearson": test_pearson}

    with open('test_performance_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Test - MSE: {test_mse:.8f}, MAE: {test_mae:.8f}, Pearson: {test_pearson:.8f}')

# Load the best model
model.load_state_dict(torch.load(best_model_path))

# Evaluate the model on test data
evaluate_model_on_test_data(model, test_loader)
