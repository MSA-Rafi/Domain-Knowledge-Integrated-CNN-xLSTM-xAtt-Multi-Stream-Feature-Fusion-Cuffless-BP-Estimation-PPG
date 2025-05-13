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

# Define the PPGNet model
class Pipeline1a(nn.Module):
    def __init__(self):
        super(Pipeline1a, self).__init__()
        inception_v3 = models.inception_v3(pretrained=True)
        
        # Remove the final fully connected layers
        self.feature_extractor = nn.Sequential(
            inception_v3.Conv2d_1a_3x3,
            inception_v3.Conv2d_2a_3x3,
            inception_v3.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception_v3.Conv2d_3b_1x1,
            inception_v3.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception_v3.Mixed_5b,
            inception_v3.Mixed_5c,
            inception_v3.Mixed_5d,
            inception_v3.Mixed_6a,
            inception_v3.Mixed_6b,
            inception_v3.Mixed_6c,
            inception_v3.Mixed_6d,
            inception_v3.Mixed_6e,
            inception_v3.Mixed_7a,
            inception_v3.Mixed_7b,
            inception_v3.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        
        # Add a fully connected layer to map to the desired output size
        self.fc = nn.Linear(inception_v3.fc.in_features, 128)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
    
class Pipeline1b(nn.Module):
    def __init__(self):
        super(Pipeline1b, self).__init__()
        inception_v3 = models.inception_v3(pretrained=True)
        
        # Remove the final fully connected layers
        self.feature_extractor = nn.Sequential(
            inception_v3.Conv2d_1a_3x3,
            inception_v3.Conv2d_2a_3x3,
            inception_v3.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception_v3.Conv2d_3b_1x1,
            inception_v3.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception_v3.Mixed_5b,
            inception_v3.Mixed_5c,
            inception_v3.Mixed_5d,
            inception_v3.Mixed_6a,
            inception_v3.Mixed_6b,
            inception_v3.Mixed_6c,
            inception_v3.Mixed_6d,
            inception_v3.Mixed_6e,
            inception_v3.Mixed_7a,
            inception_v3.Mixed_7b,
            inception_v3.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        
        # Add a fully connected layer to map to the desired output size
        self.fc = nn.Linear(inception_v3.fc.in_features, 128)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
    
class Pipeline1c(nn.Module):
    def __init__(self):
        super(Pipeline1c, self).__init__()
        inception_v3 = models.inception_v3(pretrained=True)
        
        # Remove the final fully connected layers
        self.feature_extractor = nn.Sequential(
            inception_v3.Conv2d_1a_3x3,
            inception_v3.Conv2d_2a_3x3,
            inception_v3.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception_v3.Conv2d_3b_1x1,
            inception_v3.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
            inception_v3.Mixed_5b,
            inception_v3.Mixed_5c,
            inception_v3.Mixed_5d,
            inception_v3.Mixed_6a,
            inception_v3.Mixed_6b,
            inception_v3.Mixed_6c,
            inception_v3.Mixed_6d,
            inception_v3.Mixed_6e,
            inception_v3.Mixed_7a,
            inception_v3.Mixed_7b,
            inception_v3.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten()
        )
        
        # Add a fully connected layer to map to the desired output size
        self.fc = nn.Linear(inception_v3.fc.in_features, 128)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x
    
class Pipeline2(nn.Module):
    def __init__(self):
        super(Pipeline2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=3)
        self.maxpool1 = nn.MaxPool1d(kernel_size=4)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=3)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)

    def forward(self, x):
        #x = x.permute(0, 2, 1)  # Change shape from (batch, seq_len, channels) to (batch, channels, seq_len)
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        
        x = x.permute(0, 2, 1)  # Change shape back to (batch, seq_len, features) for LSTM
        x, _ = self.lstm1(x)
        x, (h_n, c_n) = self.lstm2(x)  # h_n is the last hidden state
        
        return h_n.squeeze(0)  # h_n shape is (1, batch_size, hidden_size), so squeeze the 0th dimension

###################################################################################
# class Pipeline2(nn.Module):
#     def __init__(self):
#         super(Pipeline2, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=3)
#         self.maxpool1 = nn.MaxPool1d(kernel_size=4)
#         self.bn1 = nn.BatchNorm1d(num_features=64)
        
#         self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=9, stride=3)
#         self.maxpool2 = nn.MaxPool1d(kernel_size=4)
#         self.bn2 = nn.BatchNorm1d(num_features=128)
        
#         # Replacing LSTMs with GRUs
#         self.gru1 = nn.GRU(input_size=128, hidden_size=64, batch_first=True)
#         self.gru2 = nn.GRU(input_size=64, hidden_size=128, batch_first=True)

#     def forward(self, x):
#         # x = x.permute(0, 2, 1)  # This is not needed as Conv1D works with (batch, channels, seq_len)
#         x = self.conv1(x)
#         x = self.maxpool1(x)
#         x = self.bn1(x)
        
#         x = self.conv2(x)
#         x = self.maxpool2(x)
#         x = self.bn2(x)
        
#         x = x.permute(0, 2, 1)  # Change shape back to (batch, seq_len, features) for GRU
#         x, _ = self.gru1(x)  # Pass through the first GRU layer
#         x, h_n = self.gru2(x)  # h_n is the last hidden state
        
#         return h_n.squeeze(0)  # h_n shape is (1, batch_size, hidden_size), so squeeze the 0th dimension
###################################################################################

class Temp_Attention(nn.Module):
    def __init__(self, input_dim):
        super(Temp_Attention, self).__init__()
        self.Q = nn.Parameter(torch.rand(input_dim[-1], 1))

    def forward(self, x):
        scores = torch.matmul(x, self.Q)
        scores = F.silu(scores)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.tanh(attn_weights)
        output = x * attn_weights
        return output

class Pipeline3(nn.Module):
    def __init__(self):
        super(Pipeline3, self).__init__()
        # Bidirectional LSTM with 64 units (output size will be 64*2=128)
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, batch_first=True, bidirectional=True)
        self.attention = Temp_Attention((None, 1000, 128))
        # LSTM with 256 units
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, bidirectional=False)
        # Final LSTM with 128 units (no return sequences)
        self.lstm3 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=False)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, channels)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        # print(x.shape)
        x = self.attention(x)
        # Feed the attention output into the second LSTM
        x, _ = self.lstm2(x)
        
        # Final LSTM layer
        x, _ = self.lstm3(x)
        
        # Returning the last output
        return x[:, -1, :]  # Return the final output for each sequence in the batch

###################################################################################
# class Pipeline3(nn.Module):
#     def __init__(self):
#         super(Pipeline3, self).__init__()
#         # Bidirectional GRU with 64 units (output size will be 64*2=128)
#         self.gru1 = nn.GRU(input_size=1, hidden_size=64, batch_first=True, bidirectional=True)
#         self.attention = Temp_Attention((None, 1000, 128))
#         # GRU with 256 units
#         self.gru2 = nn.GRU(input_size=128, hidden_size=256, batch_first=True, bidirectional=False)
#         # Final GRU with 128 units (no return sequences)
#         self.gru3 = nn.GRU(input_size=256, hidden_size=128, batch_first=True, bidirectional=False)

#     def forward(self, x):
#         # Input shape: (batch_size, seq_len, channels)
#         x = x.permute(0, 2, 1)  # Change shape to (batch_size, seq_len, channels)
#         x, _ = self.gru1(x)  # Pass through the first GRU layer
#         # Apply temporal attention
#         x = self.attention(x)
#         # Pass the attention output into the second GRU
#         x, _ = self.gru2(x)
#         # Pass through the final GRU layer
#         x, _ = self.gru3(x)
#         # Return the last output for each sequence in the batch
#         return x[:, -1, :]  # Return the final output of the sequence
###################################################################################

class FusionNet(nn.Module):
    def __init__(self, dim1, dim2):
        super(FusionNet, self).__init__()
        
        self.fc1 = nn.Linear(dim1, 128)
        self.fc2 = nn.Linear(dim2, 128)

        self.fcs = nn.Linear(128 + 128, 128)

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=1, padding='same')
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=2, padding='same')
        
        self.bilstm1 = nn.LSTM(input_size=2, hidden_size=8, num_layers=1, batch_first=False, bidirectional=True)
        self.bilstm2 = nn.LSTM(input_size=16, hidden_size=1, num_layers=1, batch_first=False, bidirectional=False)

        self.fcf = nn.Linear(128 + 128 + 128, 128)

    def forward(self, x1, x2):
        x1 = self.fc1(x1).unsqueeze(1)
        x2 = self.fc2(x2).unsqueeze(1)
        xp = torch.cat((x1, x2), dim=1)

        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        
        xs = torch.cat((x1, x2), dim=1)
        xs = F.silu(self.fcs(xs))

        xconv = self.conv1(xp)
        xconv = self.conv2(xconv)
        xconv = xconv.squeeze(1)

        xp = xp.permute(0, 2, 1)
        xlstm, _ = self.bilstm1(xp)
        xlstm, _ = self.bilstm2(xlstm)
        xlstm = xlstm.squeeze(2)
        
        xconv = F.silu(xconv)
        xlstm = F.silu(xlstm)

        xx = torch.cat((xconv, xlstm), dim=1)
        x = torch.cat((xs, xconv, xlstm), dim=1)
        # x = self.fcf(x)

        return x, xx


class PPGNet(nn.Module):
    def __init__(self):
        super(PPGNet, self).__init__()
        self.cross_mdin_att = Temp_Attention((None, 3, 128))
        self.cross_ConvxLSTM_att = Temp_Attention((None, 2, 128))
        self.spat_att = Temp_Attention((None, 128, 1))

        self.layer_norm = nn.LayerNorm(128)
        
        self.pipeline1a = Pipeline1a() # PPG Image
        self.pipeline1b = Pipeline1b() # MTF Image
        self.pipeline1c = Pipeline1c() # Scalogram
        
        self.pipeline2 = Pipeline2() # ABP
        self.pipeline3 = Pipeline3() # ABP

        self.fusion = FusionNet(384, 256)
        
        self.fc1 = nn.Linear(128 + 128 + 128, 128)
        self.fc2 = nn.Linear(128 + 128, 128)
        self.fcs = nn.Linear(384, 128)
        self.fc3 = nn.Linear(384, 2)

    def forward(self, x1a, x1b, x1c, x2):
        # MDI Network
        #############
        out1a = self.pipeline1a(x1a)
        out1b = self.pipeline1b(x1b)
        out1c = self.pipeline1c(x1c)

        # Spatial attention
        out1a = self.spat_att(out1a.unsqueeze(2)).squeeze(2)
        out1b = self.spat_att(out1b.unsqueeze(2)).squeeze(2)
        out1c = self.spat_att(out1c.unsqueeze(2)).squeeze(2)

        out1a = self.layer_norm(out1a)
        out1b = self.layer_norm(out1b)
        out1c = self.layer_norm(out1c)

        # Cross Attention
        out1a = out1a.unsqueeze(1)
        out1b = out1b.unsqueeze(1)
        out1c = out1c.unsqueeze(1)
        
        out_conc = torch.cat((out1a, out1b, out1c), dim=1)
        out_conc = self.cross_mdin_att(out_conc)
        out1a = out_conc[:, 0, :]
        out1b = out_conc[:, 1, :]
        out1c = out_conc[:, 2, :]

        # Concatnation
        x1 = torch.cat((out1a, out1b, out1c), dim=1)

        # ConvxLSTM Network
        ###################
        out2 = self.pipeline2(x2)
        out3 = self.pipeline3(x2)

        # Spatial attention
        out2 = self.spat_att(out2.unsqueeze(2)).squeeze(2)
        out3 = self.spat_att(out3.unsqueeze(2)).squeeze(2)

        out2 = self.layer_norm(out2)
        out3 = self.layer_norm(out3)

        out2 = out2.unsqueeze(1)
        out3 = out3.unsqueeze(1)
        
        out_conc1 = torch.cat((out2, out3), dim=1)
        out_conc1 = self.cross_ConvxLSTM_att(out_conc1)
        out2 = out_conc1[:, 0, :]
        out3 = out_conc1[:, 1, :]

        x2 = torch.cat((out2, out3), dim=1)

        # Fusion Network
        x3, x3x = self.fusion(x1, x2)
        x3, x3x = self.fusion(x3, x3x)
        '''x1 = F.silu(self.fc1(x1))
        x2 = F.silu(self.fc2(x2))
        
        x3 = torch.cat((x1, x2), dim=1)'''

        xs = F.softmax(self.fcs(x3), dim=-1)
        xf = torch.sigmoid(self.fc3(x3))
        return xf, xs

# Example usage:
# model = PPGNet().to(device)
