import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class DepressionDetectionModel(nn.Module):
    def __init__(self, num_mfcc_features, mfcc_fixed_length, hidden_size):
        super(DepressionDetectionModel, self).__init__()

        self.mfcc_fixed_length = mfcc_fixed_length
        
        # Bi-LSTM for MFCC
        self.mfcc_lstm = nn.LSTM(input_size=num_mfcc_features, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)

        # Fusion Layer
        self.fc = nn.Sequential(
            nn.Linear(mfcc_fixed_length*hidden_size*2, 128),  
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, mfcc_input):
        
        # Process MFCCs with Bi-LSTM and Attention
        lstm_out, _ = self.mfcc_lstm(mfcc_input)

        # Flatten Features
        flattened_features = lstm_out.flatten(start_dim=1)
        flattened_features = self.fc(flattened_features)  
        
        # Classification
        outputs = self.classifier(flattened_features)

        attention_weights = None
        
        return outputs, attention_weights # Return attention weights for interpretability