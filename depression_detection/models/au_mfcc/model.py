import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class DepressionDetectionModel(nn.Module):
    def __init__(self, num_au_features, num_mfcc_features, au_fixed_length, mfcc_fixed_length, hidden_size):
        super(DepressionDetectionModel, self).__init__()

        self.au_fixed_length = au_fixed_length
        self.mfcc_fixed_length = mfcc_fixed_length
        
        self.mfcc_lstm = nn.LSTM(input_size=num_mfcc_features, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)

        self.au_lstm = nn.LSTM(input_size=num_au_features, hidden_size=hidden_size, num_layers=3, batch_first=True, bidirectional=True, dropout=0.2)

        self.fused_lstm = nn.LSTM(input_size=hidden_size*2, hidden_size=20, num_layers=1, batch_first=True, bidirectional=True)


        # Fusion Layer
        self.fc = nn.Sequential(
            nn.Linear((mfcc_fixed_length+au_fixed_length)*20*2, 128),
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
    
    def forward(self, au_input, mfcc_input):

        # Process AUs with Bi-LSTM and Attention
        mfcc_lstm_out, _= self.mfcc_lstm(mfcc_input) 
        au_lstm_out, _ = self.au_lstm(au_input)

        fused = torch.concat((au_lstm_out, mfcc_lstm_out), dim=1)
        fused_lstm_out, _ = self.fused_lstm(fused)

        # Flatten Features
        flattened_features = self.fc(fused_lstm_out.flatten(start_dim=1))  
        
        # Classification
        outputs = self.classifier(flattened_features)

        attention_weights = None
        
        return outputs, attention_weights # Return attention weights for interpretability