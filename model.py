import torch
import torch.nn as nn

class CNNLSTM_Decoder(nn.Module):
    def __init__(self, input_channels, cnn_hidden, lstm_hidden, num_classes):
        super(CNNLSTM_Decoder, self).__init__()
        
        # 1. The CNN Layer: For spatial electrode pattern learning
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_hidden),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # 2. The LSTM Layer: For spatial-temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_hidden, 
            hidden_size=lstm_hidden, 
            num_layers=1, 
            batch_first=True
        )
        
        # 3. The Output Layer: Maps the sequence to your chosen labels (words, phonemes, etc.)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        # x shape expected: (batch_size, input_channels, sequence_length)
        
        # Pass through CNN
        x = self.cnn(x)
        
        # Reshape for LSTM: from (batch, features, time) to (batch, time, features)
        x = x.permute(0, 2, 1)
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(x)
        
        # For simplicity in this skeleton, we take the output of the last time step
        # This will need to be adjusted later if we use CTC loss for sequence prediction
        final_output = self.fc(lstm_out[:, -1, :]) 
        
        return final_output

# Quick test block to ensure the architecture compiles
if __name__ == "__main__":
    # Simulating a random EEG input batch: 
    # (batch_size=8, electrodes/channels=64, time_samples=100)
    dummy_input = torch.randn(8, 64, 100) 
    
    # Initialize the empty brain
    model = CNNLSTM_Decoder(input_channels=64, cnn_hidden=128, lstm_hidden=256, num_classes=50)
    
    # Run the dummy data through the model
    output = model(dummy_input)
    print(f"Success! Output shape: {output.shape}")