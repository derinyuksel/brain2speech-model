import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Pulling in the custom/dummy classes I wrote
from model import CNNLSTM_Decoder
from dataset import DummyEEGDataset

def train_dummy_model():
    # 1. Setup the dummy data pipeline
    print("Loading dummy dataset...")
    # Matches the exact shapes Razan will eventually give us for Mel-Spectrograms
    dataset = DummyEEGDataset(num_samples=100, num_channels=64, seq_length=100, num_classes=50)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 2. Initialize the model architecture
    print("Initializing CNN-LSTM model...")
    model = CNNLSTM_Decoder(input_channels=64, cnn_hidden=128, lstm_hidden=256, num_classes=50)

    # 3. Define the learning rules
    # We use Cross-Entropy because we are trying to classify specific words right now
    criterion = nn.CrossEntropyLoss()
    # Adam is a solid default optimizer to update the network weights
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. The actual training loop
    epochs = 5  # Just keeping it short to prove the math works
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (features, labels) in enumerate(dataloader):
            # Always zero out the gradients from the last step so they don't stack
            optimizer.zero_grad()
            
            # Forward pass: make a guess
            predictions = model(features)
            
            # Calculate the penalty (how far off the guess was from the actual label)
            loss = criterion(predictions, labels)
            
            # Backward pass: calculate the updates
            loss.backward()
            
            # Optimizer step: apply the updates to the weights
            optimizer.step()
            
            total_loss += loss.item()
            
        # Calculate and print the average loss for this epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
        
    print("Training loop finished successfully!")
    
    # Save the trained weights to a file so we can hand it to Sudenur later
    torch.save(model.state_dict(), "dummy_model.pth")
    print("Saved model weights to dummy_model.pth")

if __name__ == "__main__":
    train_dummy_model()