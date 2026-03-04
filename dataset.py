import torch
from torch.utils.data import Dataset, DataLoader

class DummyEEGDataset(Dataset):
    def __init__(self, num_samples=100, num_channels=64, seq_length=100, num_classes=50):
        # We need this fake data class so I can build the training loop 
        # while Razan finishes the MNE-Python pipeline in Month 1.
        self.num_samples = num_samples
        
        # Generating random tensors to simulate Razan's extracted Mel-Spectrogram features.
        # Shape: (number of samples, mel bands/channels, time sequence length)
        self.data = torch.randn(num_samples, num_channels, seq_length)
        
        # Generating fake labels. These represent the target words (e.g., 50 different words)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        # PyTorch needs to know how many total items are in our dataset
        return self.num_samples

    def __getitem__(self, idx):
        # Grabs a single simulated brain signal and its corresponding word label
        return self.data[idx], self.labels[idx]

# Quick test to make sure our data loader spits out the right shapes
if __name__ == "__main__":
    # Create the fake dataset
    dataset = DummyEEGDataset()
    
    # DataLoader groups our data into batches (just like how we tested model.py with batch=8)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Grab one batch to test
    features, labels = next(iter(dataloader))
    
    print(f"Data ingested! Feature batch shape: {features.shape}")
    print(f"Labels batch shape: {labels.shape}")