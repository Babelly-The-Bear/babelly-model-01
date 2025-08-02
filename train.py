import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms  # Removed unused import
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class DeepInfantDataset(Dataset):
    def __init__(self, data_dir, apply_augmentation=False):
        self.data_dir = Path(data_dir)
        self.apply_augmentation = apply_augmentation
        self.samples = []
        self.labels = []
        
        # Updated label mapping based on new classes
        self.label_map = {
            'bp': 0,  # belly pain
            'bu': 1,  # burping
            'ch': 2,  # cold/hot
            'dc': 3,  # discomfort
            'hu': 4,  # hungry
            'lo': 5,  # lonely
            'sc': 6,  # scared
            'ti': 7,  # tired
            'un': 8,  # unknown
        }
        
        # Load metadata if available
        metadata_file = Path(data_dir).parent / 'metadata.csv'
        if metadata_file.exists():
            self._load_from_metadata(metadata_file)
        else:
            self._load_dataset()
    
    def _load_from_metadata(self, metadata_file):
        df = pd.read_csv(metadata_file)
        for _, row in df.iterrows():
            if row['split'] == self.data_dir.name:  # 'train' or 'test'
                audio_path = self.data_dir / row['filename']
                if audio_path.exists():
                    self.samples.append(str(audio_path))
                    self.labels.append(self.label_map[row['class_code']])
    
    def _load_dataset(self):
        for audio_file in self.data_dir.glob('*.*'):
            if audio_file.suffix in ['.wav', '.caf', '.3gp']:
                # Parse filename for label
                label = audio_file.stem.split('-')[-1][:2]  # Get reason code
                if label in self.label_map:
                    self.samples.append(str(audio_file))
                    self.labels.append(self.label_map[label])
    
    def _process_audio(self, audio_path):
        # Load audio with 16kHz sample rate
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Add basic audio augmentation (during training)
        if self.apply_augmentation:
            # Random time shift (-100ms to 100ms)
            shift = np.random.randint(-1600, 1600)
            if shift > 0:
                waveform = np.pad(waveform, (shift, 0))[:len(waveform)]
            else:
                waveform = np.pad(waveform, (0, -shift))[(-shift):]
            
            # Random noise injection
            if np.random.random() < 0.3:
                noise = np.random.normal(0, 0.005, len(waveform))
                waveform = waveform + noise
        
        # Ensure consistent length (7 seconds)
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        # Generate mel spectrogram with adjusted parameters
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,  # Reduced from 2048 for better temporal resolution
            hop_length=256,  # Reduced from 512
            n_mels=80,  # Standard for speech/audio
            fmin=20,  # Minimum frequency
            fmax=8000  # Maximum frequency, suitable for infant cries
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Add standardization for better training stability
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return torch.FloatTensor(mel_spec)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path = self.samples[idx]
        label = self.labels[idx]
        
        # Process audio to mel spectrogram
        mel_spec = self._process_audio(audio_path)
        
        return mel_spec, label

class DeepInfantModel(nn.Module):
    def __init__(self, num_classes=9):
        super(DeepInfantModel, self).__init__()
        
        # CNN layers with residual connections
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # After 3 MaxPool2d(2), dimensions are reduced by 8x
        # Original: 80 freq bins -> 10 freq bins after pooling
        # 7 seconds * 16000 / 256 hop = 438 time steps -> 54 time steps after pooling
        # So LSTM input size should be 256 channels * 10 freq bins = 2560
        
        # Bi-directional LSTM for better temporal modeling
        self.lstm = nn.LSTM(
            input_size=2560,  # 256 channels * 10 freq bins after pooling
            hidden_size=256,  # Reduced from 512 to prevent overfitting
            num_layers=1,     # Reduced from 2 layers
            batch_first=True,
            bidirectional=True,
            dropout=0.0       # Remove dropout from LSTM to help learning
        )
        
        # Final classification layers
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # 512 due to bidirectional LSTM
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch, 1, freq_bins, time_steps)
        batch_size = x.size(0)
        
        # CNN feature extraction
        x = self.conv_layers(x)
        
        # Reshape for LSTM
        freq_bins_after_pool = x.size(2)
        time_steps_after_pool = x.size(3)
        channels = x.size(1)
        
        # Reshape: combine frequency and channel dimensions
        x = x.permute(0, 3, 1, 2)  # (batch, time, channels, freq)
        x = x.reshape(batch_size, time_steps_after_pool, channels * freq_bins_after_pool)
        
        # LSTM processing
        x, _ = self.lstm(x)
        
        # Use average pooling instead of just last time step for better representation
        x = torch.mean(x, dim=1)  # Average over time dimension
        
        # Classification
        x = self.classifier(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')):
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Print batch-level info every 5 batches to monitor training
            if batch_idx % 5 == 0:
                current_loss = loss.item()
                current_acc = 100. * predicted.eq(labels).sum().item() / labels.size(0)
                print(f'Batch {batch_idx}, Loss: {current_loss:.4f}, Acc: {current_acc:.2f}%')
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Step the scheduler
        scheduler.step(val_loss / len(val_loader))
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print('-' * 50)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'deepinfant.pth')
            print(f'New best model saved with validation accuracy: {best_val_acc:.2f}%')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets using processed data
    train_dataset = DeepInfantDataset('processed_dataset/train', apply_augmentation=True)
    val_dataset = DeepInfantDataset('processed_dataset/test', apply_augmentation=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders with smaller batch size for better gradients
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model, loss function, and optimizer
    model = DeepInfantModel()
    
    # Use label smoothing to prevent overconfidence
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Use a lower learning rate and weight decay for better training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, device=str(device))

if __name__ == '__main__':
    main() 