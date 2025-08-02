import torch
import librosa
import numpy as np
from pathlib import Path
from train import DeepInfantModel  # Import the model architecture

class InfantCryPredictor:
    def __init__(self, model_path='deepinfant.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Check if model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first by running 'python train.py'")
            
        # Initialize model with correct number of classes (from train.py)
        self.model = DeepInfantModel(num_classes=9)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()
        
        # Label mapping (matches the training script)
        self.label_map = {
            0: 'belly pain',
            1: 'burping', 
            2: 'cold/hot',
            3: 'discomfort',
            4: 'hungry',
            5: 'lonely',
            6: 'scared',
            7: 'tired',
            8: 'unknown'
        }
    
    def _process_audio(self, audio_path):
        # Load audio with 16kHz sample rate (exactly like training)
        waveform, sample_rate = librosa.load(audio_path, sr=16000)
        
        print(f"Original waveform shape: {waveform.shape}, sample rate: {sample_rate}")
        print(f"Waveform range: [{waveform.min():.4f}, {waveform.max():.4f}]")
        
        # Ensure consistent length (7 seconds) - EXACTLY like training
        target_length = 7 * 16000
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))
        
        print(f"After padding waveform shape: {waveform.shape}")
        
        # Generate mel spectrogram with EXACT same parameters as training
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_fft=1024,  # Reduced from 2048 for better temporal resolution
            hop_length=256,  # Reduced from 512
            n_mels=80,  # Standard for speech/audio
            fmin=20,  # Minimum frequency
            fmax=8000  # Maximum frequency, suitable for infant cries
        )
        
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        print(f"Mel spec range before log: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
        
        # Convert to log scale - EXACTLY like training
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        print(f"Mel spec range after log: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
        
        return torch.FloatTensor(mel_spec)
    
    def predict(self, audio_path):
        """
        Predict the class of a single audio file
        Returns tuple of (predicted_label, confidence)
        """
        print(f"\n=== Predicting for: {audio_path} ===")
        
        # Process audio
        mel_spec = self._process_audio(audio_path)
        mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        mel_spec = mel_spec.to(self.device)
        
        print(f"Input tensor shape: {mel_spec.shape}")
        print(f"Input tensor range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(mel_spec)
            print(f"Raw model outputs: {outputs}")
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            print(f"Probabilities: {probabilities}")
            
            pred_class = torch.argmax(outputs, dim=1).item()
            pred_class = int(pred_class)  # Ensure it's an integer
            confidence = probabilities[0][pred_class].item()
            
            # Print all class probabilities for debugging
            print("All class probabilities:")
            for i, prob in enumerate(probabilities[0]):
                print(f"  {self.label_map[i]}: {prob.item():.4f}")
        
        print(f"Predicted class: {pred_class} ({self.label_map[pred_class]})")
        print(f"Confidence: {confidence:.4f}")
        
        return self.label_map[pred_class], confidence
    
    def predict_batch(self, audio_dir, file_extensions=('.wav', '.caf', '.3gp')):
        """
        Predict classes for all audio files in a directory
        Returns list of tuples (filename, predicted_label, confidence)
        """
        results = []
        audio_dir = Path(audio_dir)
        
        for audio_file in audio_dir.glob('*.*'):
            if audio_file.suffix.lower() in file_extensions:
                label, confidence = self.predict(str(audio_file))
                results.append((audio_file.name, label, confidence))
        
        return results

def main():
    # Example usage
    predictor = InfantCryPredictor()

    audio_path = "hungry.wav"
    label, confidence = predictor.predict(audio_path)
    print(f"\nPrediction for {audio_path}:")
    print(f"Label: {label}")
    print(f"Confidence: {confidence:.2%}")
    
    # Batch prediction
    audio_dir = "Data/v2/hungry"  # Change to your audio directory
    results = predictor.predict_batch(audio_dir)
    print("\nBatch Predictions:")
    for filename, label, confidence in results:
        print(f"{filename}: {label} ({confidence:.2%})")

if __name__ == "__main__":
    main() 