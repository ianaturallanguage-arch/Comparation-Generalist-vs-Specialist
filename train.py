import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
import os
from data_loader import create_dataloaders, PADDING_TOKEN
from model import SerineProteaseTransformer


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for input_seq, target_seq in tqdm(train_loader, desc="Training"):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        # Forward pass
        output = model(input_seq)  # [batch_size, seq_len, vocab_size]
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.size(-1))  # [batch_size * seq_len, vocab_size]
        target_seq = target_seq.reshape(-1)  # [batch_size * seq_len]
        
        # Calculate loss (padding tokens are automatically ignored if we set ignore_index)
        loss = criterion(output, target_seq)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_seq, target_seq in tqdm(val_loader, desc="Validating"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Forward pass
            output = model(input_seq)
            
            # Reshape for loss calculation
            output = output.reshape(-1, output.size(-1))
            target_seq = target_seq.reshape(-1)
            
            # Calculate loss
            loss = criterion(output, target_seq)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_model(
    data_dir: str = "/kaggle/input/serine-proteases",
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    num_epochs: int = 3,
    patience: int = 3,
    device: str = None
):
    """Main training function"""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(data_dir, batch_size=batch_size)
    
    # Initialize model
    print("Initializing model...")
    model = SerineProteaseTransformer(
        vocab_size=21,
        embedding_dim=64,
        n_layers=2,
        n_heads=4,
        dim_feedforward=128,
        max_len=300,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function (ignore padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_TOKEN)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999)
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break
    
    print("\nTraining completed!")
    
    # Save model
    model_path = "serine_protease_transformer.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, model_path)
    print(f"Model saved to {model_path}")
    
    return model, train_losses, val_losses


if __name__ == "__main__":
    # Optimize for GPU P100
    torch.backends.cudnn.benchmark = True
    
    model, train_losses, val_losses = train_model(
        data_dir="/kaggle/input/serine-proteases",
        batch_size=8,
        learning_rate=1e-3,
        num_epochs=3,
        patience=3
    )

