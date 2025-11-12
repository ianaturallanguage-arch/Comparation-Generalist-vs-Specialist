import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from progen_data_loader import create_progen_dataloaders, load_dataset, ProGenDataset
import warnings
warnings.filterwarnings('ignore')


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


def train_epoch(model, train_loader, criterion, optimizer, device, pad_token_id=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for input_seq, target_seq in tqdm(train_loader, desc="Training"):
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        # Forward pass
        outputs = model(input_ids=input_seq, labels=target_seq)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for input_seq, target_seq in tqdm(val_loader, desc="Validating"):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_seq, labels=target_seq)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def load_progen_model(model_name: str = "nferruz/ProGen2-small", device: torch.device = None):
    """Load ProGen model and tokenizer"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading ProGen model: {model_name}")
    print("This may take a few minutes on first run...")
    
    try:
        # Try to load ProGen2 model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
        )
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Successfully loaded {model_name}")
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print("Falling back to alternative model or custom implementation...")
        
        # Fallback: try ProtGPT2 or other protein models
        try:
            print("Trying ProtGPT2 as alternative...")
            model_name = "nferruz/ProtGPT2"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print(f"Successfully loaded {model_name}")
            return model, tokenizer
        except Exception as e2:
            print(f"Error loading alternative model: {e2}")
            raise


def finetune_progen(
    data_dir: str = "/kaggle/input/serine-proteases",
    model_name: str = "nferruz/ProGen2-small",
    batch_size: int = 4,  # Smaller batch for P100
    learning_rate: float = 1e-4,
    num_epochs: int = 3,
    patience: int = 3,
    gradient_accumulation_steps: int = 2,  # Effective batch size = 4 * 2 = 8
    device: str = None
):
    """Fine-tune ProGen model on serine protease sequences"""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_progen_model(model_name, device)
    model = model.to(device)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_progen_dataloaders(
        data_dir, 
        batch_size=batch_size,
        tokenizer=tokenizer
    )
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training loop
    train_losses = []
    val_losses = []
    
    print("\nStarting fine-tuning...")
    print(f"Batch size: {batch_size}, Gradient accumulation: {gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train with gradient accumulation
        model.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (input_seq, target_seq) in enumerate(tqdm(train_loader, desc="Training")):
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # Forward pass
            outputs = model(input_ids=input_seq, labels=target_seq)
            loss = outputs.loss / gradient_accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
        
        train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping triggered after epoch {epoch + 1}")
            break
    
    print("\nFine-tuning completed!")
    
    # Save model
    model_path = "progen_serine_protease.pt"
    save_dir = "progen_finetuned"
    os.makedirs(save_dir, exist_ok=True)
    
    # Save using HuggingFace format
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    
    # Also save PyTorch checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_name': model_name,
    }, model_path)
    
    print(f"Model saved to {save_dir}/")
    print(f"Checkpoint saved to {model_path}")
    
    return model, tokenizer, train_losses, val_losses


if __name__ == "__main__":
    # Optimize for GPU P100
    torch.backends.cudnn.benchmark = True
    
    # Use mixed precision if available (saves memory)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA available, using GPU acceleration")
    
    model, tokenizer, train_losses, val_losses = finetune_progen(
        data_dir="/kaggle/input/serine-proteases",
        model_name="nferruz/ProGen2-small",
        batch_size=4,  # Smaller for P100
        learning_rate=1e-4,
        num_epochs=3,
        patience=3,
        gradient_accumulation_steps=2
    )

