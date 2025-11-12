import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple

# Amino acid mapping: 20 standard amino acids + padding token (0)
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_TO_ID = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}  # 1-20
ID_TO_AA = {i+1: aa for i, aa in enumerate(AMINO_ACIDS)}
PADDING_TOKEN = 0
MAX_LEN = 300
VOCAB_SIZE = 21  # 20 amino acids + padding


class SerineProteaseDataset(Dataset):
    """Dataset for serine protease sequences"""
    
    def __init__(self, sequences: List[str], max_len: int = MAX_LEN):
        self.sequences = sequences
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # Tokenize sequence
        tokens = [AA_TO_ID.get(aa, PADDING_TOKEN) for aa in sequence]
        
        # Pad or truncate to max_len
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [PADDING_TOKEN] * (self.max_len - len(tokens))
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # For autoregressive generation: input is sequence[:-1], target is sequence[1:]
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        
        return input_seq, target_seq


def load_dataset(data_dir: str = "/kaggle/input/serine-proteases"):
    """Load dataset from Kaggle input directory"""
    
    # Try different possible file formats
    train_file = None
    val_file = None
    
    # Check for CSV files
    if os.path.exists(os.path.join(data_dir, "train.csv")):
        train_file = os.path.join(data_dir, "train.csv")
    if os.path.exists(os.path.join(data_dir, "val.csv")):
        val_file = os.path.join(data_dir, "val.csv")
    elif os.path.exists(os.path.join(data_dir, "validation.csv")):
        val_file = os.path.join(data_dir, "validation.csv")
    
    # Check for FASTA files
    if not train_file and os.path.exists(os.path.join(data_dir, "train.fasta")):
        train_file = os.path.join(data_dir, "train.fasta")
    if not val_file and os.path.exists(os.path.join(data_dir, "val.fasta")):
        val_file = os.path.join(data_dir, "val.fasta")
    
    # If no split files, try single file
    if not train_file:
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.fasta', '.txt')):
                train_file = os.path.join(data_dir, file)
                break
    
    if not train_file or not os.path.exists(train_file):
        raise FileNotFoundError(f"Could not find dataset files in {data_dir}")
    
    # Load sequences
    train_sequences = []
    val_sequences = []
    
    if train_file.endswith('.csv'):
        df_train = pd.read_csv(train_file)
        # Try common column names
        seq_col = None
        for col in ['sequence', 'Sequence', 'seq', 'Seq', 'protein_sequence']:
            if col in df_train.columns:
                seq_col = col
                break
        if seq_col:
            train_sequences = df_train[seq_col].tolist()
        else:
            # Assume first column is sequence
            train_sequences = df_train.iloc[:, 0].tolist()
    
    elif train_file.endswith('.fasta'):
        train_sequences = load_fasta(train_file)
    
    if val_file:
        if val_file.endswith('.csv'):
            df_val = pd.read_csv(val_file)
            seq_col = None
            for col in ['sequence', 'Sequence', 'seq', 'Seq', 'protein_sequence']:
                if col in df_val.columns:
                    seq_col = col
                    break
            if seq_col:
                val_sequences = df_val[seq_col].tolist()
            else:
                val_sequences = df_val.iloc[:, 0].tolist()
        elif val_file.endswith('.fasta'):
            val_sequences = load_fasta(val_file)
    
    # Filter out invalid sequences
    train_sequences = [seq for seq in train_sequences if isinstance(seq, str) and len(seq) > 0]
    val_sequences = [seq for seq in val_sequences if isinstance(seq, str) and len(seq) > 0]
    
    print(f"Loaded {len(train_sequences)} training sequences")
    print(f"Loaded {len(val_sequences)} validation sequences")
    
    return train_sequences, val_sequences


def load_fasta(filepath: str) -> List[str]:
    """Load sequences from FASTA file"""
    sequences = []
    current_seq = []
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        
        if current_seq:
            sequences.append(''.join(current_seq))
    
    return sequences


def create_dataloaders(data_dir: str = "/kaggle/input/serine-proteases", 
                       batch_size: int = 8):
    """Create train and validation dataloaders"""
    
    train_sequences, val_sequences = load_dataset(data_dir)
    
    train_dataset = SerineProteaseDataset(train_sequences, max_len=MAX_LEN)
    val_dataset = SerineProteaseDataset(val_sequences, max_len=MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 for Kaggle compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader

