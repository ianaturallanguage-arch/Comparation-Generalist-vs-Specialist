import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import List
from transformers import AutoTokenizer

# Try to use ProGen tokenizer, fallback to custom if not available
try:
    # ProGen models use a specific tokenizer
    # Common model names: "nferruz/ProGen2-small", "nferruz/ProGen2-medium"
    TOKENIZER_NAME = "nferruz/ProGen2-small"
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
        print(f"Loaded ProGen tokenizer from {TOKENIZER_NAME}")
    except:
        print(f"Could not load tokenizer from {TOKENIZER_NAME}, using custom tokenizer")
        tokenizer = None
except:
    tokenizer = None

# Custom tokenizer for amino acids (fallback)
AMINO_ACIDS = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
               'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
AA_TO_ID = {aa: i+1 for i, aa in enumerate(AMINO_ACIDS)}
PADDING_TOKEN = 0
MAX_LEN = 300


class ProGenDataset(Dataset):
    """Dataset for ProGen fine-tuning with serine protease sequences"""
    
    def __init__(self, sequences: List[str], tokenizer=None, max_len: int = MAX_LEN):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_custom = tokenizer is None
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        if self.use_custom:
            # Custom tokenization
            tokens = [AA_TO_ID.get(aa, PADDING_TOKEN) for aa in sequence]
            if len(tokens) > self.max_len:
                tokens = tokens[:self.max_len]
            else:
                tokens = tokens + [PADDING_TOKEN] * (self.max_len - len(tokens))
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            # Use ProGen tokenizer
            # ProGen tokenizer expects sequences as strings
            encoded = self.tokenizer(
                sequence,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tokens = encoded['input_ids'].squeeze(0)
        
        # For autoregressive: input is sequence[:-1], target is sequence[1:]
        input_seq = tokens[:-1]
        target_seq = tokens[1:]
        
        return input_seq, target_seq


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


def load_dataset(data_dir: str = "/kaggle/input/serine-proteases"):
    """Load dataset from Kaggle input directory"""
    
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
        seq_col = None
        for col in ['sequence', 'Sequence', 'seq', 'Seq', 'protein_sequence']:
            if col in df_train.columns:
                seq_col = col
                break
        if seq_col:
            train_sequences = df_train[seq_col].tolist()
        else:
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


def create_progen_dataloaders(data_dir: str = "/kaggle/input/serine-proteases", 
                               batch_size: int = 8,
                               tokenizer=None):
    """Create train and validation dataloaders for ProGen"""
    
    train_sequences, val_sequences = load_dataset(data_dir)
    
    train_dataset = ProGenDataset(train_sequences, tokenizer=tokenizer, max_len=MAX_LEN)
    val_dataset = ProGenDataset(val_sequences, tokenizer=tokenizer, max_len=MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
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

