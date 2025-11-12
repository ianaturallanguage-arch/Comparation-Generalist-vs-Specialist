# Serine Protease Transformer Model

An autoregressive Transformer model for generating serine protease sequences, optimized for GPU P100.

## Model Architecture

- **Embedding**: vocab_size=21 (20 amino acids + padding), embedding_dim=64
- **Positional Encoding**: Sinusoidal, max_len=300
- **Decoder Blocks**: 2 layers
- **Attention Heads**: 4 per block
- **Feed-forward Dimension**: 128
- **Output**: 21 classes (vocab_size)

## Training Configuration

- **Optimizer**: AdamW (beta1=0.9, beta2=0.999)
- **Learning Rate**: 1e-3
- **Batch Size**: 8
- **Loss**: CrossEntropyLoss (ignores padding tokens)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: patience=3 epochs
- **Max Epochs**: 3

## Data Processing

- Amino acids mapped to integers 1-20
- Padding token = 0
- All sequences padded/truncated to length 300
- Uses existing train/validation splits from dataset

## Usage

### Training

```bash
python train.py
```

### Generation and Analysis

```bash
python generate_and_analyze.py
```

### Full Pipeline

```bash
python main.py
```

## Output Files

- `serine_protease_transformer.pt`: Trained model checkpoint
- `generated_sequences.fasta`: Generated sequences in FASTA format
- `sequence_analysis.png`: Visualization plots of sequence analysis

## Analysis Features

The analysis includes:
1. Basic statistics (length distribution, mean, median, etc.)
2. Amino acid composition
3. Comparison with reference sequences (if available)
4. Sequence diversity metrics
5. Dipeptide frequency analysis
6. Hydrophobicity analysis
7. Sample generated sequences

## Requirements

See `requirements.txt` for dependencies.

