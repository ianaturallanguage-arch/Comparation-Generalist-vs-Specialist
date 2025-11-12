# ProGen Fine-Tuning for Serine Protease Sequences

Fine-tuning script for ProGen2 protein language model on serine protease sequences, optimized for GPU P100.

## Model Selection

- **Model**: ProGen2-Small (151M parameters)
- **Reason**: Fits comfortably within P100 GPU memory (16GB)
- **Alternative**: ProGen2-Medium (764M) if more memory is available

## Features

- **Memory Optimized**: Gradient checkpointing, gradient accumulation, and mixed precision
- **Efficient Training**: Batch size 4 with gradient accumulation (effective batch size 8)
- **Early Stopping**: Prevents overfitting with patience=3 epochs
- **Comprehensive Analysis**: Detailed sequence statistics and visualizations

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch>=2.0.0`
- `transformers>=4.30.0`
- `accelerate>=0.20.0`

## Usage

### Full Pipeline (Fine-tuning + Generation + Analysis)

```bash
python progen_main.py
```

### Fine-tuning Only

```bash
python progen_finetune.py
```

### Generation and Analysis Only (requires fine-tuned model)

```bash
python progen_generate.py
```

## Training Configuration

- **Model**: `nferruz/ProGen2-small`
- **Batch Size**: 4 (with gradient accumulation steps=2, effective batch size=8)
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW (beta1=0.9, beta2=0.999)
- **Gradient Clipping**: max_norm=1.0
- **Early Stopping**: patience=3 epochs
- **Max Epochs**: 3
- **Sequence Length**: 300 tokens

## Memory Optimization for P100

1. **Gradient Checkpointing**: Reduces memory usage during forward pass
2. **Gradient Accumulation**: Simulates larger batch sizes without extra memory
3. **Mixed Precision**: Uses float16 when available (automatic with transformers)
4. **Smaller Batch Size**: 4 instead of 8 to fit model in memory

## Output Files

- `progen_finetuned/`: Fine-tuned model in HuggingFace format
- `progen_serine_protease.pt`: PyTorch checkpoint with training history
- `progen_generated_sequences.fasta`: Generated sequences
- `progen_sequence_analysis.png`: Analysis visualizations

## Model Loading

The fine-tuned model can be loaded using:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("progen_finetuned")
tokenizer = AutoTokenizer.from_pretrained("progen_finetuned")
```

## Analysis Features

The analysis includes:
1. Basic statistics (length distribution, mean, median, etc.)
2. Amino acid composition
3. Comparison with reference sequences (if available)
4. Sequence diversity metrics
5. Dipeptide frequency analysis
6. Hydrophobicity analysis
7. Sample generated sequences

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Reduce batch_size to 2 or 1
2. Increase gradient_accumulation_steps to maintain effective batch size
3. Reduce max_length if sequences are very long
4. Use a smaller model variant

### Model Download Issues

If ProGen2-Small is not available:
- The script will try ProtGPT2 as a fallback
- You can manually specify a different model name
- Ensure you have internet connection for first-time model download

## Comparison with Custom Transformer

This ProGen fine-tuning approach offers:
- **Pre-trained knowledge**: Leverages protein language understanding
- **Better generalization**: Pre-trained on large protein datasets
- **Faster convergence**: Fine-tuning typically requires fewer epochs
- **Larger model capacity**: 151M parameters vs custom 64-dim model

The custom Transformer approach offers:
- **Full control**: Complete architecture customization
- **Smaller footprint**: More memory efficient
- **Faster training**: Smaller model trains faster
- **No external dependencies**: Self-contained implementation

