"""
Main script to fine-tune ProGen and analyze serine protease sequences
Optimized for GPU P100
"""

import torch
from progen_finetune import finetune_progen
from progen_generate import generate_sequences_progen, analyze_sequences, plot_analysis
from transformers import AutoModelForCausalLM, AutoTokenizer
from progen_data_loader import load_dataset
import os


def main():
    # Optimize for GPU P100
    torch.backends.cudnn.benchmark = True
    
    print("=" * 80)
    print("PROGEN FINE-TUNING FOR SERINE PROTEASE SEQUENCES")
    print("Optimized for GPU P100")
    print("=" * 80)
    
    # Fine-tuning
    print("\n" + "=" * 80)
    print("PHASE 1: FINE-TUNING PROGEN MODEL")
    print("=" * 80)
    
    model, tokenizer, train_losses, val_losses = finetune_progen(
        data_dir="/kaggle/input/serine-proteases",
        model_name="nferruz/ProGen2-small",  # 151M parameters, fits on P100
        batch_size=4,  # Smaller batch for P100 memory
        learning_rate=1e-4,
        num_epochs=3,
        patience=3,
        gradient_accumulation_steps=2  # Effective batch size = 8
    )
    
    # Generation and Analysis
    print("\n" + "=" * 80)
    print("PHASE 2: SEQUENCE GENERATION AND ANALYSIS")
    print("=" * 80)
    
    device = next(model.parameters()).device
    
    # Generate sequences
    generated_sequences = generate_sequences_progen(
        model,
        tokenizer,
        num_sequences=100,
        device=device,
        max_length=300,
        temperature=1.0
    )
    
    # Try to load reference sequences for comparison
    try:
        _, val_sequences = load_dataset("/kaggle/input/serine-proteases")
        reference_sequences = val_sequences[:100]
    except:
        reference_sequences = None
    
    # Detailed analysis
    analysis_results = analyze_sequences(generated_sequences, reference_sequences)
    
    # Create visualizations
    plot_analysis(analysis_results)
    
    # Save generated sequences
    with open("progen_generated_sequences.fasta", "w") as f:
        for i, seq in enumerate(generated_sequences):
            f.write(f">progen_generated_sequence_{i+1}\n")
            f.write(f"{seq}\n")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("Generated sequences saved to: progen_generated_sequences.fasta")
    print("Analysis plots saved to: progen_sequence_analysis.png")
    print("Fine-tuned model saved to: progen_finetuned/")
    print("Checkpoint saved to: progen_serine_protease.pt")


if __name__ == "__main__":
    main()

