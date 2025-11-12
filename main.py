"""
Main script to train and analyze serine protease Transformer model
Optimized for GPU P100
"""

import torch
from train import train_model
from generate_and_analyze import generate_sequences, analyze_sequences, plot_analysis
from model import SerineProteaseTransformer
from data_loader import ID_TO_AA, PADDING_TOKEN

def main():
    # Optimize for GPU P100
    torch.backends.cudnn.benchmark = True
    
    print("=" * 80)
    print("SERINE PROTEASE TRANSFORMER MODEL")
    print("Optimized for GPU P100")
    print("=" * 80)
    
    # Training
    print("\n" + "=" * 80)
    print("PHASE 1: TRAINING")
    print("=" * 80)
    
    model, train_losses, val_losses = train_model(
        data_dir="/kaggle/input/serine-proteases",
        batch_size=8,
        learning_rate=1e-3,
        num_epochs=3,
        patience=3
    )
    
    # Generation and Analysis
    print("\n" + "=" * 80)
    print("PHASE 2: SEQUENCE GENERATION AND ANALYSIS")
    print("=" * 80)
    
    device = next(model.parameters()).device
    
    # Generate sequences
    generated_sequences = generate_sequences(
        model,
        num_sequences=100,
        device=device,
        temperature=1.0
    )
    
    # Try to load reference sequences for comparison
    try:
        from data_loader import load_dataset
        _, val_sequences = load_dataset("/kaggle/input/serine-proteases")
        reference_sequences = val_sequences[:100]
    except:
        reference_sequences = None
    
    # Detailed analysis
    analysis_results = analyze_sequences(generated_sequences, reference_sequences)
    
    # Create visualizations
    plot_analysis(analysis_results)
    
    # Save generated sequences
    with open("generated_sequences.fasta", "w") as f:
        for i, seq in enumerate(generated_sequences):
            f.write(f">generated_sequence_{i+1}\n")
            f.write(f"{seq}\n")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("Generated sequences saved to: generated_sequences.fasta")
    print("Analysis plots saved to: sequence_analysis.png")
    print("Model saved to: serine_protease_transformer.pt")

if __name__ == "__main__":
    main()

