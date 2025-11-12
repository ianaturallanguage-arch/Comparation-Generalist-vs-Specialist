import torch
import numpy as np
from collections import Counter
from model import SerineProteaseTransformer
from data_loader import ID_TO_AA, AMINO_ACIDS, PADDING_TOKEN
import matplotlib.pyplot as plt
import seaborn as sns


def id_to_sequence(token_ids):
    """Convert token IDs to amino acid sequence"""
    sequence = []
    for token_id in token_ids:
        if token_id == PADDING_TOKEN:
            break
        if token_id in ID_TO_AA:
            sequence.append(ID_TO_AA[token_id])
    return ''.join(sequence)


def analyze_sequences(generated_sequences, reference_sequences=None):
    """Detailed analysis of generated sequences"""
    
    print("=" * 80)
    print("DETAILED SEQUENCE ANALYSIS")
    print("=" * 80)
    
    # 1. Basic Statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 80)
    lengths = [len(seq) for seq in generated_sequences]
    print(f"Number of generated sequences: {len(generated_sequences)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths):.2f}")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Std deviation: {np.std(lengths):.2f}")
    
    # 2. Amino Acid Composition
    print("\n2. AMINO ACID COMPOSITION")
    print("-" * 80)
    all_aa = ''.join(generated_sequences)
    aa_counts = Counter(all_aa)
    total_aa = len(all_aa)
    
    print(f"{'Amino Acid':<15} {'Count':<10} {'Frequency (%)':<15}")
    print("-" * 40)
    for aa in AMINO_ACIDS:
        count = aa_counts.get(aa, 0)
        freq = (count / total_aa * 100) if total_aa > 0 else 0
        print(f"{aa:<15} {count:<10} {freq:<15.2f}")
    
    # 3. Compare with Reference (if provided)
    if reference_sequences:
        print("\n3. COMPARISON WITH REFERENCE SEQUENCES")
        print("-" * 80)
        
        ref_lengths = [len(seq) for seq in reference_sequences]
        print(f"Reference mean length: {np.mean(ref_lengths):.2f}")
        print(f"Generated mean length: {np.mean(lengths):.2f}")
        
        ref_aa = ''.join(reference_sequences)
        ref_aa_counts = Counter(ref_aa)
        ref_total = len(ref_aa)
        
        print(f"\n{'Amino Acid':<15} {'Ref Freq (%)':<15} {'Gen Freq (%)':<15} {'Difference':<15}")
        print("-" * 60)
        for aa in AMINO_ACIDS:
            ref_freq = (ref_aa_counts.get(aa, 0) / ref_total * 100) if ref_total > 0 else 0
            gen_freq = (aa_counts.get(aa, 0) / total_aa * 100) if total_aa > 0 else 0
            diff = gen_freq - ref_freq
            print(f"{aa:<15} {ref_freq:<15.2f} {gen_freq:<15.2f} {diff:<15.2f}")
    
    # 4. Sequence Diversity
    print("\n4. SEQUENCE DIVERSITY")
    print("-" * 80)
    unique_sequences = len(set(generated_sequences))
    print(f"Unique sequences: {unique_sequences}/{len(generated_sequences)}")
    print(f"Diversity ratio: {unique_sequences/len(generated_sequences)*100:.2f}%")
    
    # 5. Dipeptide Analysis
    print("\n5. DIPEPTIDE FREQUENCY (Top 20)")
    print("-" * 80)
    dipeptides = []
    for seq in generated_sequences:
        for i in range(len(seq) - 1):
            dipeptides.append(seq[i:i+2])
    
    dipeptide_counts = Counter(dipeptides)
    top_dipeptides = dipeptide_counts.most_common(20)
    
    print(f"{'Dipeptide':<15} {'Count':<10} {'Frequency (%)':<15}")
    print("-" * 40)
    for dipep, count in top_dipeptides:
        freq = (count / len(dipeptides) * 100) if dipeptides else 0
        print(f"{dipep:<15} {count:<10} {freq:<15.2f}")
    
    # 6. Hydrophobicity Analysis
    print("\n6. HYDROPHOBICITY ANALYSIS")
    print("-" * 80)
    # Kyte-Doolittle scale (simplified)
    hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'}
    polar = {'G', 'S', 'T', 'C', 'Y', 'N', 'Q'}
    charged = {'D', 'E', 'K', 'R', 'H'}
    
    hydro_count = sum(1 for aa in all_aa if aa in hydrophobic)
    polar_count = sum(1 for aa in all_aa if aa in polar)
    charged_count = sum(1 for aa in all_aa if aa in charged)
    
    print(f"Hydrophobic residues: {hydro_count} ({hydro_count/total_aa*100:.2f}%)")
    print(f"Polar residues: {polar_count} ({polar_count/total_aa*100:.2f}%)")
    print(f"Charged residues: {charged_count} ({charged_count/total_aa*100:.2f}%)")
    
    # 7. Sequence Examples
    print("\n7. SAMPLE GENERATED SEQUENCES")
    print("-" * 80)
    for i, seq in enumerate(generated_sequences[:10]):
        print(f"Sequence {i+1} (length {len(seq)}):")
        print(f"  {seq[:80]}{'...' if len(seq) > 80 else ''}")
        print()
    
    return {
        'lengths': lengths,
        'aa_composition': dict(aa_counts),
        'diversity': unique_sequences / len(generated_sequences),
        'dipeptides': dict(dipeptide_counts),
        'hydrophobicity': {
            'hydrophobic': hydro_count / total_aa,
            'polar': polar_count / total_aa,
            'charged': charged_count / total_aa
        }
    }


def generate_sequences(model, num_sequences: int = 100, device: torch.device = None, 
                      temperature: float = 1.0, start_token: int = 1):
    """Generate multiple sequences"""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    sequences = []
    
    print(f"Generating {num_sequences} sequences...")
    for i in range(num_sequences):
        token_ids = model.generate(
            start_token=start_token,
            max_length=300,
            temperature=temperature,
            device=device
        )
        sequence = id_to_sequence(token_ids)
        sequences.append(sequence)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_sequences} sequences...")
    
    return sequences


def plot_analysis(analysis_results, save_path: str = "sequence_analysis.png"):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Length distribution
    axes[0, 0].hist(analysis_results['lengths'], bins=30, edgecolor='black')
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Generated Sequence Lengths')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Amino acid composition
    aa_comp = analysis_results['aa_composition']
    aa_names = sorted(aa_comp.keys())
    aa_counts = [aa_comp[aa] for aa in aa_names]
    
    axes[0, 1].bar(aa_names, aa_counts)
    axes[0, 1].set_xlabel('Amino Acid')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Amino Acid Composition')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Top dipeptides
    dipeptides = analysis_results['dipeptides']
    top_dipeps = sorted(dipeptides.items(), key=lambda x: x[1], reverse=True)[:20]
    dipep_names = [d[0] for d in top_dipeps]
    dipep_counts = [d[1] for d in top_dipeps]
    
    axes[1, 0].barh(dipep_names, dipep_counts)
    axes[1, 0].set_xlabel('Count')
    axes[1, 0].set_ylabel('Dipeptide')
    axes[1, 0].set_title('Top 20 Dipeptide Frequencies')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Hydrophobicity pie chart
    hydro = analysis_results['hydrophobicity']
    labels = ['Hydrophobic', 'Polar', 'Charged']
    sizes = [hydro['hydrophobic'], hydro['polar'], hydro['charged']]
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Residue Type Distribution')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis plots saved to {save_path}")
    plt.close()


def main():
    """Main function for generation and analysis"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    print("Loading trained model...")
    model = SerineProteaseTransformer(
        vocab_size=21,
        embedding_dim=64,
        n_layers=2,
        n_heads=4,
        dim_feedforward=128,
        max_len=300,
        dropout=0.1
    ).to(device)
    
    checkpoint = torch.load("serine_protease_transformer.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded successfully!")
    
    # Generate sequences
    generated_sequences = generate_sequences(
        model, 
        num_sequences=100, 
        device=device,
        temperature=1.0
    )
    
    # Load reference sequences for comparison (optional)
    try:
        from data_loader import load_dataset
        _, val_sequences = load_dataset("/kaggle/input/serine-proteases")
        reference_sequences = val_sequences[:100]  # Sample for comparison
    except:
        reference_sequences = None
    
    # Analyze sequences
    analysis_results = analyze_sequences(generated_sequences, reference_sequences)
    
    # Create visualizations
    plot_analysis(analysis_results)
    
    # Save generated sequences
    with open("generated_sequences.fasta", "w") as f:
        for i, seq in enumerate(generated_sequences):
            f.write(f">generated_sequence_{i+1}\n")
            f.write(f"{seq}\n")
    
    print("\nGenerated sequences saved to generated_sequences.fasta")


if __name__ == "__main__":
    main()

