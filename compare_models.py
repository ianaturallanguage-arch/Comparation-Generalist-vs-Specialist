"""
Compare generation results from SerineProteaseTransformer and fine-tuned ProGen
Creates comprehensive visualizations and statistical comparisons
"""

import torch
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Import custom model
from model import SerineProteaseTransformer
from data_loader import ID_TO_AA, AMINO_ACIDS, PADDING_TOKEN

# Import ProGen
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('dark_background')
sns.set_palette("husl")


def id_to_sequence(token_ids):
    """Convert token IDs to amino acid sequence"""
    sequence = []
    for token_id in token_ids:
        if token_id == PADDING_TOKEN:
            break
        if token_id in ID_TO_AA:
            sequence.append(ID_TO_AA[token_id])
    return ''.join(sequence)


def generate_custom_model(model, num_sequences=100, device=None, temperature=1.0):
    """Generate sequences using custom SerineProteaseTransformer"""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    sequences = []
    
    print(f"Generating {num_sequences} sequences with custom Transformer...")
    for i in range(num_sequences):
        token_ids = model.generate(
            start_token=1,
            max_length=300,
            temperature=temperature,
            device=device
        )
        sequence = id_to_sequence(token_ids)
        if len(sequence) > 0:
            sequences.append(sequence)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_sequences}...")
    
    return sequences


def generate_progen_model(model, tokenizer, num_sequences=100, device=None, 
                          max_length=300, temperature=1.0):
    """Generate sequences using fine-tuned ProGen"""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    sequences = []
    
    print(f"Generating {num_sequences} sequences with fine-tuned ProGen...")
    with torch.no_grad():
        for i in range(num_sequences):
            try:
                inputs = tokenizer("", return_tensors="pt").to(device)
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                sequence = ''.join([aa for aa in generated_text if aa in AMINO_ACIDS])
                
                if len(sequence) > 0:
                    sequences.append(sequence)
                
                if (i + 1) % 20 == 0:
                    print(f"  Generated {i + 1}/{num_sequences}...")
            except Exception as e:
                print(f"  Error generating sequence {i+1}: {e}")
                continue
    
    return sequences


def calculate_statistics(sequences):
    """Calculate comprehensive statistics for sequences"""
    if not sequences:
        return {}
    
    lengths = [len(seq) for seq in sequences]
    all_aa = ''.join(sequences)
    aa_counts = Counter(all_aa)
    total_aa = len(all_aa)
    
    # Dipeptides
    dipeptides = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            dipeptides.append(seq[i:i+2])
    dipeptide_counts = Counter(dipeptides)
    
    # Hydrophobicity
    hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'P'}
    polar = {'G', 'S', 'T', 'C', 'Y', 'N', 'Q'}
    charged = {'D', 'E', 'K', 'R', 'H'}
    
    hydro_count = sum(1 for aa in all_aa if aa in hydrophobic)
    polar_count = sum(1 for aa in all_aa if aa in polar)
    charged_count = sum(1 for aa in all_aa if aa in charged)
    
    # Amino acid frequencies
    aa_freqs = {aa: (aa_counts.get(aa, 0) / total_aa * 100) if total_aa > 0 else 0 
                for aa in AMINO_ACIDS}
    
    return {
        'lengths': lengths,
        'mean_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'std_length': np.std(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'aa_composition': aa_counts,
        'aa_frequencies': aa_freqs,
        'total_aa': total_aa,
        'unique_sequences': len(set(sequences)),
        'diversity': len(set(sequences)) / len(sequences) if sequences else 0,
        'dipeptides': dipeptide_counts,
        'hydrophobicity': {
            'hydrophobic': hydro_count / total_aa if total_aa > 0 else 0,
            'polar': polar_count / total_aa if total_aa > 0 else 0,
            'charged': charged_count / total_aa if total_aa > 0 else 0
        }
    }


def create_comparison_plots(custom_stats, progen_stats, save_path="model_comparison.png"):
    """Create comprehensive comparison visualizations"""
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Length Distribution Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    if custom_stats['lengths'] and progen_stats['lengths']:
        ax1.hist(custom_stats['lengths'], bins=30, alpha=0.6, label='Custom Transformer', 
                color='#3498db', edgecolor='black')
        ax1.hist(progen_stats['lengths'], bins=30, alpha=0.6, label='ProGen Fine-tuned', 
                color='#e74c3c', edgecolor='black')
        ax1.set_xlabel('Sequence Length', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Sequence Length Distribution Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
    
    # 2. Length Statistics Bar Chart
    ax2 = fig.add_subplot(gs[0, 2])
    metrics = ['Mean', 'Median', 'Std']
    custom_vals = [custom_stats['mean_length'], custom_stats['median_length'], 
                   custom_stats['std_length']]
    progen_vals = [progen_stats['mean_length'], progen_stats['median_length'], 
                   progen_stats['std_length']]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax2.bar(x - width/2, custom_vals, width, label='Custom TF', color='#3498db', alpha=0.8)
    ax2.bar(x + width/2, progen_vals, width, label='ProGen', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('Length', fontsize=12)
    ax2.set_title('Length Statistics', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Amino Acid Composition Comparison
    ax3 = fig.add_subplot(gs[1, :])
    aa_names = sorted(AMINO_ACIDS)
    custom_freqs = [custom_stats['aa_frequencies'].get(aa, 0) for aa in aa_names]
    progen_freqs = [progen_stats['aa_frequencies'].get(aa, 0) for aa in aa_names]
    
    x = np.arange(len(aa_names))
    width = 0.35
    ax3.bar(x - width/2, custom_freqs, width, label='Custom Transformer', 
           color='#3498db', alpha=0.8)
    ax3.bar(x + width/2, progen_freqs, width, label='ProGen Fine-tuned', 
           color='#e74c3c', alpha=0.8)
    ax3.set_xlabel('Amino Acid', fontsize=12)
    ax3.set_ylabel('Frequency (%)', fontsize=12)
    ax3.set_title('Amino Acid Composition Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(aa_names, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Top Dipeptides Comparison
    ax4 = fig.add_subplot(gs[2, :2])
    custom_top_dipeps = dict(custom_stats['dipeptides'].most_common(15))
    progen_top_dipeps = dict(progen_stats['dipeptides'].most_common(15))
    
    all_dipeps = set(list(custom_top_dipeps.keys()) + list(progen_top_dipeps.keys()))
    dipep_list = sorted(all_dipeps, key=lambda x: max(custom_top_dipeps.get(x, 0), 
                                                       progen_top_dipeps.get(x, 0)), 
                       reverse=True)[:15]
    
    custom_dipep_vals = [custom_top_dipeps.get(d, 0) for d in dipep_list]
    progen_dipep_vals = [progen_top_dipeps.get(d, 0) for d in dipep_list]
    
    x = np.arange(len(dipep_list))
    width = 0.35
    ax4.barh(x - width/2, custom_dipep_vals, width, label='Custom TF', 
            color='#3498db', alpha=0.8)
    ax4.barh(x + width/2, progen_dipep_vals, width, label='ProGen', 
            color='#e74c3c', alpha=0.8)
    ax4.set_xlabel('Count', fontsize=12)
    ax4.set_ylabel('Dipeptide', fontsize=12)
    ax4.set_title('Top 15 Dipeptide Frequencies', fontsize=12, fontweight='bold')
    ax4.set_yticks(x)
    ax4.set_yticklabels(dipep_list)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')
    
    # 5. Hydrophobicity Comparison
    ax5 = fig.add_subplot(gs[2, 2])
    categories = ['Hydrophobic', 'Polar', 'Charged']
    custom_hydro = [custom_stats['hydrophobicity']['hydrophobic'] * 100,
                   custom_stats['hydrophobicity']['polar'] * 100,
                   custom_stats['hydrophobicity']['charged'] * 100]
    progen_hydro = [progen_stats['hydrophobicity']['hydrophobic'] * 100,
                   progen_stats['hydrophobicity']['polar'] * 100,
                   progen_stats['hydrophobicity']['charged'] * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    ax5.bar(x - width/2, custom_hydro, width, label='Custom TF', 
           color='#3498db', alpha=0.8)
    ax5.bar(x + width/2, progen_hydro, width, label='ProGen', 
           color='#e74c3c', alpha=0.8)
    ax5.set_ylabel('Percentage (%)', fontsize=12)
    ax5.set_title('Residue Type Distribution', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(categories, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Diversity and Statistics Summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    summary_text = f"""
    MODEL COMPARISON SUMMARY
    {'='*80}
    
    CUSTOM TRANSFORMER:
      • Total Sequences Generated: {len(custom_stats['lengths'])}
      • Unique Sequences: {custom_stats['unique_sequences']}
      • Diversity Ratio: {custom_stats['diversity']*100:.2f}%
      • Mean Length: {custom_stats['mean_length']:.2f} ± {custom_stats['std_length']:.2f}
      • Length Range: {custom_stats['min_length']} - {custom_stats['max_length']}
      • Total Amino Acids: {custom_stats['total_aa']:,}
    
    PROGEN FINE-TUNED:
      • Total Sequences Generated: {len(progen_stats['lengths'])}
      • Unique Sequences: {progen_stats['unique_sequences']}
      • Diversity Ratio: {progen_stats['diversity']*100:.2f}%
      • Mean Length: {progen_stats['mean_length']:.2f} ± {progen_stats['std_length']:.2f}
      • Length Range: {progen_stats['min_length']} - {progen_stats['max_length']}
      • Total Amino Acids: {progen_stats['total_aa']:,}
    
    STATISTICAL COMPARISON:
      • Length Difference: {abs(custom_stats['mean_length'] - progen_stats['mean_length']):.2f}
      • Diversity Difference: {abs(custom_stats['diversity'] - progen_stats['diversity'])*100:.2f}%
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Serine Protease Generation: Custom Transformer vs ProGen Fine-tuned', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plots saved to {save_path}")
    plt.close()


def create_aa_correlation_plot(custom_stats, progen_stats, save_path="aa_correlation.png"):
    """Create amino acid frequency correlation plot"""
    
    aa_names = sorted(AMINO_ACIDS)
    custom_freqs = [custom_stats['aa_frequencies'].get(aa, 0) for aa in aa_names]
    progen_freqs = [progen_stats['aa_frequencies'].get(aa, 0) for aa in aa_names]
    
    # Calculate correlation
    correlation = np.corrcoef(custom_freqs, progen_freqs)[0, 1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(custom_freqs, progen_freqs, s=100, alpha=0.6, color='#2c3e50')
    
    # Add diagonal line
    max_val = max(max(custom_freqs), max(progen_freqs))
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Correlation')
    
    # Add labels for each amino acid
    for i, aa in enumerate(aa_names):
        ax.annotate(aa, (custom_freqs[i], progen_freqs[i]), 
                   fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Custom Transformer Frequency (%)', fontsize=12)
    ax.set_ylabel('ProGen Fine-tuned Frequency (%)', fontsize=12)
    ax.set_title(f'Amino Acid Frequency Correlation\n(Pearson r = {correlation:.3f})', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Amino acid correlation plot saved to {save_path}")
    plt.close()


def print_detailed_comparison(custom_stats, progen_stats):
    """Print detailed text comparison"""
    
    print("\n" + "="*80)
    print("DETAILED MODEL COMPARISON")
    print("="*80)
    
    print("\n1. SEQUENCE LENGTH STATISTICS")
    print("-"*80)
    print(f"{'Metric':<25} {'Custom Transformer':<25} {'ProGen Fine-tuned':<25}")
    print("-"*80)
    print(f"{'Mean Length':<25} {custom_stats['mean_length']:<25.2f} {progen_stats['mean_length']:<25.2f}")
    print(f"{'Median Length':<25} {custom_stats['median_length']:<25.2f} {progen_stats['median_length']:<25.2f}")
    print(f"{'Std Deviation':<25} {custom_stats['std_length']:<25.2f} {progen_stats['std_length']:<25.2f}")
    print(f"{'Min Length':<25} {custom_stats['min_length']:<25} {progen_stats['min_length']:<25}")
    print(f"{'Max Length':<25} {custom_stats['max_length']:<25} {progen_stats['max_length']:<25}")
    
    print("\n2. DIVERSITY METRICS")
    print("-"*80)
    print(f"{'Metric':<25} {'Custom Transformer':<25} {'ProGen Fine-tuned':<25}")
    print("-"*80)
    print(f"{'Total Sequences':<25} {len(custom_stats['lengths']):<25} {len(progen_stats['lengths']):<25}")
    print(f"{'Unique Sequences':<25} {custom_stats['unique_sequences']:<25} {progen_stats['unique_sequences']:<25}")
    print(f"{'Diversity Ratio (%)':<25} {custom_stats['diversity']*100:<25.2f} {progen_stats['diversity']*100:<25.2f}")
    
    print("\n3. AMINO ACID COMPOSITION (Top 10 by frequency)")
    print("-"*80)
    custom_top_aa = sorted(custom_stats['aa_frequencies'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
    progen_top_aa = sorted(progen_stats['aa_frequencies'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
    
    print(f"{'Rank':<6} {'Custom TF':<20} {'Freq (%)':<12} {'ProGen':<20} {'Freq (%)':<12}")
    print("-"*80)
    for i in range(10):
        custom_aa, custom_freq = custom_top_aa[i] if i < len(custom_top_aa) else ('-', 0)
        progen_aa, progen_freq = progen_top_aa[i] if i < len(progen_top_aa) else ('-', 0)
        print(f"{i+1:<6} {custom_aa:<20} {custom_freq:<12.2f} {progen_aa:<20} {progen_freq:<12.2f}")
    
    print("\n4. HYDROPHOBICITY ANALYSIS")
    print("-"*80)
    print(f"{'Residue Type':<20} {'Custom TF (%)':<20} {'ProGen (%)':<20}")
    print("-"*80)
    print(f"{'Hydrophobic':<20} {custom_stats['hydrophobicity']['hydrophobic']*100:<20.2f} {progen_stats['hydrophobicity']['hydrophobic']*100:<20.2f}")
    print(f"{'Polar':<20} {custom_stats['hydrophobicity']['polar']*100:<20.2f} {progen_stats['hydrophobicity']['polar']*100:<20.2f}")
    print(f"{'Charged':<20} {custom_stats['hydrophobicity']['charged']*100:<20.2f} {progen_stats['hydrophobicity']['charged']*100:<20.2f}")
    
    # Statistical test for length difference
    if len(custom_stats['lengths']) > 0 and len(progen_stats['lengths']) > 0:
        t_stat, p_value = stats.ttest_ind(custom_stats['lengths'], progen_stats['lengths'])
        print(f"\n5. STATISTICAL TEST (Length Distribution)")
        print("-"*80)
        print(f"T-test statistic: {t_stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Significant difference: {'Yes' if p_value < 0.05 else 'No'} (p < 0.05)")


def main():
    """Main comparison function"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load custom model
    print("\n" + "="*80)
    print("LOADING CUSTOM TRANSFORMER MODEL")
    print("="*80)
    try:
        custom_model = SerineProteaseTransformer(
            vocab_size=21,
            embedding_dim=64,
            n_layers=2,
            n_heads=4,
            dim_feedforward=128,
            max_len=300,
            dropout=0.1
        ).to(device)
        
        checkpoint = torch.load("serine_protease_transformer.pt", map_location=device)
        custom_model.load_state_dict(checkpoint['model_state_dict'])
        print("Custom Transformer loaded successfully!")
    except Exception as e:
        print(f"Error loading custom model: {e}")
        print("Please train the custom model first using train.py")
        return
    
    # Load ProGen model
    print("\n" + "="*80)
    print("LOADING FINE-TUNED PROGEN MODEL")
    print("="*80)
    try:
        save_dir = "progen_finetuned"
        if os.path.exists(save_dir):
            progen_model = AutoModelForCausalLM.from_pretrained(
                save_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
            ).to(device)
            progen_tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
            print("Fine-tuned ProGen loaded successfully!")
        else:
            print(f"Fine-tuned model not found at {save_dir}")
            print("Please fine-tune ProGen first using progen_finetune.py")
            return
    except Exception as e:
        print(f"Error loading ProGen model: {e}")
        return
    
    # Generate sequences from both models
    print("\n" + "="*80)
    print("GENERATING SEQUENCES")
    print("="*80)
    num_sequences = 100
    
    custom_sequences = generate_custom_model(custom_model, num_sequences, device)
    progen_sequences = generate_progen_model(progen_model, progen_tokenizer, 
                                            num_sequences, device)
    
    # Calculate statistics
    print("\n" + "="*80)
    print("CALCULATING STATISTICS")
    print("="*80)
    custom_stats = calculate_statistics(custom_sequences)
    progen_stats = calculate_statistics(progen_sequences)
    
    # Print detailed comparison
    print_detailed_comparison(custom_stats, progen_stats)
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    create_comparison_plots(custom_stats, progen_stats)
    create_aa_correlation_plot(custom_stats, progen_stats)
    
    # Save sequences for reference
    with open("comparison_custom_sequences.fasta", "w") as f:
        for i, seq in enumerate(custom_sequences):
            f.write(f">custom_sequence_{i+1}\n")
            f.write(f"{seq}\n")
    
    with open("comparison_progen_sequences.fasta", "w") as f:
        for i, seq in enumerate(progen_sequences):
            f.write(f">progen_sequence_{i+1}\n")
            f.write(f"{seq}\n")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("Generated files:")
    print("  - model_comparison.png: Comprehensive comparison plots")
    print("  - aa_correlation.png: Amino acid frequency correlation")
    print("  - comparison_custom_sequences.fasta: Custom model sequences")
    print("  - comparison_progen_sequences.fasta: ProGen model sequences")


if __name__ == "__main__":
    main()

