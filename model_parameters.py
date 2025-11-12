"""
Calculate and display parameters for both models:
1. Custom SerineProteaseTransformer
2. Fine-tuned ProGen2-Small
"""

import torch
from model import SerineProteaseTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_number(num):
    """Format number with commas and appropriate suffix"""
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num/1_000:.2f}K"
    else:
        return str(num)


def analyze_custom_model():
    """Analyze custom SerineProteaseTransformer parameters"""
    print("=" * 80)
    print("CUSTOM SERINE PROTEASE TRANSFORMER")
    print("=" * 80)
    
    model = SerineProteaseTransformer(
        vocab_size=21,
        embedding_dim=64,
        n_layers=2,
        n_heads=4,
        dim_feedforward=128,
        max_len=300,
        dropout=0.1
    )
    
    total_params, trainable_params = count_parameters(model)
    
    print("\nARCHITECTURE:")
    print("-" * 80)
    print(f"  Vocab Size:              21 (20 amino acids + padding)")
    print(f"  Embedding Dimension:     64")
    print(f"  Number of Layers:        2")
    print(f"  Attention Heads:         4 per layer")
    print(f"  Feed-forward Dimension:  128")
    print(f"  Max Sequence Length:     300")
    print(f"  Dropout:                 0.1")
    
    print("\nPARAMETER BREAKDOWN:")
    print("-" * 80)
    
    # Embedding layer
    embedding_params = model.embedding.weight.numel()
    print(f"  Embedding Layer:         {embedding_params:,} ({format_number(embedding_params)})")
    print(f"    - vocab_size × embedding_dim = 21 × 64 = {embedding_params:,}")
    
    # Positional encoding (no parameters, just buffers)
    print(f"  Positional Encoding:      0 (sinusoidal, no learnable params)")
    
    # Decoder blocks
    decoder_params = 0
    for i, block in enumerate(model.decoder_blocks):
        block_params = sum(p.numel() for p in block.parameters())
        decoder_params += block_params
        print(f"  Decoder Block {i+1}:")
        print(f"    - Self-Attention:     {sum(p.numel() for p in block.self_attn.parameters()):,}")
        print(f"    - Feed-Forward:       {sum(p.numel() for p in block.ff.parameters()):,}")
        print(f"    - Layer Norms:        {sum(p.numel() for p in [block.norm1, block.norm2]):,}")
        print(f"    - Total Block:        {block_params:,} ({format_number(block_params)})")
    
    # Output projection
    output_params = model.output_proj.weight.numel() + model.output_proj.bias.numel()
    print(f"  Output Projection:        {output_params:,} ({format_number(output_params)})")
    print(f"    - embedding_dim × vocab_size + bias = 64 × 21 + 21 = {output_params:,}")
    
    print("\nTOTAL PARAMETERS:")
    print("-" * 80)
    print(f"  Total Parameters:        {total_params:,} ({format_number(total_params)})")
    print(f"  Trainable Parameters:    {trainable_params:,} ({format_number(trainable_params)})")
    print(f"  Non-trainable:           0")
    
    # Memory estimate (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"\nMEMORY ESTIMATE (float32):")
    print(f"  Model Size:              {memory_mb:.2f} MB")
    
    return {
        'model': model,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'architecture': {
            'vocab_size': 21,
            'embedding_dim': 64,
            'n_layers': 2,
            'n_heads': 4,
            'dim_feedforward': 128,
            'max_len': 300,
            'dropout': 0.1
        }
    }


def analyze_progen_model():
    """Analyze ProGen2-Small model parameters"""
    print("\n\n" + "=" * 80)
    print("PROGEN2-SMALL (FINE-TUNED)")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "nferruz/ProGen2-small"
    
    try:
        print(f"\nLoading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32  # Use float32 for accurate parameter count
        )
        
        total_params, trainable_params = count_parameters(model)
        
        print("\nMODEL INFORMATION:")
        print("-" * 80)
        print(f"  Model Name:             {model_name}")
        print(f"  Model Type:             GPT-style Transformer (Causal LM)")
        
        # Try to get architecture details from model config
        if hasattr(model, 'config'):
            config = model.config
            print(f"\nARCHITECTURE (from config):")
            print("-" * 80)
            if hasattr(config, 'vocab_size'):
                print(f"  Vocab Size:              {config.vocab_size:,}")
            if hasattr(config, 'n_embd') or hasattr(config, 'd_model'):
                d_model = getattr(config, 'n_embd', getattr(config, 'd_model', 'N/A'))
                print(f"  Hidden Dimension:       {d_model}")
            if hasattr(config, 'n_layer') or hasattr(config, 'num_layers'):
                n_layers = getattr(config, 'n_layer', getattr(config, 'num_layers', 'N/A'))
                print(f"  Number of Layers:       {n_layers}")
            if hasattr(config, 'n_head') or hasattr(config, 'num_heads'):
                n_heads = getattr(config, 'n_head', getattr(config, 'num_heads', 'N/A'))
                print(f"  Attention Heads:         {n_heads}")
            if hasattr(config, 'n_inner') or hasattr(config, 'dim_feedforward'):
                ff_dim = getattr(config, 'n_inner', getattr(config, 'dim_feedforward', 'N/A'))
                print(f"  Feed-forward Dimension:  {ff_dim}")
            if hasattr(config, 'n_positions') or hasattr(config, 'max_position_embeddings'):
                max_len = getattr(config, 'n_positions', 
                                getattr(config, 'max_position_embeddings', 'N/A'))
                print(f"  Max Sequence Length:     {max_len}")
        
        print("\nPARAMETER BREAKDOWN:")
        print("-" * 80)
        
        # Count parameters by module type
        embedding_params = 0
        transformer_params = 0
        output_params = 0
        other_params = 0
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            if 'embed' in name.lower() or 'wte' in name.lower() or 'token' in name.lower():
                embedding_params += param_count
            elif 'transformer' in name.lower() or 'h.' in name.lower() or 'layer' in name.lower():
                transformer_params += param_count
            elif 'lm_head' in name.lower() or 'output' in name.lower() or 'head' in name.lower():
                output_params += param_count
            else:
                other_params += param_count
        
        print(f"  Embedding Layer:         {embedding_params:,} ({format_number(embedding_params)})")
        print(f"  Transformer Blocks:      {transformer_params:,} ({format_number(transformer_params)})")
        print(f"  Output Head:             {output_params:,} ({format_number(output_params)})")
        if other_params > 0:
            print(f"  Other:                    {other_params:,} ({format_number(other_params)})")
        
        print("\nTOTAL PARAMETERS:")
        print("-" * 80)
        print(f"  Total Parameters:        {total_params:,} ({format_number(total_params)})")
        print(f"  Trainable Parameters:    {trainable_params:,} ({format_number(trainable_params)})")
        print(f"  Non-trainable:           {total_params - trainable_params:,}")
        
        # Memory estimates
        memory_mb_fp32 = (total_params * 4) / (1024 * 1024)
        memory_mb_fp16 = (total_params * 2) / (1024 * 1024)
        print(f"\nMEMORY ESTIMATES:")
        print(f"  Model Size (float32):    {memory_mb_fp32:.2f} MB ({memory_mb_fp32/1024:.2f} GB)")
        print(f"  Model Size (float16):    {memory_mb_fp16:.2f} MB ({memory_mb_fp16/1024:.2f} GB)")
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_name': model_name
        }
    
    except Exception as e:
        print(f"\nError loading ProGen model: {e}")
        print("Using estimated values for ProGen2-Small:")
        print("\nESTIMATED PARAMETERS (ProGen2-Small):")
        print("-" * 80)
        print("  Total Parameters:        ~151,000,000 (~151M)")
        print("  Model Type:              GPT-style Transformer")
        print("  Architecture:            Similar to GPT-2 architecture")
        return {
            'model': None,
            'total_params': 151_000_000,
            'trainable_params': 151_000_000,
            'model_name': model_name,
            'estimated': True
        }


def compare_models(custom_info, progen_info):
    """Compare both models side by side"""
    print("\n\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'Custom Transformer':<25} {'ProGen2-Small':<25}")
    print("-" * 80)
    
    # Parameter comparison
    custom_total = custom_info['total_params']
    progen_total = progen_info['total_params']
    
    print(f"{'Total Parameters':<30} {custom_total:>20,} {progen_total:>20,}")
    print(f"{'  (Formatted)':<30} {format_number(custom_total):>20} {format_number(progen_total):>20}")
    
    ratio = progen_total / custom_total if custom_total > 0 else 0
    print(f"{'  (Ratio)':<30} {'1x':>20} {f'{ratio:.1f}x':>20}")
    
    # Memory comparison
    custom_mem = (custom_total * 4) / (1024 * 1024)
    progen_mem = (progen_total * 4) / (1024 * 1024)
    
    print(f"\n{'Memory (float32)':<30} {custom_mem:>18.2f} MB {progen_mem:>18.2f} MB")
    
    # Architecture comparison
    print(f"\n{'Architecture Details':<30} {'Custom Transformer':<25} {'ProGen2-Small':<25}")
    print("-" * 80)
    
    custom_arch = custom_info['architecture']
    print(f"{'Embedding Dim':<30} {custom_arch['embedding_dim']:>20} {'~768 (estimated)':>20}")
    print(f"{'Number of Layers':<30} {custom_arch['n_layers']:>20} {'~24 (estimated)':>20}")
    print(f"{'Attention Heads':<30} {custom_arch['n_heads']:>20} {'~12 (estimated)':>20}")
    print(f"{'FF Dimension':<30} {custom_arch['dim_feedforward']:>20} {'~3072 (estimated)':>20}")
    print(f"{'Max Sequence Length':<30} {custom_arch['max_len']:>20} {'~1024 (estimated)':>20}")
    
    print("\nKEY DIFFERENCES:")
    print("-" * 80)
    print("  • ProGen2-Small is ~" + f"{ratio:.0f}" + "x larger than Custom Transformer")
    print("  • ProGen2-Small is pre-trained on large protein datasets")
    print("  • Custom Transformer is trained from scratch on serine proteases")
    print("  • ProGen2-Small has more capacity but requires more memory")
    print("  • Custom Transformer is more memory-efficient for P100 GPU")


def main():
    """Main function to analyze both models"""
    print("\n" + "=" * 80)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 80)
    
    # Analyze custom model
    custom_info = analyze_custom_model()
    
    # Analyze ProGen model
    progen_info = analyze_progen_model()
    
    # Compare models
    compare_models(custom_info, progen_info)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

