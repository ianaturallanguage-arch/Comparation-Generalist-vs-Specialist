# Model Parameters Comparison

## Custom SerineProteaseTransformer

### Architecture
- **Vocab Size**: 21 (20 amino acids + padding token)
- **Embedding Dimension**: 64
- **Number of Layers**: 2 decoder blocks
- **Attention Heads**: 4 per layer
- **Feed-forward Dimension**: 128
- **Max Sequence Length**: 300
- **Dropout**: 0.1

### Parameter Breakdown

#### Embedding Layer
- `vocab_size × embedding_dim = 21 × 64 = 1,344 parameters`

#### Positional Encoding
- 0 parameters (sinusoidal, no learnable parameters)

#### Decoder Blocks (2 blocks)
Each block contains:
- **Self-Attention**: 
  - Query/Key/Value projections: `3 × (embedding_dim × embedding_dim) = 3 × (64 × 64) = 12,288`
  - Output projection: `embedding_dim × embedding_dim = 64 × 64 = 4,096`
  - Bias terms: `3 × embedding_dim + embedding_dim = 256`
  - **Total per block**: ~16,640 parameters

- **Feed-Forward Network**:
  - First linear: `embedding_dim × dim_feedforward = 64 × 128 = 8,192`
  - Second linear: `dim_feedforward × embedding_dim = 128 × 64 = 8,192`
  - Bias terms: `dim_feedforward + embedding_dim = 192`
  - **Total per block**: ~16,576 parameters

- **Layer Normalization** (2 per block):
  - `2 × (embedding_dim × 2) = 2 × 128 = 256` parameters

- **Total per Decoder Block**: ~33,472 parameters
- **Total for 2 blocks**: ~66,944 parameters

#### Output Projection
- `embedding_dim × vocab_size + bias = 64 × 21 + 21 = 1,365 parameters`

### Total Parameters
- **Total**: ~69,653 parameters (~70K)
- **Trainable**: ~69,653 parameters
- **Memory (float32)**: ~0.27 MB

---

## ProGen2-Small (Fine-tuned)

### Architecture
- **Model Type**: GPT-style Transformer (Causal Language Model)
- **Model Name**: `nferruz/ProGen2-small`
- **Architecture**: Similar to GPT-2 architecture

### Estimated Architecture (ProGen2-Small)
- **Vocab Size**: ~25,000 (includes special tokens)
- **Hidden Dimension**: ~768
- **Number of Layers**: ~24
- **Attention Heads**: ~12 per layer
- **Feed-forward Dimension**: ~3,072 (4× hidden dimension)
- **Max Sequence Length**: ~1,024

### Parameter Breakdown

#### Embedding Layer
- Token embeddings: `vocab_size × hidden_dim ≈ 25,000 × 768 ≈ 19,200,000`
- Position embeddings: `max_len × hidden_dim ≈ 1,024 × 768 ≈ 786,432`
- **Total**: ~20,000,000 parameters

#### Transformer Blocks (~24 blocks)
Each block contains:
- **Self-Attention**:
  - Query/Key/Value projections: `3 × (hidden_dim × hidden_dim) = 3 × (768 × 768) ≈ 1,769,472`
  - Output projection: `hidden_dim × hidden_dim = 768 × 768 ≈ 589,824`
  - **Total per block**: ~2,359,296 parameters

- **Feed-Forward Network**:
  - First linear: `hidden_dim × ff_dim = 768 × 3,072 ≈ 2,359,296`
  - Second linear: `ff_dim × hidden_dim = 3,072 × 768 ≈ 2,359,296`
  - **Total per block**: ~4,718,592 parameters

- **Layer Normalization** (2 per block):
  - `2 × (hidden_dim × 2) = 2 × 1,536 ≈ 3,072` parameters

- **Total per Transformer Block**: ~7,080,960 parameters
- **Total for 24 blocks**: ~169,943,040 parameters

#### Output Head (Language Model Head)
- `hidden_dim × vocab_size ≈ 768 × 25,000 ≈ 19,200,000 parameters`

### Total Parameters
- **Total**: ~151,000,000 parameters (~151M)
- **Trainable**: ~151,000,000 parameters
- **Memory (float32)**: ~576 MB
- **Memory (float16)**: ~288 MB

---

## Comparison Summary

| Metric | Custom Transformer | ProGen2-Small | Ratio |
|--------|-------------------|---------------|-------|
| **Total Parameters** | ~70K | ~151M | **~2,157x** |
| **Memory (float32)** | ~0.27 MB | ~576 MB | **~2,133x** |
| **Memory (float16)** | ~0.27 MB | ~288 MB | **~1,067x** |
| **Embedding Dim** | 64 | ~768 | **12x** |
| **Number of Layers** | 2 | ~24 | **12x** |
| **Attention Heads** | 4 | ~12 | **3x** |
| **FF Dimension** | 128 | ~3,072 | **24x** |
| **Max Sequence Length** | 300 | ~1,024 | **3.4x** |
| **Vocab Size** | 21 | ~25,000 | **~1,190x** |

## Key Differences

### Custom Transformer
- ✅ **Lightweight**: Only ~70K parameters
- ✅ **Memory Efficient**: Fits easily on P100 GPU
- ✅ **Fast Training**: Trains quickly from scratch
- ✅ **Focused**: Trained specifically on serine proteases
- ❌ **Limited Capacity**: Smaller model may have less expressive power
- ❌ **No Pre-training**: Starts from random initialization

### ProGen2-Small
- ✅ **Pre-trained**: Leverages knowledge from large protein datasets
- ✅ **High Capacity**: 151M parameters for complex patterns
- ✅ **Better Generalization**: Pre-trained on diverse proteins
- ✅ **Transfer Learning**: Fine-tuning adapts pre-trained knowledge
- ❌ **Memory Intensive**: Requires more GPU memory
- ❌ **Slower Training**: Larger model takes longer to fine-tune

## Training Configuration Comparison

| Setting | Custom Transformer | ProGen2-Small |
|---------|-------------------|---------------|
| **Batch Size** | 8 | 4 (with grad accumulation=2) |
| **Effective Batch Size** | 8 | 8 |
| **Learning Rate** | 1e-3 | 1e-4 |
| **Optimizer** | AdamW (β1=0.9, β2=0.999) | AdamW (β1=0.9, β2=0.999) |
| **Gradient Clipping** | max_norm=1.0 | max_norm=1.0 |
| **Early Stopping** | patience=3 | patience=3 |
| **Max Epochs** | 3 | 3 |
| **Training Type** | From scratch | Fine-tuning |

## Usage Recommendations

### Use Custom Transformer when:
- Limited GPU memory (P100 with 16GB)
- Need fast training/inference
- Dataset is large enough for training from scratch
- Want full control over architecture

### Use ProGen2-Small when:
- Have sufficient GPU memory
- Want to leverage pre-trained protein knowledge
- Dataset is relatively small (benefits from transfer learning)
- Need high-quality generation with biological plausibility

