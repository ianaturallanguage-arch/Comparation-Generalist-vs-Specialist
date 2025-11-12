# Model Comparison: Custom Transformer vs ProGen Fine-tuned

This script provides comprehensive comparison between the custom SerineProteaseTransformer and the fine-tuned ProGen model.

## Usage

```bash
python compare_models.py
```

**Prerequisites:**
- Both models must be trained/fine-tuned first:
  - Custom Transformer: Run `python train.py` or `python main.py`
  - ProGen: Run `python progen_finetune.py` or `python progen_main.py`

## Output

The script generates:

### 1. **model_comparison.png** - Comprehensive comparison dashboard
   - Sequence length distribution (histogram)
   - Length statistics (mean, median, std)
   - Amino acid composition comparison
   - Top 15 dipeptide frequencies
   - Residue type distribution (hydrophobicity)
   - Summary statistics table

### 2. **aa_correlation.png** - Amino acid frequency correlation
   - Scatter plot showing correlation between models
   - Pearson correlation coefficient
   - Individual amino acid labels

### 3. **FASTA files**
   - `comparison_custom_sequences.fasta`: Sequences from custom model
   - `comparison_progen_sequences.fasta`: Sequences from ProGen model

### 4. **Console Output**
   - Detailed statistical comparison
   - T-test for length distribution differences
   - Top amino acids by frequency
   - Diversity metrics

## Comparison Metrics

1. **Length Statistics**
   - Mean, median, standard deviation
   - Min/max length ranges
   - Distribution histograms

2. **Diversity Analysis**
   - Unique sequence count
   - Diversity ratio (unique/total)
   - Sequence variety assessment

3. **Amino Acid Composition**
   - Frequency of each amino acid
   - Top 10 most frequent amino acids
   - Correlation between models

4. **Dipeptide Analysis**
   - Top 15 most common dipeptides
   - Frequency comparison

5. **Hydrophobicity**
   - Hydrophobic, polar, and charged residue percentages
   - Biological property comparison

6. **Statistical Tests**
   - T-test for length distribution differences
   - P-value significance testing

## Visualization Features

- **Side-by-side comparisons**: Direct visual comparison of metrics
- **Color coding**: Blue for Custom Transformer, Red for ProGen
- **Statistical annotations**: Correlation coefficients, p-values
- **Comprehensive layout**: All metrics in a single dashboard

## Example Output

The comparison will show:
- Which model generates longer/shorter sequences
- Which model has higher diversity
- Amino acid composition differences
- Biological property distributions
- Statistical significance of differences

This helps identify:
- Model strengths and weaknesses
- Generation quality differences
- Biological plausibility
- Training effectiveness

