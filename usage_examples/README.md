# GeoMapr Usage Examples

This directory contains example files and scripts demonstrating how to use the GeoMapr metadata mapping functionality.

## Files Overview

### Example Data Files
- **`example_input_metadata.csv`** - Sample input metadata from GEO/SRA processing
- **`example_target_template.csv`** - Target CSV template with controlled vocabulary columns
- **`demo_mapped_output.csv`** - Output file created by running the mapper (generated)
- **`mapped_output.csv`** - Output from simple example (generated)

### Example Scripts

#### Original GeoMapr Examples
- **`simple_example.py`** - Basic GEO metadata retrieval example
- **`example.py`** - More comprehensive GEO processing example

#### Metadata Mapper Examples  
- **`metadata_mapper_example.py`** - Simple AI mapping example
- **`complete_example.py`** - Full workflow with metadata retrieval + AI mapping

### Documentation
- **`METADATA_MAPPER_README.md`** - Detailed documentation for the metadata mapper

## Quick Start

### Option 1: Basic GEO Metadata Retrieval
```bash
# From the project root directory
python usage_examples/simple_example.py
python usage_examples/example.py
```

### Option 2: AI-Powered Metadata Mapping

#### Set up credentials first
```bash
# From the project root directory
cp example_creds.yaml creds.yaml
# Edit creds.yaml and add your OpenRouter API key
```

#### Run the mapper examples
```bash
# Simple AI mapping example
python usage_examples/metadata_mapper_example.py

# Complete workflow (GEO retrieval + AI mapping)
python usage_examples/complete_example.py
```

#### Use the CLI directly
```bash
# From the project root directory
geomapr-map usage_examples/example_input_metadata.csv usage_examples/example_target_template.csv usage_examples/my_output.csv
```

## Understanding the Example Data

### Input Metadata (`example_input_metadata.csv`)
Contains free-text metadata values that need to be mapped:
- `experiment_type`: "TRANSCRIPTOMIC SINGLE CELL", "GENOMIC VARIANT", etc.
- `tissue_source`: "cerebral cortex", "blood", etc.
- `sequencing_platform`: "Illumina HiSeq", "Oxford Nanopore", etc.

### Target Template (`example_target_template.csv`)
Contains columns that expect controlled vocabulary terms:
- `assay`: Should use terms like "single-cell RNA-seq", "whole genome sequencing"
- `specimenType`: Should use terms like "cerebral cortex", "blood"
- `platform`: Should use standardized platform names

### Expected Mappings
The AI should suggest mappings like:
- "TRANSCRIPTOMIC SINGLE CELL" → "single-cell RNA-seq"
- "GENOMIC VARIANT" → "whole genome sequencing"  
- "cerebral cortex" → "cerebral cortex" (already standardized)

## Column-by-Column Review Process

When you run the mapper, you'll see output like:

```
Processing column: assay
Found 195 controlled vocabulary terms

AI Suggestions (2):
  1. 'TRANSCRIPTOMIC SINGLE CELL' → 'single-cell RNA-seq'
     Confidence: 0.95
     Reasoning: Direct semantic match for single cell transcriptomics

  2. 'GENOMIC VARIANT' → 'whole genome sequencing'
     Confidence: 0.88
     Reasoning: Genomic variant detection typically uses whole genome sequencing

Accept suggestions for 'assay'? (y/n/review): y
```

## Troubleshooting

1. **Missing credentials**: Make sure you've created `creds.yaml` with your OpenRouter API key
2. **File not found errors**: Run scripts from the project root directory, not from inside `usage_examples/`
3. **No suggestions**: The AI might not find good matches - this is expected for some data
4. **API errors**: Check your OpenRouter API key and internet connection

## Next Steps

After running these examples:
1. Try with your own input data
2. Create your own target templates
3. Adjust configuration in `config.yaml`
4. Integrate into your analysis pipeline

See `METADATA_MAPPER_README.md` for detailed documentation.
