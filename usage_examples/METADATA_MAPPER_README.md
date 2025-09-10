# Metadata Mapper

AI-powered tool for mapping metadata values to controlled vocabularies using the [NF Metadata Dictionary](https://github.com/nf-osi/nf-metadata-dictionary).

## Overview

The Metadata Mapper helps standardize metadata by intelligently mapping free-text values from your input CSV to controlled vocabulary terms defined in the NF Metadata Dictionary. It uses AI (via OpenRouter) to suggest mappings and provides a human-in-the-loop interface for review and approval.

## Features

- 🤖 **AI-Powered Mapping**: Uses LLMs to intelligently suggest mappings from input values to controlled vocabulary terms
- 🔍 **Controlled Vocabulary Validation**: Ensures all mapped values exist in the NF Metadata Dictionary
- 👥 **Human-in-the-Loop**: Column-by-column review process for quality control
- 📊 **Flexible Input**: Works with any CSV structure - no hardcoded assumptions about column names
- 🌐 **Live Schema Fetching**: Automatically fetches the latest NF metadata dictionary with local caching
- ⚙️ **Configurable**: Easily customize LLM models, confidence thresholds, and behavior

## Setup

### 1. Install Dependencies

```bash
pip install PyYAML requests pandas
```

### 2. Configure Credentials

Copy the example credentials file and add your OpenRouter API key:

```bash
cp example_creds.yaml creds.yaml
```

Edit `creds.yaml`:
```yaml
openrouter:
  api_key: "your_openrouter_api_key_here"
```

Get your API key from [OpenRouter](https://openrouter.ai/keys).

### 3. Configure Settings (Optional)

The tool comes with sensible defaults in `config.yaml`, but you can customize:

```yaml
llm:
  model: "google/gemini-2-flash-exp"  # OpenRouter model
  temperature: 0.1
  max_tokens: 4000

mapping:
  confidence_threshold: 0.7
  auto_accept_threshold: 0.95
```

## Usage

### Command Line Interface

```bash
# Using the CLI command
geomapr-map input_metadata.csv target_template.csv output_mapped.csv

# Or run directly
python -m geomapr.metadata_mapper input_metadata.csv target_template.csv output_mapped.csv
```

### Python API

```python
from geomapr.metadata_mapper import MetadataMapper

mapper = MetadataMapper()
mapper.process_csv_mapping(
    "input_metadata.csv",
    "target_template.csv", 
    "output_mapped.csv"
)
```

### Example Workflow

1. **Input CSV** (`example_input_metadata.csv`):
```csv
file_name,experiment_type,tissue_source,sequencing_platform
file1.fastq,TRANSCRIPTOMIC SINGLE CELL,cerebral cortex,Illumina HiSeq
file2.fastq,GENOMIC VARIANT,blood,Illumina NovaSeq
```

2. **Target Template** (`example_target_template.csv`):
```csv
id,fileName,assay,specimenType,platform
NF_001,file1.fastq,,,
NF_002,file2.fastq,,,
```

3. **Run the mapper**:
```bash
geomapr-map example_input_metadata.csv example_target_template.csv mapped_output.csv
```

4. **Review AI suggestions** column by column:
```
Processing column: assay
Found 15 controlled vocabulary terms

AI Suggestions:
1. 'TRANSCRIPTOMIC SINGLE CELL' → 'single-cell RNA-seq'
   Confidence: 0.95
   Reasoning: Direct semantic match for single cell transcriptomics

Accept suggestions for 'assay'? (y/n/review): y
```

5. **Output** (`mapped_output.csv`):
```csv
id,fileName,assay,specimenType,platform
NF_001,file1.fastq,single-cell RNA-seq,cerebral cortex,
NF_002,file2.fastq,whole genome sequencing,blood,
```

## How It Works

### 1. Schema Fetching
- Downloads the latest NF Metadata Dictionary from GitHub
- Extracts controlled vocabularies for each field
- Caches locally to avoid repeated downloads

### 2. AI Mapping
- For each target column, finds relevant controlled vocabulary terms
- Extracts potential input values from the source CSV
- Uses LLM to suggest mappings with confidence scores and reasoning

### 3. Validation
- Ensures all suggested terms exist in the controlled vocabulary
- Filters out invalid suggestions before presenting to user

### 4. Human Review
- Presents AI suggestions column by column
- Shows confidence scores and reasoning
- Allows acceptance, rejection, or detailed review

### 5. Application
- Applies approved mappings to the target CSV
- Leaves cells blank when no mapping is found or approved
- Preserves existing data in pre-populated columns

## Configuration Options

### LLM Settings
```yaml
llm:
  model: "google/gemini-2-flash-exp"  # Any OpenRouter model
  temperature: 0.1                    # Lower = more consistent
  max_tokens: 4000                    # Response length limit
```

### Mapping Behavior
```yaml
mapping:
  confidence_threshold: 0.7      # Minimum confidence to suggest
  max_suggestions: 3             # Max suggestions per field
  strict_validation: true        # Validate against controlled vocab
```

### Interaction
```yaml
interaction:
  show_confidence: true          # Show confidence scores
  allow_manual_input: false      # Allow manual term entry
  auto_accept_threshold: 0.95    # Auto-accept above this confidence
```

## Supported Models

Any model available through OpenRouter:
- `google/gemini-2-flash-exp` (recommended, fast and accurate)
- `anthropic/claude-3-sonnet`
- `openai/gpt-4-turbo`
- And many others...

## Advanced Features

### Custom Schema URLs
Point to a different metadata dictionary:
```yaml
metadata_dictionary:
  jsonld_url: "https://example.com/custom-schema.jsonld"
```

### Row Matching Logic
The tool currently uses a simplified approach for matching rows between input and target CSVs. For more sophisticated matching (e.g., by file name or ID), you can extend the `_apply_mappings_to_target` method.

## Troubleshooting

### Common Issues

1. **"No controlled vocabulary found"**
   - Check that column names in target CSV match terms in the NF dictionary
   - The tool tries exact, case-insensitive, and partial matching

2. **"Error fetching schema"**
   - Check internet connection
   - Verify the schema URL is accessible
   - Check if cached file is corrupted (delete `nf_metadata_cache.json`)

3. **"No relevant input values found"**
   - Input CSV may have mostly system IDs/filenames
   - Tool filters out obvious non-metadata values automatically

### Debug Mode
Enable verbose logging by modifying the config:
```yaml
debug: true
```

## Contributing

The metadata mapper is designed to be extensible:
- Add new controlled vocabulary sources
- Implement custom row matching logic
- Extend the human-in-the-loop interface
- Add support for additional LLM providers

## License

Same as the main geomapr package.
