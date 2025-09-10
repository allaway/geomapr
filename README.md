# GeoMapr

A Python package for retrieving metadata from GEO/SRA and intelligently mapping it to standardized metadata templates using AI.

## Features

- 🧬 **GEO/SRA Integration**: Retrieve comprehensive metadata from GEO series and associated SRA files
- 🤖 **AI-Powered Mapping**: Intelligently map retrieved metadata to standardized templates using OpenRouter LLMs
- 📊 **File-Level Detail**: Get individual file information including S3 locations, checksums, and sizes  
- 📋 **Flexible Output**: Generate CSV and Excel files with confidence-based mapping results
- 🔍 **Comprehensive Data**: Combines GEO sample metadata with detailed SRA file information
- 🎯 **Controlled Vocabularies**: Map to standardized terms from NF metadata dictionary
- 💰 **Cost Tracking**: Real-time token usage and cost tracking with OpenRouter Usage Accounting
- ⚡ **Easy Setup**: Simple configuration with API credentials

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd geomapr
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

5. **Set up API credentials:**
   ```bash
   cp example_creds.yaml creds.yaml
   # Edit creds.yaml with your OpenRouter API key
   ```

## Quick Start

### Retrieve GEO/SRA Metadata
```bash
geomapr retrieve GSE212964
```

### AI-Powered Metadata Mapping
```bash
# Map retrieved metadata to a template
geomapr map input_metadata.csv target_template.csv output_mapped.csv
```

### Complete Pipeline
```bash
# Retrieve + Map in one command
geomapr pipeline GSE212964 target_template.csv final_output.csv
```

## Usage

### 🧬 GEO/SRA Metadata Retrieval

```python
from geomapr import MetadataProcessor

# Initialize processor (no credentials needed for retrieval!)
processor = MetadataProcessor()

# Process a GEO series
result_df = processor.process_geo_series('GSE212964', 'GSE212964_metadata.csv')

# Display results
print(f"Generated {len(result_df)} rows of metadata")
print(result_df.head())
```

### 🤖 AI-Powered Metadata Mapping

```python
from geomapr.metadata_mapper import MetadataMapper

# Initialize mapper (requires OpenRouter API key)
mapper = MetadataMapper()

# Map retrieved metadata to a standardized template
mapped_df = mapper.process_csv_mapping(
    input_csv='GSE212964_metadata.csv',
    target_csv='nf_template.csv',
    output_xlsx='mapped_results.xlsx'
)

# Results include confidence scores and color-coded Excel output
print(f"Mapped {len(mapped_df)} rows with AI assistance")
```

### 🚀 Command Line Interface

```bash
# Retrieve GEO/SRA metadata
geomapr retrieve GSE212964

# Map metadata to template with AI
geomapr map input.csv template.csv output.xlsx

# Complete pipeline: retrieve + map
geomapr pipeline GSE212964 template.csv final_output.xlsx
```

### 📊 Output Formats

**Retrieved Metadata CSV:**
Each row represents an individual file with comprehensive metadata:

| sample_title | file_name | file_format | file_size_mb | organism_name | library_strategy | ... |
|--------------|-----------|-------------|--------------|---------------|------------------|-----|
| Sample_1 | EP5_S5_L001_R1_001.fastq.gz | fastq | 25615.06 | Homo sapiens | RNA-Seq | ... |
| Sample_1 | EP5_S5_L001_R2_001.fastq.gz | fastq | 20067.88 | Homo sapiens | RNA-Seq | ... |

**AI-Mapped Output Excel:**
Color-coded results with confidence scores:

| Filename | species | assay | specimenID | age | ... |
|----------|---------|-------|------------|-----|-----|
| file1.fastq.gz | Homo sapiens ✅ | RNA-seq ✅ | SAMN123456 🟡 | 45 🟡 | ... |

- ✅ Green: High confidence (0.8+)
- 🟡 Yellow: Medium confidence (0.5-0.8)  
- 🔴 Red: Low confidence (<0.5)

### 🔍 Key Features

**GEO/SRA Retrieval:**
- Complete metadata from GEO series and SRA files
- File-level details with S3 locations and checksums
- No credentials required for retrieval

**AI Mapping:**
- Maps to controlled vocabularies from NF metadata dictionary
- Handles both controlled terms and freetext columns
- Provides confidence scores for all mappings
- Intelligent filename-to-row matching
- Cost tracking with OpenRouter Usage Accounting

## Configuration

### OpenRouter API Setup

1. Get an API key from [OpenRouter](https://openrouter.ai/keys)
2. Copy the example credentials file:
   ```bash
   cp example_creds.yaml creds.yaml
   ```
3. Edit `creds.yaml` with your API key:
   ```yaml
   openrouter:
     api_key: "your-api-key-here"
   ```

### Model Configuration

Edit `config.yaml` to customize the AI model:
```yaml
llm:
  model: "google/gemini-2.5-flash"  # Fast and cost-effective
  temperature: 0.1
  max_tokens: 4000
```

## Advanced Usage

### Process Different GEO Series

```python
from geomapr import MetadataProcessor

processor = MetadataProcessor()

# Small dataset for testing
processor.process_geo_series('GSE212964', 'small_dataset.csv')

# Larger dataset  
processor.process_geo_series('GSE75748', 'large_dataset.csv')
```

### Custom Template Mapping

```python
from geomapr.metadata_mapper import MetadataMapper

mapper = MetadataMapper()

# Use custom target template
result = mapper.process_csv_mapping(
    input_csv='my_geo_data.csv',
    target_csv='custom_template.csv', 
    output_xlsx='custom_mapped.xlsx'
)

print(f"Mapped {len(result)} rows")
print(f"Cost: ${mapper.ai_logger.total_actual_cost:.6f}")
```

## Requirements

**Core Dependencies:**
- requests
- pandas  
- beautifulsoup4
- lxml
- pysradb

**AI Mapping Dependencies:**
- PyYAML
- openpyxl

## License

This project is licensed under the MIT License - see the LICENSE file for details.