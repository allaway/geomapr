# GeoMapr

A Python package for retrieving and transforming metadata from GEO/SRA for listing in other data repositories.

## Features

- 🧬 **GEO/SRA Integration**: Retrieve comprehensive metadata from GEO series and associated SRA files
- 📊 **File-Level Detail**: Get individual file information including S3 locations, checksums, and sizes  
- 📋 **Flexible Output**: Generate CSV files with one row per individual file
- 🔍 **Comprehensive Data**: Combines GEO sample metadata with detailed SRA file information
- ⚡ **Easy Setup**: No credentials required for core functionality
- 🎯 **API-Only Values**: All output comes directly from APIs, no constructed values

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

## Quick Start

**No setup required!** Just run:

```bash
python usage_examples/simple_example.py
```

For more examples and AI-powered metadata mapping, see the [`usage_examples/`](usage_examples/) directory.

## Usage

### 🧬 Basic GEO/SRA Metadata Retrieval

```python
from geomapr import MetadataProcessor

# Initialize processor (no credentials needed!)
processor = MetadataProcessor()

# Process a GEO series
result_df = processor.process_geo_series('GSE212964', 'output.csv')

# Display results
print(f"Generated {len(result_df)} rows of metadata")
print(result_df.head())
```

### 🚀 Command Line Interface

```bash
# Process a GEO series
geomapr GSE212964

# Specify output file  
geomapr GSE212964 -o my_data.csv

# Show summary only
geomapr GSE212964 --summary-only
```

### 📊 Example Output

Each row represents an individual file with metadata:

| sample_title | file_name | file_format | file_size_mb | file_s3_location | file_md5 | ... |
|--------------|-----------|-------------|--------------|------------------|----------|-----|
| Sample_1 | EP5_S5_L001_R1_001.fastq.gz | fastq | 25615.06 | s3://sra-pub-src-1/... | a1b2c3... | ... |
| Sample_1 | EP5_S5_L001_R2_001.fastq.gz | fastq | 20067.88 | s3://sra-pub-src-1/... | d4e5f6... | ... |

### 🔍 Metadata Fields

The output CSV includes:

**GEO Metadata:**
- `series_title`, `series_summary`
- `sample_title`, `sample_description` 
- `organism`, `platform`, `library_strategy`

**SRA File Details:**
- `file_name`, `file_format`, `file_size_mb`
- `file_s3_location`, `file_download_url`
- `file_md5`, `sra_run_id`

**And many more fields from both GEO and SRA APIs!**

## Examples

See the [`usage_examples/`](usage_examples/) directory for:

- **Basic GEO metadata retrieval** (`simple_example.py`, `example.py`)
- **AI-powered metadata mapping** (`metadata_mapper_example.py`)
- **Complete workflows** (`complete_example.py`)
- **Sample data files** and comprehensive documentation

### Process Different GEO Series

```python
from geomapr import MetadataProcessor

processor = MetadataProcessor()

# Small dataset for testing
processor.process_geo_series('GSE212964', 'small_dataset.csv')

# Larger dataset  
processor.process_geo_series('GSE75748', 'large_dataset.csv')
```

### Get Series Summary

```python
# Quick summary without processing all files
summary = processor.get_series_summary('GSE212964')
print(f"Title: {summary['title']}")
print(f"Samples: {summary['unique_samples']}")
print(f"Total files: {summary['total_sra_runs']}")
```

## Output Format

The package generates CSV files where:
- **One row per individual file** (FASTQ, SRA, index files, etc.)
- **Exact S3 locations** from SRA API
- **MD5 checksums** for file integrity
- **All values from APIs** - no constructed or fallback values

## Requirements

- requests
- pandas  
- beautifulsoup4
- lxml
- pysradb

## License

This project is licensed under the MIT License - see the LICENSE file for details.