# Sentinel-Swahili

A comprehensive data engineering and preprocessing pipeline for Swahili text data. This project provides tools for data ingestion, parsing, cleaning, inspection, and analysis of Swahili text datasets.

## Features

- **Data Pipeline**: Automated data ingestion and parsing from multiple file formats
- **Text Cleaning**: Specialized Swahili text cleaning with custom stop-word removal and stemming
- **Data Inspection**: Comprehensive dataset analysis and statistics generation
- **Dataset Splitting**: Tools for splitting and encoding datasets for machine learning
- **Report Generation**: JSON report generation for dataset analysis

## Project Structure

```
.
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
├── src/                    # Source code package
│   ├── __init__.py        # Package initialization
│   ├── data_pipeline.py   # Data ingestion and parsing
│   ├── data_inspector.py  # Dataset inspection and analysis
│   ├── text_cleaner.py    # Swahili text cleaning and preprocessing
│   └── dataset_splitter.py # Dataset splitting and encoding utilities
├── data/                   # Input data directory
│   ├── train.txt
│   ├── test.txt
│   └── valid.txt
├── reports/                # Generated reports directory
├── docs/                   # Documentation
│   └── GRP14-PROJECT-PROPOSAL.pdf
└── tests/                  # Test files (optional)
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/solomon-njogo/Sentinel-Swahili.git
cd Sentinel-Swahili
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Download NLTK stopwords data:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

## Usage

### Basic Usage

Process all default datasets (train.txt, test.txt, valid.txt):

```bash
py main.py
```

### RoBERTa Fine-tuning for Swahili Conversation

The project now bundles a `train.py` script that fine-tunes transformer models (including `roberta-base`) with LoRA adapters. Three training objectives are supported:

- `classification`: original supervised intent classification flow
- `mlm`: masked-language modeling
- `causal`: auto-regressive conversational modeling (recommended for Swahili dialogue)

Run causal fine-tuning on the bundled corpora:

```bash
py train.py --training-task causal --model-name roberta-base --data-dir data \
  --block-size 512 --num-epochs 3 --learning-rate 2e-5
```

Key arguments:

- `--training-task {classification, mlm, causal}`
- `--model-name` / `--tokenizer-name`
- `--block-size` and `--block-stride` (causal chunking)
- `--no-causal-padding`, `--no-shuffle-chunks`
- `--mlm-probability` (for MLM task)
- Standard hyperparameters: `--batch-size`, `--learning-rate`, `--num-epochs`

The script automatically reads `data/train.txt`, `data/valid.txt`, and `data/test.txt`, builds the appropriate dataset (including token-level chunking for causal LM), and saves checkpoints under `outputs/` and `checkpoints/`.

### Advanced `main.py` Usage

Process specific files:

```bash
py main.py --files train.txt test.txt
```

Specify custom data directory:

```bash
py main.py --data-dir /path/to/data
```

Generate detailed JSON reports:

```bash
py main.py --save-report
```

Save reports to custom directory:

```bash
py main.py --save-report --output-dir /path/to/reports
```

Key arguments:

- `--data-dir`: Directory containing data files (default: `data`)
- `--output-dir`: Directory to save reports (default: `reports`)
- `--files`: List of data files to process (default: `train.txt test.txt valid.txt`)
- `--save-report`: Save detailed report to JSON file

## Data Format

The pipeline supports multiple data formats:

- Numeric prefix format: `1 text content`
- Tab-separated: `text content\tlabel`
- Comma-separated: `text content,label`
- Pipe-separated: `text content|label`

## Components

### DataPipeline

Handles data ingestion, parsing, and feature/target separation with automatic format detection.

### DataInspector

Provides comprehensive dataset analysis including:

- Text statistics (word count, character count, vocabulary size)
- Label distribution
- Dataset quality metrics
- Summary reports

### SwahiliTextCleaner

Specialized text cleaning for Swahili including:

- Custom Swahili stop-word removal
- Swahili-specific stemming
- Text normalization

### DatasetSplitter

Utilities for dataset manipulation:

- Label encoding
- Dataset loading and structuring
- Feature-target separation

## Requirements

- Python 3.7+
- numpy >= 1.23.0
- nltk >= 3.8 (optional, for enhanced stop-word support)

## License

This project is part of SWE2020 coursework.

## Author

Solomon Njogo

Lewis Mwangi

## Repository

https://github.com/solomon-njogo/Sentinel-Swahili.git
