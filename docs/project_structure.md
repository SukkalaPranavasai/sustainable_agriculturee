# Project Structure Documentation

## Directory Overview

### `data/`
- **`raw/`**: Contains original, unprocessed datasets
  - Leaf images from various sources
  - Metadata files
  - Original dataset files
- **`processed/`**: Contains preprocessed and augmented data
  - Resized and normalized images
  - Augmented datasets
  - Train/validation/test splits

### `models/`
- Trained machine learning models
- Model checkpoints
- Model configuration files
- Pre-trained weights

### `reports/figures/`
- Generated charts and plots
- Performance metrics visualizations
- Data distribution plots
- Model comparison charts

### `notebooks/`
- Jupyter notebooks for:
  - Exploratory data analysis
  - Model training and evaluation
  - Data preprocessing
  - Results visualization

### `src/`
- Python source code modules
- Data preprocessing utilities
- Model training scripts
- Evaluation functions
- Utility functions

### `docs/`
- Project documentation
- API references
- User guides
- Technical specifications

## File Naming Conventions

- Use snake_case for Python files and directories
- Use descriptive names that indicate purpose
- Include version numbers for models when appropriate
- Use consistent date formats (YYYY-MM-DD) for time-sensitive files

## Data Flow

1. Raw data → `data/raw/`
2. Preprocessing → `data/processed/`
3. Model training → `models/`
4. Results analysis → `reports/figures/`
5. Documentation → `docs/`
