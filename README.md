# ComGen for CBCMS+ (Compliance Policy Generator)

This repository implements the **Compliance Policy Generator (ComGen)**, a **Random Forest-based** solution designed to generate compliance-related policies based on structured input features. The model is part of the CBCMS+ framework, enabling efficient, accurate, and real-time compliance policy generation, comprehensively supporting global software development.

## Features

- **Data Preprocessing**: Converts raw data into a structured format suitable for training and evaluation.
- **Model Training**: Utilizes a Random Forest classifier to predict binary labels across multiple outputs.
- **Hyperparameter Tuning**: Advanced 5-fold cross-validation for optimal parameter selection.
- **Model Evaluation**: Assesses the model using precision, recall, and F1 score metrics.
- **Flexible Training Modes**: Supports both standard training and cross-validation-based hyperparameter optimization.

## Project Structure

```
ComGen_CBCMS_plus/
├── data/
│   ├── raw/                         # Directory for storing raw data (CSV format: includes "Input Features" and grouped labels).
│   ├── ComGen_Annotation_Manual.pdf # Detailed annotation manual to guide users in preparing their datasets.
│   └── README.md                    # Instructions for preparing and organizing data files.
├── output/
│   ├── README.md                    # Description of the output directory and files.
├── src/
│   ├── preprocess.py                # Data preprocessing functions, including `load_data`.
│   ├── cross_validation.py          # 5-fold cross-validation with hyperparameter grid search.
│   ├── train.py                     # Standard training logic, including model initialization and evaluation.
│   ├── main.py                      # Main script coordinating the training workflow with both modes.
├── requirements.txt                 # List of Python dependencies required to run the project.
├── .gitignore                       # Configuration file for Git, specifying files and directories to ignore.
└── README.md                        # Overview of the project, installation instructions, and usage details.
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ComGen_CBCMS_plus
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training Modes

The system supports two training modes:

#### 1. Standard Training

Use predefined hyperparameters for quick model training.

```bash
python src/main.py --data_path "data/raw/train_data.csv" --output_dir "output" --n_estimators 100 --max_depth 15 --min_samples_split 5 --min_samples_leaf 2
```

#### 2. Cross-Validation Training (Recommended)

Automatically finds optimal hyperparameters using 5-fold cross-validation:

```bash
python src/main.py --data_path "data/raw/train_data.csv" --output_dir "output" --cross_validate
```

### Data Preparation

1. Prepare your input data in the `data/raw/` directory (refer to the annotation manual).
2. Follow the format described in `data/README.md` and the **annotation manual** `data/ComGen_Annotation_Manual.pdf`.
3. Ensure the dataset follows the specified format:
   - **Input Features**: A list of integers representing feature values (data category, sensitivity, jurisdictions).
   - **Action Fields**, **Liability Fields**, **Extension Fields**: Lists of binary labels.

> **Note:** The original annotated dataset used for ComGen model training cannot be released publicly due to licensing constraints.  
> Nonetheless, the accompanying annotation manual and schema specifications fully describe the dataset structure, enabling faithful reconstruction and replication of all results.

## Hyperparameter Configuration

### Standard Training Parameters

- `--n_estimators`: Number of trees in the Random Forest (default: 100)
- `--max_depth`: Maximum depth of each tree (default: 15)
- `--min_samples_split`: Minimum samples required to split an internal node (default: 5)
- `--min_samples_leaf`: Minimum samples required to be at a leaf node (default: 2)

### Cross-Validation Search Space

When using `--cross_validate`, the system automatically searches through:

- `n_estimators`: {50, 100, 150}
- `max_depth`: {10, 15, 20}
- `min_samples_split`: {2, 5, 10}
- `min_samples_leaf`: {2, 4, 6}

## Output Files

See `output/README.md`  for details.

## Example Commands

### Quick Start with Cross-Validation

```bash
python src/main.py --cross_validate
```

### Custom Standard Training

```bash
python src/main.py --data_path "data/raw/train_data.csv" --output_dir "my_output" --n_estimators 150 --max_depth 20 --min_samples_split 10 --min_samples_leaf 4
```

### Full Custom Cross-Validation

```bash
python src/main.py --data_path "data/raw/custom_data.csv" --output_dir "results" --cross_validate
```

## Performance Optimization

- The cross-validation mode automatically finds the optimal parameter configuration.
- Based on research, the optimal configuration typically consists of:
  - 100 trees
  - Maximum depth of 15
  - Minimum split size of 5
  - 2 samples per leaf
- The system uses stratified sampling to maintain class distribution.
- Parallel processing is enabled for faster training (`n_jobs=-1`).

## Troubleshooting

- Ensure your data file exists at the specified path.
- Verify data format matches the expected structure.
- Check that all required dependencies are installed.
- Review logs for detailed error information.

For detailed annotation guidelines and data preparation instructions, refer to `data/ComGen_Annotation_Manual.pdf`.
