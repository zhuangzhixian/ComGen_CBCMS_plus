# ComGen for CBCMS+ (Compliance Policy Generator)

This repository implements the **Compliance Policy Generator (ComGen)**, a Random Forest-based solution designed to generate compliance-related policies based on structured input features. The model is part of the CBCMS+ framework, enabling efficient, accurate, and real-time compliance policy generation for cross-border data transfers.

## Features

- **Data Preprocessing**: Converts raw data into a structured format suitable for training and evaluation.
- **Model Training**: Utilizes a Random Forest classifier to predict binary labels across multiple outputs.
- **Model Evaluation**: Assesses the model using precision, recall, and F1 score metrics.
- **Hyperparameter Tuning**: Allows easy configuration of key model parameters such as tree count, depth, and split criteria.

## Project Structure
```
ComGen_CBCMS_plus/
├── data/
│   ├── raw/                       # Directory for storing raw data (CSV format: includes "Input Features" and grouped labels).
│   ├── ComGen_Annotation_Manual.pdf # Detailed annotation manual to guide users in preparing their datasets.
│   └── README.md                  # Instructions for preparing and organizing data files.
├── output/
│   ├── README.md                  # Description of the output directory and files.
├── src/
│   ├── preprocess.py              # Data preprocessing functions, including `load_data`.
│   ├── train.py                   # Training logic, including model initialization and evaluation.
│   ├── main.py                    # Main script coordinating the training and evaluation workflow.
├── requirements.txt               # List of Python dependencies required to run the project.
├── .gitignore                     # Configuration file for Git, specifying files and directories to ignore.
└── README.md                      # Overview of the project, installation instructions, and usage details.

```

## Usage

### Training the Model

1. Prepare your input data in the `data/raw/` directory, you can refer to the annotation manual. Ensure the dataset follows the specified format:
   - `Input Features`: A list of integers representing feature values, such as data category, sensitivity and jurisdictions.
   - `Action Fields`, `Responsibility Fields`, `Semantic Extension Fields`: Lists of binary labels.
2. Run the following command to train and evaluate the model:

```bash
python src/main.py --data_path "data/raw/train_data.csv" --output_dir "output" --n_estimators 100 --max_depth 15 --min_samples_split 5 --min_samples_leaf 2
```

3. The trained model will be saved in the `output/` directory as `ComGen_model.pkl`.

## Configuration

You can adjust hyperparameters via command-line arguments:

- `--n_estimators`: Number of trees in the Random Forest (default: 100).
- `--max_depth`: Maximum depth of each tree (default: 15).
- `--min_samples_split`: Minimum samples required to split an internal node (default: 5).
- `--min_samples_leaf`: Minimum samples required to be at a leaf node (default: 2).

## Example Command

```bash
python src/main.py --data_path "data/raw/train_data.csv" --output_dir "output" --n_estimators 150 --max_depth 20 --min_samples_split 10 --min_samples_leaf 4
```