
# Data Directory

This directory is designated for storing the raw data required for training and evaluating the Compliance Policy Generator (ComGen).

## Directory Structure
```
data/
├── raw/                       # Directory to store raw data files
├── ComGen_Annotation_Manual.pdf # Detailed annotation manual to guide users in preparing their datasets
└── README.md                  # Description of the raw data directory (this file)
```
## Raw Data Format

The raw data must be in CSV format, adhering to the following structure:

### Columns
- **Input Features**: 
  - A list of integers representing the characteristics of the data, such as:
    - Data category (e.g., personal data).
    - Sensitivity level (e.g., high).
    - Jurisdictions involved (e.g., source and target).
  
- **Action Fields**: 
  - Binary labels specifying the measures or actions required to ensure compliance, including but not limited to:
    - Security measures.
    - Data subject rights protection.
    - Compliance enforcement requirements.

- **Liability Fields**: 
  - Binary labels defining the entities responsible for enforcing or adhering to compliance policies.

- **Extension Fields**: 
  - Binary labels to account for additional, jurisdiction-specific requirements, allowing flexibility in adapting to legal nuances.

### Example CSV File Format

| Input Features        | Action Fields                       | Liability Fields      | Extension Fields          |
|-----------------------|-------------------------------------|-----------------------|---------------------------|
| [0, 1, 2, 3]          | [0, 1, 1, 0, ..., 0]                | [0, 1, 1, 0, ..., 0]  | [0, 1, 1, 0, ..., 0]      |
| [3, 2, 1, 0]          | [1, 0, 1, 1, ..., 0]                | [1, 0, 1, 1, ..., 0]  | [1, 0, 1, 1, ..., 0]      |

## Instructions for Users

1. **Data Placement**: 
   - Place your raw dataset files in the `data/raw/` directory before running any preprocessing, training, or evaluation scripts.

2. **Format Validation**:
   - Ensure the dataset adheres strictly to the specified format (including column order and data types) to maintain compatibility with the preprocessing and training scripts.

3. **File Naming**:
   - Use clear and descriptive filenames for your raw datasets (e.g., `train_data.csv`, `test_data.csv`).

4. **Data Privacy**:
   - If your dataset contains sensitive information, ensure it complies with all relevant privacy and security regulations before usage.

## Notes

- **Compatibility**: Data that does not conform to the specified format may cause errors during preprocessing or training, so please carefully read the annotation manual before you prepare your dataset.
- **Extendability**: Additional columns can be added to the dataset to capture new features or labels as long as the preprocessing pipeline is adjusted accordingly.