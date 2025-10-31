import argparse
import logging
from train import train_and_evaluate
from cross_validation import cross_validate_and_train

# 数据文件路径
DATA_PATH = "data/train_data.csv"

def main(args):
    """
    主函数，用于调用训练和评估功能

    Args:
        args: 命令行参数
    """
    if args.cross_validate:
        logging.info("Starting cross-validation training...")
        cross_validate_and_train(args)
    else:
        logging.info("Starting standard training...")
        train_and_evaluate(args)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Train and evaluate a Random Forest model for ComGen.")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path to the input data CSV file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save the trained model.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the Random Forest.")
    parser.add_argument("--max_depth", type=int, default=15, help="Maximum depth of the trees.")
    parser.add_argument("--min_samples_split", type=int, default=5, help="Minimum number of samples required to split an internal node.")
    parser.add_argument("--min_samples_leaf", type=int, default=2, help="Minimum number of samples required to be at a leaf node.")
    parser.add_argument("--cross_validate", action="store_true", help="Whether to perform cross-validation for hyperparameter tuning.")

    args = parser.parse_args()

    main(args)