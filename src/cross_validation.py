import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score
from preprocess import load_data

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def perform_cross_validation(X_train, y_train):
    """
    执行5折交叉验证来寻找最佳超参数
    
    Args:
        X_train: 训练特征
        y_train: 训练标签
    
    Returns:
        dict: 最佳超参数配置
    """
    logging.info("Starting 5-fold cross-validation for hyperparameter tuning...")
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4, 6]
    }
    
    best_score = 0
    best_params = {}
    
    # 遍历所有参数组合
    total_combinations = (
        len(param_grid['n_estimators']) * 
        len(param_grid['max_depth']) * 
        len(param_grid['min_samples_split']) * 
        len(param_grid['min_samples_leaf'])
    )
    
    current_combination = 0
    
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    current_combination += 1
                    logging.info(f"Testing combination {current_combination}/{total_combinations}: "
                               f"n_estimators={n_est}, max_depth={max_d}, "
                               f"min_samples_split={min_split}, min_samples_leaf={min_leaf}")
                    
                    # 创建模型
                    rf = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_split=min_split,
                        min_samples_leaf=min_leaf,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1
                    )
                    
                    model = MultiOutputClassifier(rf)
                    
                    try:
                        # 使用分层5折交叉验证
                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        
                        # 对于多输出分类，我们使用第一个输出进行分层
                        # 计算平均F1分数
                        f1_scores = []
                        for train_idx, val_idx in skf.split(X_train, y_train[:, 0]):  # 使用第一个标签进行分层
                            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                            
                            model.fit(X_fold_train, y_fold_train)
                            y_pred = model.predict(X_fold_val)
                            
                            # 计算宏平均F1分数
                            fold_f1 = f1_score(y_fold_val, y_pred, average='macro', zero_division=0)
                            f1_scores.append(fold_f1)
                        
                        mean_f1 = np.mean(f1_scores)
                        std_f1 = np.std(f1_scores)
                        
                        logging.info(f"Mean F1-score: {mean_f1:.4f} (+/- {std_f1:.4f})")
                        
                        # 更新最佳参数
                        if mean_f1 > best_score:
                            best_score = mean_f1
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': max_d,
                                'min_samples_split': min_split,
                                'min_samples_leaf': min_leaf
                            }
                            logging.info(f"New best parameters found! F1-score: {best_score:.4f}")
                            
                    except Exception as e:
                        logging.warning(f"Failed to evaluate combination: {e}")
                        continue
    
    logging.info("Cross-validation completed!")
    logging.info(f"Best parameters: {best_params}")
    logging.info(f"Best cross-validation F1-score: {best_score:.4f}")
    
    return best_params

def train_with_best_params(args, best_params, X_train, X_test, y_train, y_test):
    """
    使用最佳参数训练最终模型并在测试集上评估
    
    Args:
        args: 命令行参数
        best_params: 最佳超参数
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
    """
    logging.info("Training final model with best parameters...")
    
    # 使用最佳参数创建模型
    rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    model = MultiOutputClassifier(rf)
    
    # 在整个训练集上训练
    model.fit(X_train, y_train)
    
    # 保存模型
    model_path = os.path.join(args.output_dir, "ComGen_model_tuned.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Tuned model saved to {model_path}")
    
    # 在测试集上评估
    logging.info("Evaluating tuned model on test set...")
    y_pred = model.predict(X_test)
    
    logging.info("Final Classification Report:")
    logging.info("\n" + classification_report(
        y_test, y_pred, 
        target_names=[f"Label {i+1}" for i in range(y_train.shape[1])],
        digits=4,
        zero_division=0
    ))
    
    # 计算整体F1分数
    test_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    logging.info(f"Test set macro F1-score: {test_f1:.4f}")
    
    return model

def cross_validate_and_train(args):
    """
    执行完整的交叉验证和训练流程
    """
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        logging.error(f"Data file not found: {args.data_path}")
        return

    try:
        logging.info("Loading data for cross-validation...")
        X, y = load_data(args.data_path)
        
        if X.shape[0] == 0 or y.shape[0] == 0:
            logging.error("Data is empty or invalid.")
            return
            
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return

    logging.info(f"Data shape: X={X.shape}, y={y.shape}")

    # 划分训练集和测试集 (80% 训练, 20% 测试)
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y[:, 0] if y.shape[1] > 0 else None  # 使用第一个标签进行分层
    )

    logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # 执行交叉验证寻找最佳参数
    best_params = perform_cross_validation(X_train, y_train)
    
    # 使用最佳参数训练最终模型
    final_model = train_with_best_params(args, best_params, X_train, X_test, y_train, y_test)
    
    return final_model, best_params