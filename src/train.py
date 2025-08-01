import os
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from preprocess import load_data

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_and_evaluate(args):
    """
    训练模型并评估其性能
    """
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_path):
        logging.error(f"Data file not found: {args.data_path}")
        return

    try:
        logging.info("Loading data...")
        X, y = load_data(args.data_path)
        
        if X.shape[0] == 0 or y.shape[0] == 0:
            logging.error("Data is empty or invalid.")
            return
            
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        return

    logging.info(f"Training Parameters: {vars(args)}")

    # 划分数据集时添加stratify参数保持类别分布
    logging.info("Splitting data with stratification...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y  # 添加分层抽样
    )

    # 初始化带类别平衡的随机森林
    logging.info("Initializing Balanced Random Forest model...")
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        random_state=42,
        class_weight='balanced',  # 添加类别平衡
        n_jobs=-1  # 启用并行加速
    )
    
    model = MultiOutputClassifier(rf)

    try:
        logging.info("Training balanced model...")
        model.fit(X_train, y_train)
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return

    # 保存模型
    model_path = os.path.join(args.output_dir, "ComGen_model_balanced.pkl")
    joblib.dump(model, model_path)
    logging.info(f"Balanced model saved to {model_path}")

    # 评估时添加zero_division参数避免警告
    logging.info("Evaluating balanced model...")
    y_pred = model.predict(X_val)
    logging.info("Classification Report with Class Balancing:")
    logging.info("\n" + classification_report(
        y_val, y_pred, 
        target_names=[f"Label {i+1}" for i in range(y.shape[1])],
        digits=4,
        zero_division=0  # 处理未预测类别
    ))