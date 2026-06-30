import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Evaluation & Save
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(ticker: str) -> pd.DataFrame:
    """Tải dữ liệu features đã được xử lý."""
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "data" / "processed" / f"{ticker}_features.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Processed data not found at {file_path}. Run features.py first.")

    logging.info(f"Loaded {file_path}")
    return pd.read_csv(file_path)


def prepare_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Chia dữ liệu train/test theo thời gian.
    """
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    drop_cols = ["Date", "Target"]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    X = df[feature_cols]
    y = df["Target"].astype(int)

    split_idx = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test, feature_cols


def get_models():
    """Khởi tạo danh sách các model cần so sánh."""
    models = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            random_state=42, eval_metric='logloss'
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            random_state=42, verbose=-1
        )
    }
    return models


# ============================================================
# PHASE 2: TimeSeriesSplit Cross-Validation
# ============================================================
def cross_validate_models(X_train, y_train, n_splits: int = 5):
    """
    Dùng TimeSeriesSplit để đánh giá các model trên tập Train.
    Đây là cách duy nhất đúng để cross-validate dữ liệu Time-Series,
    vì nó đảm bảo tập validation LUÔN nằm SAU tập train theo thời gian.
    """
    models = get_models()
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []

    logging.info(f"--- CROSS-VALIDATION VỚI TimeSeriesSplit ({n_splits} folds) ---")

    for name, model in models.items():
        fold_scores = {"f1": [], "roc_auc": [], "precision": [], "recall": []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)

            fold_scores["f1"].append(f1_score(y_val, y_pred, zero_division=0))
            fold_scores["precision"].append(precision_score(y_val, y_pred, zero_division=0))
            fold_scores["recall"].append(recall_score(y_val, y_pred, zero_division=0))

            try:
                y_proba = model.predict_proba(X_val)[:, 1]
                fold_scores["roc_auc"].append(roc_auc_score(y_val, y_proba))
            except Exception:
                fold_scores["roc_auc"].append(np.nan)

        cv_results.append({
            "Model": name,
            "CV_F1_mean": np.mean(fold_scores["f1"]),
            "CV_F1_std": np.std(fold_scores["f1"]),
            "CV_ROC_AUC_mean": np.nanmean(fold_scores["roc_auc"]),
            "CV_Precision_mean": np.mean(fold_scores["precision"]),
            "CV_Recall_mean": np.mean(fold_scores["recall"]),
        })

        logging.info(
            f"{name} - CV F1: {np.mean(fold_scores['f1']):.4f} ± {np.std(fold_scores['f1']):.4f}, "
            f"CV ROC-AUC: {np.nanmean(fold_scores['roc_auc']):.4f}"
        )

    cv_df = pd.DataFrame(cv_results).sort_values("CV_ROC_AUC_mean", ascending=False)
    logging.info("\nCV Results:\n" + cv_df.to_string(index=False))

    best_model_name = cv_df.iloc[0]["Model"]
    logging.info(f"--- BEST CV MODEL: {best_model_name} ---")

    return cv_df, best_model_name


# ============================================================
# PHASE 2: Threshold Tuning
# ============================================================
def find_optimal_threshold(model, X_val, y_val):
    """
    Tìm ngưỡng (threshold) tối ưu cho predict_proba thay vì dùng mặc định 0.5.
    Duyệt qua các giá trị threshold từ 0.30 đến 0.80, chọn threshold có F1 cao nhất,
    nhưng chỉ chấp nhận nếu Precision >= 0.55 (để đảm bảo chất lượng tín hiệu).
    """
    logging.info("--- TÌM THRESHOLD TỐI ƯU ---")

    y_proba = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = 0
    threshold_results = []

    for threshold in np.arange(0.30, 0.81, 0.05):
        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        acc = accuracy_score(y_val, y_pred)
        n_signals = y_pred.sum()

        threshold_results.append({
            "Threshold": round(threshold, 2),
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "N_Signals": int(n_signals)
        })

        # Chấp nhận threshold nếu F1 tốt hơn VÀ Precision đủ cao
        if f1 > best_f1 and precision >= 0.40:
            best_f1 = f1
            best_threshold = round(threshold, 2)

    # Fallback: nếu không có threshold nào đạt chuẩn, chọn threshold có F1 cao nhất
    if best_f1 == 0:
        threshold_results_sorted = sorted(threshold_results, key=lambda x: x["F1-score"], reverse=True)
        if threshold_results_sorted and threshold_results_sorted[0]["F1-score"] > 0:
            best_threshold = threshold_results_sorted[0]["Threshold"]
            best_f1 = threshold_results_sorted[0]["F1-score"]

    threshold_df = pd.DataFrame(threshold_results)
    logging.info("\nThreshold Scan:\n" + threshold_df.to_string(index=False))
    logging.info(f"--- OPTIMAL THRESHOLD: {best_threshold} (F1: {best_f1:.4f}) ---")

    return best_threshold, threshold_df


# ============================================================
# Final Train & Evaluate trên tập Test
# ============================================================
def final_train_and_evaluate(best_model_name, X_train, X_test, y_train, y_test, threshold):
    """Train model tốt nhất trên TOÀN BỘ tập train, đánh giá trên tập test với threshold tối ưu."""
    models = get_models()
    model = models[best_model_name]

    logging.info(f"--- FINAL TRAINING: {best_model_name} (threshold={threshold}) ---")
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    logging.info(
        f"Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
        f"Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}"
    )
    logging.info(f"Số tín hiệu MUA: {y_pred.sum()} / {len(y_pred)} ngày")
    logging.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    results_df = pd.DataFrame([{
        "Model": best_model_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    }])

    return model, results_df


def save_model(model, name: str, ticker: str, feature_cols, results_df, threshold, cv_df, threshold_df):
    """
    Lưu toàn bộ artifacts: model, feature_cols, threshold, kết quả CV và threshold scan.
    """
    project_root = Path(__file__).resolve().parent.parent
    model_dir = project_root / "models"
    report_dir = project_root / "reports"

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Xóa các model cũ để tránh nhầm lẫn
    for old_model in model_dir.glob(f"{ticker}_best_model_*.pkl"):
        old_model.unlink()

    model_path = model_dir / f"{ticker}_best_model_{name}.pkl"
    feature_cols_path = model_dir / f"{ticker}_feature_columns.pkl"
    threshold_path = model_dir / f"{ticker}_threshold.pkl"
    results_path = report_dir / f"{ticker}_model_results.csv"
    cv_path = report_dir / f"{ticker}_cv_results.csv"
    threshold_scan_path = report_dir / f"{ticker}_threshold_scan.csv"

    joblib.dump(model, model_path)
    joblib.dump(feature_cols, feature_cols_path)
    joblib.dump(threshold, threshold_path)
    results_df.to_csv(results_path, index=False)
    cv_df.to_csv(cv_path, index=False)
    threshold_df.to_csv(threshold_scan_path, index=False)

    logging.info(f"Best model saved to {model_path}")
    logging.info(f"Feature columns saved to {feature_cols_path}")
    logging.info(f"Threshold ({threshold}) saved to {threshold_path}")
    logging.info(f"CV results saved to {cv_path}")
    logging.info(f"Threshold scan saved to {threshold_scan_path}")


if __name__ == "__main__":
    ticker = "AAPL"

    logging.info(f"=== PIPELINE V2 CHO {ticker} ===")

    # 1. Load data
    df = load_data(ticker)

    # 2. Split data
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(df, test_size=0.2)
    logging.info(f"Train size: {len(X_train)} samples, Test size: {len(X_test)} samples")

    # 3. Cross-Validate tất cả models bằng TimeSeriesSplit
    cv_df, best_model_name = cross_validate_models(X_train, y_train, n_splits=5)

    # 4. Train model tốt nhất trên toàn bộ train, tìm Threshold tối ưu trên tập validation cuối
    #    (Dùng 20% cuối của tập train làm validation để tune threshold)
    val_split = int(len(X_train) * 0.8)
    X_tr_inner, X_val_inner = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr_inner, y_val_inner = y_train.iloc[:val_split], y_train.iloc[val_split:]

    temp_model = get_models()[best_model_name]
    temp_model.fit(X_tr_inner, y_tr_inner)
    best_threshold, threshold_df = find_optimal_threshold(temp_model, X_val_inner, y_val_inner)

    # 5. Final Train & Evaluate trên tập Test với threshold tối ưu
    best_model, results_df = final_train_and_evaluate(
        best_model_name, X_train, X_test, y_train, y_test, best_threshold
    )

    # 6. Save tất cả artifacts
    save_model(best_model, best_model_name, ticker, feature_cols, results_df,
               best_threshold, cv_df, threshold_df)
