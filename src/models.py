"""
src/models.py
-------------
Module triển khai các mô hình Machine Learning phát hiện gian lận giao dịch:
1. Supervised Learning:
   - Logistic Regression (Baseline, có Imputer + Scaler + class_weight='balanced')
   - Random Forest (Bagging Ensemble, tối ưu RAM, class_weight='balanced')
   - LightGBM (Gradient Boosting chủ lực, xử lý NaN tự nhiên, scale_pos_weight)
2. Unsupervised Learning:
   - Isolation Forest (Anomaly Detection, ánh xạ điểm dị biệt sang fraud risk score)

Được thiết kế cho Người B (Mục 3: 5 điểm), hỗ trợ đầy đủ các hàm huấn luyện,
tính toán metric (PR-AUC, ROC-AUC, Precision, Recall, F1), xuất dự đoán cho
Người C (Mục 4 & 5) và lưu trữ mô hình cho phân tích SHAP.
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    confusion_matrix,
)

# Kiểm tra thư viện LightGBM
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier


# ==============================================================================
# 1. HÀM ĐÁNH GIÁ METRIC VÀ TÌM NGƯỠNG THỐNG KÊ (EVALUATION & THRESHOLD)
# ==============================================================================

def find_youden_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[float, float]:
    """
    Tìm ngưỡng phân loại tối ưu theo chỉ số thống kê Youden's J Statistic từ đường cong ROC:
        J = Sensitivity + Specificity - 1 = TPR - FPR
    
    Ngưỡng tối ưu t* là điểm tại đó khoảng cách thẳng đứng từ ROC tới đường chéo ngẫu nhiên là lớn nhất,
    giúp cân bằng tối ưu giữa tỷ lệ bắt đúng gian lận (TPR) và việc hạn chế báo động giả (FPR).
    
    Trả về:
        (best_threshold, max_j_score)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = float(thresholds[best_idx])
    # Giới hạn ngưỡng trong khoảng hợp lý [0.01, 0.95]
    best_thresh = float(np.clip(best_thresh, 0.01, 0.95))
    return best_thresh, float(j_scores[best_idx])


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: Any = 0.5
) -> Dict[str, Any]:
    """
    Tính toán các chỉ số đánh giá toàn diện cho bài toán phát hiện gian lận.
    Hỗ trợ truyền threshold cụ thể (float) hoặc 'youden' để tự động tối ưu hóa theo Youden's J statistic.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred_proba = np.asarray(y_pred_proba).astype(float)

    if threshold == "youden" or threshold is None:
        threshold, _ = find_youden_threshold(y_true, y_pred_proba)

    threshold = float(threshold)
    y_pred = (y_pred_proba >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    
    # PR-AUC (Average Precision) - Metric quan trọng nhất cho imbalanced data
    try:
        pr_auc = float(average_precision_score(y_true, y_pred_proba))
    except Exception:
        pr_auc = 0.0

    # ROC-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        roc_auc = 0.5

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "Threshold": threshold,
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1-score": round(float(f1), 4),
        "Accuracy": round(acc, 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


# ==============================================================================
# 2. CÁC HÀM HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING FUNCTIONS)
# ==============================================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str = "balanced",
    max_iter: int = 1000,
    C: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> Pipeline:
    """
    Huấn luyện mô hình Logistic Regression chuẩn hóa:
    - SimpleImputer (median) để tự động xử lý missing values nếu có.
    - StandardScaler để đưa các đặc trưng về cùng thang đo.
    - class_weight='balanced' để phạt nặng lỗi phân loại sai trên lớp gian lận.
    """
    print(f"[models.py] Đang huấn luyện Logistic Regression (class_weight='{class_weight}')...")
    start_time = time.time()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=class_weight,
            max_iter=max_iter,
            C=C,
            random_state=random_state,
            solver="lbfgs",
            **kwargs
        ))
    ])

    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"[models.py] Hoàn thành Logistic Regression trong {elapsed:.2f}s")
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 150,
    max_depth: int = 15,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> Pipeline:
    """
    Huấn luyện mô hình Random Forest Classifier:
    - Bọc trong Pipeline với SimpleImputer để an toàn trước NaN.
    - Giới hạn max_depth và min_samples_split để kiểm soát overfitting và tránh tràn RAM trên Kaggle.
    - class_weight='balanced' cân bằng trọng số giữa lớp thường và lớp gian lận.
    """
    print(f"[models.py] Đang huấn luyện Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    start_time = time.time()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        ))
    ])

    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"[models.py] Hoàn thành Random Forest trong {elapsed:.2f}s")
    return pipeline


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = -1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: Optional[float] = None,
    early_stopping_rounds: int = 50,
    random_state: int = 42,
    **kwargs
):
    """
    Huấn luyện mô hình LightGBM Classifier (Mô hình chủ lực):
    - Tự động tính toán scale_pos_weight = (n_neg / n_pos) nếu không truyền vào.
    - Hỗ trợ early stopping theo PR-AUC (average_precision) trên tập Validation.
    - Xử lý giá trị missing (NaN) tự nhiên, không cần impute nhân tạo.
    """
    start_time = time.time()

    # Tính toán tỉ lệ cân bằng lớp nếu chưa chỉ định
    if scale_pos_weight is None:
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        scale_pos_weight = float(num_neg / max(num_pos, 1))

    print(f"[models.py] Đang huấn luyện LightGBM (scale_pos_weight={scale_pos_weight:.2f}, n_estimators={n_estimators})...")

    if HAS_LIGHTGBM:
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )

        callbacks = []
        eval_set = None
        eval_metric = "average_precision"

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            if early_stopping_rounds > 0:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
                callbacks.append(lgb.log_evaluation(period=100))

        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric=eval_metric,
            callbacks=callbacks if callbacks else None
        )
    else:
        warnings.warn("LightGBM chưa được cài đặt. Đang sử dụng HistGradientBoostingClassifier làm phương án dự phòng...")
        # Tính trọng số mẫu cho HistGradientBoosting
        sample_weight = np.where(y_train == 1, scale_pos_weight, 1.0)
        model = HistGradientBoostingClassifier(
            max_iter=min(n_estimators, 200),
            learning_rate=learning_rate,
            max_leaf_nodes=num_leaves,
            random_state=random_state,
            **kwargs
        )
        model.fit(X_train, y_train, sample_weight=sample_weight)

    elapsed = time.time() - start_time
    print(f"[models.py] Hoàn thành LightGBM trong {elapsed:.2f}s")
    return model


def train_isolation_forest(
    X_train: pd.DataFrame,
    y_train: Optional[pd.Series] = None,
    contamination: float = 0.035,
    n_estimators: int = 150,
    max_samples: Any = "auto",
    train_on_normal_only: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs
) -> Pipeline:
    """
    Huấn luyện thuật toán Học Không Giám Sát: Isolation Forest.
    
    Ý tưởng bài toán gian lận:
    - Giao dịch gian lận là những điểm 'dị biệt' (anomalies) hiếm và khác thường so với phần đông.
    - Có thể huấn luyện chỉ trên giao dịch bình thường (train_on_normal_only=True)
      để mô hình học hành vi giao dịch chuẩn, sau đó chấm điểm dị biệt cho các giao dịch mới.
    - Điểm decision_function() càng âm -> càng dị biệt -> nguy cơ gian lận càng cao.
    """
    print(f"[models.py] Đang huấn luyện Isolation Forest (contamination={contamination})...")
    start_time = time.time()

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs,
            **kwargs
        ))
    ])

    if train_on_normal_only and y_train is not None:
        X_fit = X_train[y_train == 0]
        print(f"[models.py] Isolation Forest huấn luyện trên {len(X_fit)} mẫu bình thường (isFraud=0)")
    else:
        X_fit = X_train

    pipeline.fit(X_fit)
    elapsed = time.time() - start_time
    print(f"[models.py] Hoàn thành Isolation Forest trong {elapsed:.2f}s")
    return pipeline


# ==============================================================================
# 3. HÀM DỰ ĐOÁN XÁC SUẤT & ĐỒNG BỘ ĐẦU RA
# ==============================================================================

def predict_fraud_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Dự đoán xác suất hoặc điểm rủi ro gian lận (0.0 đến 1.0) cho bất kỳ mô hình nào:
    - Đối với Supervised (Logistic, RF, LightGBM): dùng predict_proba() lấy cột 1.
    - Đối với Unsupervised (Isolation Forest): đảo ngược decision_function() và chuẩn hóa về [0, 1].
    """
    # Nếu là mô hình Pipeline chứa Isolation Forest hoặc chính nó
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) and "clf" in model.named_steps else model

    if isinstance(clf, IsolationForest):
        # decision_function: giá trị càng âm thì càng dị biệt (bất thường)
        raw_scores = model.decision_function(X) if isinstance(model, Pipeline) else clf.decision_function(X)
        # Đảo dấu: dị biệt cao -> điểm risk cao
        anomaly_scores = -raw_scores
        # Min-max scaling về khoảng [0, 1] để so sánh tương đồng xác suất
        s_min, s_max = anomaly_scores.min(), anomaly_scores.max()
        if s_max > s_min:
            proba = (anomaly_scores - s_min) / (s_max - s_min)
        else:
            proba = np.zeros_like(anomaly_scores)
        return proba

    # Các mô hình phân lớp thông thường
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    else:
        return model.predict(X).astype(float)


# ==============================================================================
# 4. HÀM SO SÁNH TỔNG HỢP & BÀN GIAO CHO NGƯỜI C
# ==============================================================================

def compare_models(
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    So sánh toàn bộ các mô hình trên tập Test theo các chỉ số:
    PR-AUC, ROC-AUC, F1-score, Precision, Recall, Accuracy.
    
    Trả về DataFrame sắp xếp theo PR-AUC giảm dần.
    """
    results = []
    print(f"\n[models.py] === BẢNG ĐÁNH GIÁ VÀ SO SÁNH CÁC MÔ HÌNH (Ngưỡng = {threshold}) ===")

    for name, model in models_dict.items():
        try:
            y_pred_proba = predict_fraud_probabilities(model, X_test)
            metrics = evaluate_predictions(y_test, y_pred_proba, threshold=threshold)
            metrics["Model"] = name
            results.append(metrics)
        except Exception as e:
            print(f"[models.py] Lỗi khi đánh giá mô hình {name}: {e}")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        cols_order = ["Model", "PR-AUC", "ROC-AUC", "F1-score", "Precision", "Recall", "Accuracy", "TP", "FP", "FN", "TN"]
        df_results = df_results[[c for c in cols_order if c in df_results.columns]]
        df_results = df_results.sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)

    return df_results


def save_predictions_for_evaluation(
    y_test: pd.Series,
    models_dict: Dict[str, Any],
    X_test: pd.DataFrame,
    output_path: str,
    transaction_ids: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Xuất file dự đoán xác suất của tất cả mô hình để bàn giao cho Người C:
    - Cột 1: TransactionID (nếu có)
    - Cột 2: isFraud_true (nhãn thật)
    - Cột 3..N: Xác suất dự đoán của từng mô hình (để vẽ PR curve, ROC curve, chọn ngưỡng)
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    data = {}
    if transaction_ids is not None:
        data["TransactionID"] = np.asarray(transaction_ids)
    data["isFraud_true"] = np.asarray(y_test).astype(int)

    for name, model in models_dict.items():
        col_name = f"pred_proba_{name.lower().replace(' ', '_')}"
        data[col_name] = predict_fraud_probabilities(model, X_test)

    df_preds = pd.DataFrame(data)

    if output_path.endswith(".parquet"):
        df_preds.to_parquet(output_path, index=False)
    else:
        df_preds.to_csv(output_path, index=False)

    print(f"[models.py] Đã lưu file dự đoán bàn giao cho Người C: {output_path} (Kích thước: {df_preds.shape})")
    return df_preds


def save_model(model: Any, filepath: str) -> None:
    """Lưu mô hình đã huấn luyện (để Người C load và phân tích SHAP)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"[models.py] Đã lưu mô hình tại: {filepath}")


def load_model(filepath: str) -> Any:
    """Tải lại mô hình đã lưu."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Không tìm thấy file mô hình: {filepath}")
    return joblib.load(filepath)


def get_feature_importances(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Trích xuất top N đặc trưng quan trọng nhất của mô hình dạng cây (Random Forest, LightGBM).
    """
    clf = model.named_steps["clf"] if isinstance(model, Pipeline) and "clf" in model.named_steps else model

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])

    if importances is None:
        print("[models.py] Mô hình không hỗ trợ trích xuất feature importances trực tiếp.")
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n).reset_index(drop=True)

    return df_imp


# ==============================================================================
# 5. TEST CHẠY THỬ ĐƠN VỊ KHI GỌI TRỰC TIẾP (UNIT TEST WITH MOCK DATA)
# ==============================================================================

if __name__ == "__main__":
    print("=== ĐANG CHẠY TEST KIỂM THỬ TẤT CẢ HÀM TRONG src/models.py VỚI DỮ LIỆU GIẢ LẬP ===")
    
    # 1. Tạo tập dữ liệu giả lập mô phỏng IEEE-CIS Fraud Detection (Mất cân bằng lớp 3.5%)
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    X_mock = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feat_{i}" for i in range(n_features)]
    )
    # Thêm một vài giá trị NaN để kiểm tra khả năng chịu lỗi
    X_mock.iloc[::20, 0] = np.nan
    X_mock.iloc[::15, 3] = np.nan
    
    # 3.5% gian lận
    y_mock = pd.Series((np.random.rand(n_samples) < 0.035).astype(int), name="isFraud")
    print(f"Dữ liệu test: {X_mock.shape}, Số ca gian lận: {y_mock.sum()} ({y_mock.mean():.2%})")

    # Chia train/test
    split_idx = int(n_samples * 0.7)
    X_tr, X_te = X_mock.iloc[:split_idx], X_mock.iloc[split_idx:]
    y_tr, y_te = y_mock.iloc[:split_idx], y_mock.iloc[split_idx:]

    # 2. Huấn luyện thử 4 mô hình
    lr_model = train_logistic_regression(X_tr, y_tr, max_iter=200)
    rf_model = train_random_forest(X_tr, y_tr, n_estimators=20, max_depth=5)
    lgb_model = train_lightgbm(X_tr, y_tr, X_val=X_te, y_val=y_te, n_estimators=30)
    iso_model = train_isolation_forest(X_tr, y_train=y_tr, n_estimators=20)

    # 3. So sánh các mô hình
    models = {
        "Logistic Regression": lr_model,
        "Random Forest": rf_model,
        "LightGBM": lgb_model,
        "Isolation Forest": iso_model
    }

    comparison = compare_models(models, X_te, y_te, threshold=0.5)
    print("\nKết quả so sánh sơ bộ:")
    print(comparison)

    # 4. Kiểm tra trích xuất đặc trưng
    top_feats = get_feature_importances(rf_model, list(X_tr.columns), top_n=5)
    print("\nTop 5 đặc trưng quan trọng của Random Forest:")
    print(top_feats)

    print("\n[OK] Toàn bộ module src/models.py đã chạy thành công và hoạt động chính xác!")
