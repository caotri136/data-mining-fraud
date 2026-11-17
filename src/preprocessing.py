"""
src/preprocessing.py
---------------------
Các hàm tiền xử lý dùng chung cho bộ dữ liệu IEEE-CIS Fraud Detection.

Nguyên tắc xuyên suốt file này: MỌI hàm có chữ "_fit" chỉ được gọi trên
train_df (để "học" tham số: median, danh sách category, tần suất...).
Hàm có chữ "_apply" dùng tham số đã học đó để áp dụng cho bất kỳ tập nào
(train/val/test). Tách fit/apply riêng là cách tránh rò rỉ dữ liệu
(data leakage) giữa các tập.

Cách dùng (trong notebook, sau khi đã copy load_data.py và gọi xong
load_data() + time_based_split()):

    import sys
    sys.path.append("../src")
    from preprocessing import (
        missing_group_split, compute_signal_table, apply_missing_rule,
        impute_fit, impute_apply, encode_fit, encode_apply,
        feature_engineering_fit, feature_engineering_apply,
    )

    # Bước 1-3: lọc cột theo mức độ thiếu + Cramér's V (làm trên toàn bộ df,
    # TRƯỚC khi chia tập — đây là bước phân tích/chọn cột, không phải "học
    # tham số" nên chấp nhận được làm trên toàn bộ dữ liệu)
    safe_cols, suspect_cols, drop_candidate_cols = missing_group_split(df)
    signal_df = compute_signal_table(df, suspect_cols + drop_candidate_cols)
    df_clean, meaningful_cols, noise_cols = apply_missing_rule(df, signal_df, safe_cols)

    # Bước 4: chia tập (dùng time_based_split từ load_data.py)
    train_df, val_df, test_df = time_based_split(df_clean)

    # Bước 5-7: impute / encode / feature engineering — CHỈ fit trên train
    numeric_cols = [...]        # xem ví dụ đầy đủ trong notebook 02_preprocessing
    categorical_cols = [...]

    impute_values = impute_fit(train_df, numeric_cols, categorical_cols)
    for split_df in [train_df, val_df, test_df]:
        impute_apply(split_df, impute_values, numeric_cols, categorical_cols)

    encoders = encode_fit(train_df, categorical_cols)
    for split_df in [train_df, val_df, test_df]:
        encode_apply(split_df, encoders, categorical_cols)

    card_freq = feature_engineering_fit(train_df)
    for split_df in [train_df, val_df, test_df]:
        feature_engineering_apply(split_df, card_freq)
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.preprocessing import LabelEncoder


# ===================== BƯỚC 1-2: Phân tích mức độ thiếu =====================

def missing_group_split(df: pd.DataFrame, safe_threshold: float = 50,
                         drop_threshold: float = 90):
    """
    Chia toàn bộ cột của df thành 3 nhóm theo % giá trị thiếu.

    Trả về:
        (safe_cols, suspect_cols, drop_candidate_cols)
        - safe_cols: thiếu < safe_threshold% -> giữ nguyên, impute bình thường
        - suspect_cols: thiếu trong khoảng [safe_threshold%, drop_threshold%)
          -> cần kiểm tra Cramér's V trước khi quyết định
        - drop_candidate_cols: thiếu >= drop_threshold% -> cần kiểm tra kỹ,
          nhiều khả năng sẽ bị bỏ
    """
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)

    safe_cols = missing_pct[missing_pct < safe_threshold].index.tolist()
    suspect_cols = missing_pct[
        (missing_pct >= safe_threshold) & (missing_pct < drop_threshold)
    ].index.tolist()
    drop_candidate_cols = missing_pct[missing_pct >= drop_threshold].index.tolist()

    print(f"[missing_group_split] an toàn: {len(safe_cols)}, "
          f"nghi ngờ: {len(suspect_cols)}, ứng viên bỏ: {len(drop_candidate_cols)}")

    return safe_cols, suspect_cols, drop_candidate_cols


def _cramers_v(contingency_table) -> tuple:
    """
    Tính Cramér's V (độ mạnh liên hệ, 0-1) và p-value từ 1 bảng chéo (2x2).

    Cramér's V không bị "ảo" bởi cỡ mẫu lớn như p-value đơn thuần, nên
    dùng chỉ số này để đánh giá mức độ liên quan THỰC SỰ, không chỉ là
    "có ý nghĩa thống kê" (statistically significant).
    """
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    v = np.sqrt(chi2 / (n * min_dim))
    return v, p


def compute_signal_table(df: pd.DataFrame, cols_to_check: list,
                          label_col: str = "isFraud",
                          effect_threshold: float = 0.10) -> pd.DataFrame:
    """
    Với mỗi cột trong cols_to_check, kiểm tra xem việc THIẾU/KHÔNG THIẾU
    giá trị ở cột đó có thực sự liên quan đến label_col (isFraud) hay không.

    Cách làm: lập bảng chéo 2x2 (thiếu hay không x isFraud 0/1), dùng
    Chi-square test + Cramér's V. Có áp dụng Bonferroni correction để bù
    lại việc kiểm định nhiều cột cùng lúc (giảm nguy cơ "dương tính giả").

    Trả về DataFrame với cột 'meaningful_effect' (True/False) — đây là
    tiêu chí chính để quyết định giữ/bỏ cột ở apply_missing_rule().
    """
    alpha_corrected = 0.05 / len(cols_to_check)
    missing_pct = df.isna().mean() * 100

    results = []
    for col in cols_to_check:
        is_missing = df[col].isna()
        contingency_table = pd.crosstab(is_missing, df[label_col])
        v, p_value = _cramers_v(contingency_table)

        results.append({
            "column": col,
            "missing_pct": missing_pct[col],
            "n_present": int((~is_missing).sum()),
            "p_value": p_value,
            "cramers_v": v,
            "significant_statistically": p_value < alpha_corrected,
            "meaningful_effect": v >= effect_threshold,
        })

    signal_df = pd.DataFrame(results).sort_values("cramers_v", ascending=False)
    print(f"[compute_signal_table] có tín hiệu thật: "
          f"{signal_df['meaningful_effect'].sum()}/{len(signal_df)} cột")
    return signal_df


def apply_missing_rule(df: pd.DataFrame, signal_df: pd.DataFrame,
                        safe_cols: list) -> tuple:
    """
    Áp dụng luật quyết định cuối cùng:
        - safe_cols: giữ nguyên.
        - Cột trong signal_df có meaningful_effect=True: giữ cột gốc +
          thêm cột cờ "<col>_is_missing" (0/1) để model khai thác riêng
          tín hiệu "thiếu/không thiếu".
        - Cột trong signal_df có meaningful_effect=False: bỏ hẳn.

    Trả về:
        (df_clean, meaningful_cols, noise_cols)
    """
    meaningful_cols = signal_df.loc[signal_df["meaningful_effect"], "column"].tolist()
    noise_cols = signal_df.loc[~signal_df["meaningful_effect"], "column"].tolist()

    df = df.copy()
    for col in meaningful_cols:
        df[f"{col}_is_missing"] = df[col].isna().astype(int)

    df_clean = df.drop(columns=noise_cols)

    print(f"[apply_missing_rule] giữ+cờ: {len(meaningful_cols)}, "
          f"bỏ hẳn: {len(noise_cols)}, shape cuối: {df_clean.shape}")

    return df_clean, meaningful_cols, noise_cols


# ===================== BƯỚC 5: Impute giá trị thiếu =====================

def impute_fit(train_df: pd.DataFrame, numeric_cols: list,
               categorical_cols: list) -> dict:
    """
    Học giá trị dùng để impute CHỈ TỪ train_df:
        - cột số: median
        - cột phân loại: chuỗi cố định "missing"

    QUAN TRỌNG: chỉ gọi hàm này với train_df, không bao giờ gọi với
    val_df/test_df hay toàn bộ df — nếu không sẽ gây rò rỉ dữ liệu.
    """
    impute_values = {}
    for col in numeric_cols:
        impute_values[col] = train_df[col].median()
    for col in categorical_cols:
        impute_values[col] = "missing"
    return impute_values


def impute_apply(df: pd.DataFrame, impute_values: dict, numeric_cols: list,
                  categorical_cols: list) -> pd.DataFrame:
    """
    Áp dụng bộ giá trị impute (đã học từ impute_fit) vào df (có thể là
    train_df, val_df hoặc test_df). Sửa trực tiếp trên df truyền vào.
    """
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(impute_values[col])
    for col in categorical_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(impute_values[col])
    return df


# ===================== BƯỚC 6: Mã hóa cột phân loại =====================

def encode_fit(train_df: pd.DataFrame, categorical_cols: list) -> dict:
    """
    Học danh sách giá trị duy nhất của từng cột phân loại CHỈ TỪ train_df,
    và thêm sẵn nhãn "unknown" để xử lý giá trị lạ xuất hiện ở val/test.

    Trả về dict {tên_cột: LabelEncoder đã fit}.
    """
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_values = train_df[col].astype(str)
        le.fit(train_values)
        # Thêm "unknown" vào danh sách lớp để encode_apply() dùng cho giá trị lạ
        le.classes_ = np.append(le.classes_, "unknown")
        encoders[col] = le
    return encoders


def encode_apply(df: pd.DataFrame, encoders: dict,
                  categorical_cols: list) -> pd.DataFrame:
    """
    Áp dụng encoder (đã học từ encode_fit) vào df. Giá trị nào ở df mà
    encoder chưa từng thấy ở train sẽ được gán thành "unknown" trước khi
    encode, tránh lỗi ValueError của LabelEncoder.
    """
    for col in categorical_cols:
        if col not in df.columns:
            continue
        le = encoders[col]
        values = df[col].astype(str)
        # classes_ không tính "unknown" (phần tử cuối) khi kiểm tra giá trị đã biết
        known_classes = set(le.classes_[:-1])
        values = values.where(values.isin(known_classes), "unknown")
        df[col] = le.transform(values)
    return df


# ===================== BƯỚC 7: Feature Engineering =====================

def feature_engineering_fit(train_df: pd.DataFrame) -> pd.Series:
    """
    Học tần suất xuất hiện của mỗi giá trị card1 CHỈ TỪ train_df.
    Trả về Series: {giá_trị_card1: số_lần_xuất_hiện_trong_train}.
    """
    return train_df["card1"].value_counts()


def feature_engineering_apply(df: pd.DataFrame, card_freq: pd.Series) -> pd.DataFrame:
    """
    Tạo 3 đặc trưng mới trên df:
        - hour_of_day: giờ trong ngày, suy ra từ TransactionDT (phép biến
          đổi thuần theo từng dòng, không "học" gì từ dữ liệu -> an toàn
          áp dụng trực tiếp cho mọi tập).
        - log_TransactionAmt: log(1 + số tiền), giúp giảm độ lệch phân bố.
        - card1_freq: tần suất của card1, lấy từ card_freq đã học ở train;
          thẻ nào val/test có mà train chưa từng thấy sẽ được gán 0.
    """
    df["hour_of_day"] = (df["TransactionDT"] / 3600) % 24
    df["log_TransactionAmt"] = np.log1p(df["TransactionAmt"])
    df["card1_freq"] = df["card1"].map(card_freq).fillna(0)
    return df


if __name__ == "__main__":
    # Chạy thử nhanh toàn bộ pipeline với sample nhỏ: python src/preprocessing.py
    import sys
    sys.path.append(".")
    from load_data import load_data, time_based_split

    df = load_data(path="../data/raw/", nrows=5000)

    safe_cols, suspect_cols, drop_candidate_cols = missing_group_split(df)
    signal_df = compute_signal_table(df, suspect_cols + drop_candidate_cols)
    df_clean, meaningful_cols, noise_cols = apply_missing_rule(df, signal_df, safe_cols)

    train_df, val_df, test_df = time_based_split(df_clean)

    numeric_cols = [c for c in train_df.select_dtypes(include=["float64", "int64"]).columns
                    if c != "isFraud"]
    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()

    impute_values = impute_fit(train_df, numeric_cols, categorical_cols)
    for split_df in [train_df, val_df, test_df]:
        impute_apply(split_df, impute_values, numeric_cols, categorical_cols)

    encoders = encode_fit(train_df, categorical_cols)
    for split_df in [train_df, val_df, test_df]:
        encode_apply(split_df, encoders, categorical_cols)

    card_freq = feature_engineering_fit(train_df)
    for split_df in [train_df, val_df, test_df]:
        feature_engineering_apply(split_df, card_freq)

    print("Test OK. Train shape:", train_df.shape)