"""
src/load_data.py
------------
hàm dùng chung để nạp và merge bộ dữ liệu IEEE-CIS Fraud Detection.

dùng chung file này để cả 3 thành viên không phải viết lại code load data
mỗi người một kiểu (tránh lỗi path, lỗi merge sai, v.v.)

cách dùng (trong notebook):

    import sys
    sys.path.append("../src")       # nếu notebook nằm trong thư mục notebooks/
    from load_data import load_data, time_based_split

    df = load_data(path="/kaggle/input/<ten-dataset-cua-ban>/", nrows=None)
    train_df, val_df, test_df = time_based_split(df)
"""

import os
import pandas as pd


def load_data(path: str, nrows: int = None) -> pd.DataFrame:
    """
    nạp và merge train_transaction.csv + train_identity.csv.

    tham số:
        path: đường dẫn thư mục chứa 2 file csv, PHẢI kết thúc bằng dấu "/"
              ví dụ Kaggle:  "/kaggle/input/<ten-dataset>/"
              ví dụ local:   "../data/raw/"
        nrows: số dòng muốn đọc thử (dùng khi debug cho nhanh).
               để None nếu muốn đọc full data.

    trả về:
        dataFrame đã merge theo transactionID (left join: giữ toàn bộ
        giao dịch, kể cả giao dịch không có thông tin identity).
    """
    transaction_path = os.path.join(path, "train_transaction.csv")
    identity_path = os.path.join(path, "train_identity.csv")

    if not os.path.exists(transaction_path):
        raise FileNotFoundError(
            f"không tìm thấy file: {transaction_path}\n"
            f"Kiểm tra lại biến `path`, dùng os.walk('/kaggle/input') "
            f"để xem đường dẫn thật nếu đang chạy trên Kaggle."
        )

    tr = pd.read_csv(transaction_path, nrows=nrows)
    idt = pd.read_csv(identity_path, nrows=nrows)

    # left join để có được toàn bộ giao dịch, kể cả giao dịch không có thông tin identity
    df = tr.merge(idt, on="TransactionID", how="left")

    # in ra số dòng, cột của từng file và sau khi merge, để kiểm tra nhanh
    print(f"[load_data] transaction: {tr.shape}, identity: {idt.shape}, "
          f"merged: {df.shape}")

    # in ra tỷ lệ gian lận (isFraud=1)
    if "isFraud" in df.columns:
        fraud_rate = df["isFraud"].mean()
        print(f"[load_data] Tỷ lệ gian lận (isFraud=1): {fraud_rate:.4%}")

    return df


def time_based_split(df: pd.DataFrame, train_frac: float = 0.70,
                      val_frac: float = 0.15):
    """
    Chia dữ liệu theo THỜI GIAN (dựa vào cột TransactionDT), KHÔNG chia
    ngẫu nhiên.

    Lý do: đây là dữ liệu giao dịch thực tế theo trình tự thời gian, nếu
    dùng train_test_split() ngẫu nhiên sẽ làm lộ thông tin tương lai vào
    tập train (data leakage), khiến kết quả đánh giá bị "ảo" (cao hơn
    thực tế).

    Tham số:
        df: DataFrame đã load từ load_data()
        train_frac: tỉ lệ dùng làm tập train (mặc định 70%)
        val_frac: tỉ lệ dùng làm tập validation (mặc định 15%)
                  phần còn lại (15%) tự động dùng làm tập test

    Trả về:
        (train_df, val_df, test_df)
    """

    # sắp xếp theo thời gian (TransactionDT) và reset index
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df_sorted)

    # tính chỉ số của các mốc chia train/val/test
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    # chia dữ liệu theo các chỉ số trên
    train_df = df_sorted.iloc[:train_end]
    val_df = df_sorted.iloc[train_end:val_end]
    test_df = df_sorted.iloc[val_end:]

    print(f"[time_based_split] train: {train_df.shape}, "
          f"val: {val_df.shape}, test: {test_df.shape}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Chạy thử nhanh: python src/load_data.py
    # (chỉnh lại đường dẫn bên dưới cho đúng máy bạn trước khi chạy)
    df = load_data(path="../data/raw/", nrows=5000)
    print(df.head())