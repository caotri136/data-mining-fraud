# BTL Data Mining — Phát hiện gian lận giao dịch thẻ tín dụng

Đề tài: Phát hiện gian lận trong giao dịch thẻ tín dụng, dùng bộ dữ liệu
[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle).

Repo này là nơi 3 thành viên cùng viết code. Môi trường chạy thật (vì
data nặng, máy không ai có GPU) là **Kaggle Notebook**; GitHub chỉ dùng
để lưu và review code, không dùng để chạy.

---

## 0. Việc đầu tiên mỗi người PHẢI làm (làm 1 lần)

1. Tạo tài khoản Kaggle (nếu chưa có) → vào trang competition → tab
   **Rules** → bấm **Late Submission** → Accept.
2. Clone repo này về máy:
   ```bash
   git clone <link-repo-github-cua-nhom>
   cd btl-fraud-starter
   ```
3. Tạo nhánh (branch) riêng cho mình, đặt tên theo phần việc:
   ```bash
   git checkout -b feature/eda          # Người A
   git checkout -b feature/modeling     # Người B
   git checkout -b feature/evaluation   # Người C
   ```
   **Không code trực tiếp trên nhánh `main`.**
4. Cài thư viện (nếu code/debug ở máy local với sample nhỏ):
   ```bash
   pip install -r requirements.txt
   ```

---

## 1. Cấu trúc thư mục

```
btl-fraud-starter/
├── README.md                     # File này
├── requirements.txt              # Thư viện cần cài
├── .gitignore                    # Loại trừ data, cache khỏi Git
│
├── data/
│   ├── raw/                      # Data gốc (KHÔNG commit — tự tải/add riêng)
│   └── processed/                # Data đã xử lý (KHÔNG commit)
│
├── notebooks/                    # Nơi code thử nghiệm, chạy trên Kaggle
│   ├── 01_eda.ipynb               # Người A
│   ├── 02_preprocessing.ipynb     # Người A
│   ├── 03_modeling.ipynb          # Người B
│   └── 04_evaluation.ipynb        # Người C
│
├── src/                           # Hàm dùng chung, đã "chốt" ổn định
│   ├── load_data.py               # ĐÃ CÓ SẴN — dùng hàm này để load data
│   ├── preprocessing.py           # Người A viết
│   ├── models.py                  # Người B viết
│   └── evaluate.py                # Người C viết
│
└── reports/
    ├── figures/                   # Hình xuất ra để chèn vào báo cáo
    └── BaiTapLon_NhomX.docx       # Báo cáo Word cuối cùng
```

---

## 2. Cách chạy — HƯỚNG DẪN TỪNG BƯỚC (copy-paste được)

### Bước A — Add data vào Kaggle Notebook

1. Vào trang competition trên Kaggle → tab **Code** → **New Notebook**.
2. Trong notebook, bấm **Add Input** (góc phải) → gõ "ieee-fraud-detection"
   → Add.
3. Chạy lệnh sau để tìm đúng đường dẫn data (đường dẫn Kaggle hay lồng
   thư mục khác nhau tùy cách add, nên luôn chạy lệnh này trước để
   chắc chắn):
   ```python
   import os
   for root, dirs, files in os.walk("/kaggle/input"):
       for f in files:
           print(os.path.join(root, f))
   ```
   Copy đúng phần đường dẫn thư mục (không copy tên file) để dùng ở
   bước B.

### Bước B — Copy file `load_data.py` vào notebook

Vì Kaggle Notebook không tự đọc được file `src/load_data.py` từ GitHub,
mỗi người cần copy nội dung file đó vào 1 cell đầu tiên của notebook
mình đang làm (copy nguyên văn, không sửa):

```python
# ==== Dán nguyên nội dung src/load_data.py vào đây ====
# (mở file src/load_data.py trong repo, copy toàn bộ, paste vào cell này)
```

Sau đó ở cell tiếp theo, gọi hàm:

```python
df = load_data(path="/kaggle/input/<đường-dẫn-bạn-copy-ở-bước-A>/", nrows=5000)
# Khi code chạy ổn với nrows=5000, đổi nrows=None để chạy full data
```

> Lý do copy-paste thay vì import trực tiếp: Kaggle Notebook không tự
> động thấy được repo GitHub của bạn trừ khi bạn add GitHub làm Dataset
> riêng (phức tạp hơn cần thiết cho quy mô bài tập này). Copy-paste vài
> chục dòng code là đủ nhanh và không đáng để setup phức tạp hơn.

### Bước C — Code phần việc của mình

Làm theo đúng file notebook được phân công (xem mục Phân công bên
dưới). Luôn luôn:

- Debug trước với `nrows=5000` cho nhanh.
- Khi chạy ổn, đổi sang full data (`nrows=None`) để lấy kết quả thật
  cho báo cáo.
- Ghi chú (markdown cell) giải thích TẠI SAO làm bước đó, không chỉ
  code — vì báo cáo cần giải thích, không chỉ cần kết quả.

### Bước D — Đưa hàm đã ổn định vào `src/`, rồi push lên GitHub

Khi 1 đoạn code đã chạy đúng và ổn định (ví dụ hàm xử lý missing value,
hàm train model...), **tách nó thành hàm** và đặt vào file tương ứng
trong `src/` (không để code rời rạc chỉ nằm trong notebook — để 2 bạn
kia dùng lại được).

```bash
git add .
git commit -m "Thêm hàm xử lý missing value cho các cột V1-V339"
git push origin feature/eda      # đổi tên nhánh theo người đang push
```

Sau đó lên GitHub, tạo **Pull Request** từ nhánh của mình vào `main`.
Nhóm trưởng review và bấm Merge.

### Bước E — Khi bắt đầu công việc dựa trên phần người khác đã làm

Trước khi code tiếp (ví dụ: Người B cần dùng data đã được Người A làm
sạch), luôn kéo code mới nhất về trước:

```bash
git checkout main
git pull origin main
git checkout feature/modeling
git merge main       # cập nhật code mới của người khác vào nhánh mình
```

---

## 3. Phân công theo 5 mục chấm điểm của đề

| Người                       | Phụ trách                                                                                                                                                 | File                                                             |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **A**                       | Mục 1 (Mô tả bài toán, 2đ) + Mục 2 (Tiền xử lý, 3đ)                                                                                                       | `01_eda.ipynb`, `02_preprocessing.ipynb`, `src/preprocessing.py` |
| **B**                       | Mục 3 (Áp dụng thuật toán, 5đ): Classification (Logistic Regression, Random Forest, LightGBM) + 1 kỹ thuật không giám sát (Isolation Forest hoặc K-Means) | `03_modeling.ipynb`, `src/models.py`                             |
| **C**                       | Mục 4 (Phân tích/thảo luận, 3đ): so sánh kết quả, SHAP, chọn threshold + Mục 5 (Ràng buộc bổ sung, 2đ) + tổng hợp báo cáo Word                            | `04_evaluation.ipynb`, `src/evaluate.py`, `reports/`             |
| **Nhóm trưởng** (caotri136) | Review Pull Request, merge code, giữ deadline, soát lại báo cáo cuối                                                                                      | —                                                                |

**Input/output giữa các phần (để biết mình cần gì từ ai):**

- A → B: `02_preprocessing.ipynb` phải xuất ra data đã làm sạch (lưu
  thành `data/processed/train_clean.csv`, báo cho B biết tên cột cuối
  cùng dùng được).
- B → C: `03_modeling.ipynb` phải lưu lại dự đoán của từng model (ví dụ
  `y_pred_proba` của từng model) để C dùng vẽ PR curve so sánh.

---

## 4. Deadline đề xuất

Hôm nay: **Thứ Năm 12/11**. Lịch dưới đây dành ~3 tuần cho phần Data
Mining, chừa thời gian buffer trước khi chuyển sang phần xây dựng
website.

| Deadline           | Việc cần xong                                                                                                      | Ai chịu trách nhiệm                       |
| ------------------ | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- |
| **T7 14/11**       | Mỗi người tự setup xong môi trường (clone repo, add data Kaggle, chạy thử `load_data()` với sample nhỏ thành công) | Cả 3 người                                |
| **Thứ 4 18/11**    | A xong EDA + tiền xử lý, push `data/processed/train_clean.csv` kèm code lên nhánh, tạo PR                          | A                                         |
| **CN 22/11**       | B xong training + so sánh các model, có bảng kết quả Precision/Recall/F1/PR-AUC                                    | B (bắt đầu từ 19/11 khi có data sạch)     |
| **Thứ 4 25/11**    | C xong SHAP + phân tích + ràng buộc bổ sung, có bản nháp báo cáo Word                                              | C (bắt đầu từ 22/11 khi có kết quả model) |
| **Thứ 7 28/11**    | Nhóm trưởng review toàn bộ, merge hết PR vào `main`, hoàn thiện báo cáo Word, chốt bản cuối                        | Nhóm trưởng + cả 3                        |
| **CN 29/11 → hết** | Buffer dự phòng (sửa lỗi phát sinh, làm slide thuyết trình) + bắt đầu chuyển sang phần xây dựng website            | Cả 3                                      |

**Lưu ý quan trọng:** B và C không thể bắt đầu thật sự tới khi A xong
(B cần data sạch, C cần kết quả model). Để không bị động, B và C nên
**viết khung code trước với data mẫu 5000 dòng tự giả định cấu trúc
cột** (xem trước `02_preprocessing.ipynb` dự kiến xuất cột gì), chứ
không ngồi chờ không làm gì đến 19/11.

Nếu nhóm thấy mốc nào trễ, báo sớm trong nhóm chat để dồn thời gian
buffer qua, không để dồn vào sát ngày nộp.

---

## 5. Checklist trước khi nộp

- [ ] Code trong `src/` chạy được không lỗi từ đầu đến cuối (test lại
      bằng cách chạy full pipeline 1 lần)
- [ ] File Word báo cáo đủ 5 mục đúng theo khung điểm của đề
- [ ] Có ít nhất 2 kỹ thuật Data Mining (đề yêu cầu tối thiểu 2)
- [ ] Dùng đúng metric (Precision/Recall/F1/PR-AUC), không chỉ dùng
      Accuracy (vì data mất cân bằng lớp nặng)
- [ ] Nêu rõ ràng ít nhất 3 ràng buộc bổ sung (mục 5)
- [ ] Nén đúng định dạng `.zip` theo yêu cầu nộp bài, gồm: danh sách
      thành viên, file Word, mã nguồn, dataset (nếu đề yêu cầu kèm)
