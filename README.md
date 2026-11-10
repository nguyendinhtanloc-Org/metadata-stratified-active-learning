# Metadata-Stratified Active Learning for Object Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-green.svg)](https://github.com/ultralytics/ultralytics)
[![Dataset](https://img.shields.io/badge/Dataset-BDD100K-orange.svg)](https://bdd-data.berkeley.edu/)
[![Tests](https://img.shields.io/badge/tests-35%20passed-brightgreen.svg)]()

> **Tiểu luận chuyên ngành (Kỹ thuật Dữ liệu)**  
> **Sinh viên:** Nguyễn Đình Tấn Lộc (MSSV: 23133041)  
> **Giảng viên hướng dẫn:** TS. Nguyễn Thanh Tuấn  
> **Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Công nghệ Kỹ thuật TP.HCM (HCMUTE)

---

## 1. Giới thiệu đề tài

Trong các hệ thống thị giác máy tính cho xe tự hành (Autonomous Driving), chi phí gán nhãn bounding box rất tốn kém. **Active Learning (AL)** giúp giảm chi phí này bằng cách chọn lọc các ảnh mang lại lượng thông tin cao nhất để gán nhãn.

Tuy nhiên, các chiến lược chọn mẫu theo độ bất định (*Uncertainty Sampling*) truyền thống thường gặp hiện tượng **sụp đổ độ đa dạng (Batch Diversity Collapse)**: các mẫu có uncertainty cao thường tập trung vào cùng một điều kiện quen thuộc (ví dụ đường phố ban ngày), khiến mô hình bị học lệch (bias) và suy giảm hiệu năng nghiêm trọng tại các lát cắt nguy hiểm (*Adverse Weather*, *Night*, *Rare Scenes*).

Dự án này đề xuất giải pháp **Phân tầng theo Metadata vật lý (Metadata Stratification)** kết hợp:
1. **Lọc khó đa tầng (Hybrid Top-K Filtering):** Chỉ lọc trong tập ảnh có độ không chắc chắn cao nhất.
2. **Phân bổ Quota Căn bậc hai tần suất ($w_k \propto \sqrt{N_k}$):** Cân bằng Pareto tối ưu giữa tần suất tự nhiên và chia đều.
3. **Cân bằng biên đa chiều bằng thuật toán Iterative Proportional Fitting (IPF / Sinkhorn):** Giải bài toán ràng buộc biên đồng thời trên 3 trục metadata (`weather`, `scene`, `timeofday`) với sai số $L_1 \approx 0$.

---

## 2. Tóm tắt kết quả kiểm định thực nghiệm

Thực nghiệm được tiến hành trên tập **BDD100K**, so sánh giữa **Baseline** (Uncertainty-only) và **Stratified** (Đề xuất) qua **3 seeds độc lập (42, 123, 456)**, đánh giá trên tập kiểm thử cố định **2.000 ảnh (seed 2024)**:

| Chỉ số thực nghiệm | Baseline (Mean ± Std) | Stratified (Mean ± Std) | Kiểm định thống kê / Chênh lệch | Kết luận |
| :--- | :---: | :---: | :---: | :---: |
| **Entropy Batch Round 1** | $3.8366 \pm 0.0353$ bits | **$5.2031 \pm 0.0494$ bits** | $t = 36.26, p = 0.00076$, Cohen's $d = 20.93$ | **Bác bỏ $H_0$ ($p < 0.001$)** |
| **Entropy Batch Round 2** | $3.7010 \pm 0.0909$ bits | **$5.0640 \pm 0.0605$ bits** | $t = 16.82, p = 0.00352$, Cohen's $d = 9.71$ | **Bác bỏ $H_0$ ($p < 0.005$)** |
| **Domain `adverse_weather` $\text{mAP}_{50}$** | $0.3223 \pm 0.0055$ | **$0.3430 \pm 0.0147$** | **$+0.0208$ ($+2.08\%$)** | Vượt Baseline ở mọi seed |
| **Domain `adverse_weather` $\text{mAP}_{50\text{--}95}$** | $0.1753 \pm 0.0032$ | **$0.1910 \pm 0.0087$** | **$+0.0157$ ($+1.57\%$)** | Cải thiện độ siết bounding box |
| **Overall $\text{mAP}_{50}$ (Round 2)** | $0.3775 \pm 0.0009$ | $0.3655 \pm 0.0062$ | $-0.0120$ (Đánh đổi mật độ nhãn) | Trade-off an toàn miền hiếm |

---

## 3. Cấu trúc dự án (Repository Structure)

```
├── config/                 # Cấu hình tham số thực nghiệm (pipeline.yaml, database.yaml)
├── data/                   # Quản lý dữ liệu BDD100K (raw, interim, processed)
├── docker/                 # Môi trường container hóa (Dockerfile, docker-compose.yml)
├── kaggle/                 # Pipeline huấn luyện tự động trên Kaggle GPU T4
│   ├── 07_main_experiment.ipynb  # Notebook thực nghiệm chính (6 runs x 3 seeds)
│   └── requirements_kaggle.txt   # Dependencies môi trường Kaggle
├── results/                # Số liệu thực nghiệm tổng hợp
│   └── main_experiment/   # File combined_results.csv và domain_map50.csv
├── scripts/                # Script thực thi thực nghiệm tự động
│   ├── run_experiment.py   # Script thực thi dòng lệnh
│   ├── run_baseline.sh     # Script chạy baseline sampling
│   └── run_stratify.sh     # Script chạy stratified sampling
├── src/                    # Mã nguồn chính của đề tài
│   ├── active_learning/    # Vòng lặp Active Learning (ActiveLearningLoop)
│   ├── analysis/           # Thống kê Paired t-test, Cohen's d, Partial Correlation
│   ├── data/               # Trích xuất metadata và indexing nhãn BDD100K
│   ├── metrics/            # Tính toán Shannon Entropy theo metadata
│   ├── sampling/           # Chiến lược chọn mẫu:
│   │   ├── uncertainty.py  # Box-level uncertainty & class distribution entropy
│   │   ├── baseline.py     # Uncertainty-only baseline
│   │   └── stratified.py   # Water-filling allocation + IPF Sinkhorn rounding
│   └── training/           # Wrapper huấn luyện mô hình YOLOv8
└── test/                   # Kiểm thử tự động (Unit tests & Regression tests)
    ├── test_metrics.py     # Kiểm thử hàm tính toán Shannon entropy
    └── test_sampling.py    # Kiểm thử phân bổ quota, sai số L1 và độ phủ metadata
```

---

## 4. Cài đặt & Hướng dẫn sử dụng

### 4.1. Cài đặt môi trường

```bash
git clone https://github.com/dinhtanloc/metadata-stratified-active-learning.git
cd metadata-stratified-active-learning
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2. Chạy kiểm thử tự động (Unit Tests)

```bash
pytest test/ -v
```

### 4.3. Chạy thực nghiệm cục bộ (Local / Server)

```bash
# Chạy pipeline Baseline
bash scripts/run_baseline.sh

# Chạy pipeline Stratified
bash scripts/run_stratify.sh
```

---

## 5. Tài liệu nghiên cứu nền tảng (References)

1. **Elezi et al. (NVIDIA / TUM)** - *Not All Labels Are Equal: Rationalizing The Labeling Costs for Training Object Detection*.
2. **Kao et al. (MERL / UC Santa Barbara)** - *Localization-Aware Active Learning for Object Detection*.
3. **Citovsky et al. (Google Research)** - *Batch Active Learning at Scale (Cluster-Margin)*.
4. **Zhdanov (Amazon Research)** - *Diverse mini-batch Active Learning (DMBAL)*.
5. **Ren et al. (ACM Computing Surveys)** - *A Survey of Deep Active Learning*.
