# Metadata-Stratified Active Learning

Đồ án tốt nghiệp TLCN - Cải thiện Active Learning bằng chiến lược sampling phân tầng dựa trên metadata.

## Project Structure

```
├── config/                 # Cấu hình pipeline, data, seeds
├── data/                   # Dataset (raw, interim, processed)
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docker/                 # Dockerfile & docker-compose
├── experiments/
│   └── runs/               # Kết quả chạy thực nghiệm theo từng run
├── kaggle/                 # Kaggle training kernel
├── mlruns/                 # MLflow tracking
├── notebook/               # Jupyter notebooks phân tích
├── report/                 # Báo cáo
├── results/
│   ├── figures/            # Biểu đồ
│   ├── tables/             # Bảng kết quả
│   └── summary.csv         # Tổng hợp kết quả
├── scripts/                # Shell & Python scripts chạy experiment
├── src/                    # Source code
│   ├── active_learning/    # Active learning loop
│   ├── analysis/           # Phân tích kết quả
│   ├── data/               # Data loading & preprocessing
│   ├── metrics/            # Evaluation metrics
│   ├── sampling/           # Sampling strategies
│   ├── tracking/           # MLflow integration
│   ├── training/           # Model training
│   └── utils/              # Utilities
└── test/                   # Unit tests
```

## Quick Start

```bash
pip install -r requirements.txt
```

## License

TLCN 2026
