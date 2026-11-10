# Kaggle Execution Guide

Thư mục này chứa toàn bộ các Jupyter Notebook dùng để thực thi các bước kiểm định dữ liệu, kiểm thử đơn vị, và chạy thực nghiệm chính trên nền tảng Kaggle GPU (Tesla T4).

## 1. Yêu cầu Môi trường (Environment Requirements)
* **Kaggle GPU**: Tesla T4 (16GB VRAM) hoặc tương đương
* **Internet**: Bật (ON) để clone repository và cài đặt thư viện
* Cài đặt thư viện:
  ```bash
  pip install -r kaggle/requirements_kaggle.txt
  ```

## 2. Thứ tự Thực thi Notebooks
1. **01_verify_dataset.ipynb**: Kiểm tra đường dẫn và tính sẵn sàng của tập dữ liệu BDD100K trên Kaggle working directory.
2. **02_verify_labels.ipynb**: Xác thực cấu trúc nhãn bounding box (box2d) và phân bố thuộc tính metadata.
3. **03_test_labels.ipynb**: Kiểm tra tính hợp lệ của việc chuyển đổi nhãn sang định dạng YOLO (`cls x y w h`).
4. **04_verify_image_paths.ipynb**: Quét chỉ mục ảnh và kiểm tra tốc độ nạp ảnh qua bộ đệm nhớ cache.
5. **05_quick_test.ipynb**: Chạy thử nghiệm mini-batch (1 vòng AL) để kiểm tra luồng huấn luyện trước khi chạy toàn bộ.
6. **06_unit_tests.ipynb**: Chạy toàn bộ 43/43 unit tests để kiểm tra mã nguồn trước khi huấn luyện.
7. **07_main_experiment.ipynb**: Notebook thực nghiệm chính — huấn luyện 3 seeds $\times$ 2 pipelines (Baseline vs Stratified) $\times$ 3 rounds, tự động tính kiểm định thống kê và đánh giá domain mAP.
