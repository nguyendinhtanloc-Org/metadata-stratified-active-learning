"""
Active Learning Loop.
Kết hợp tất cả modules để chạy AL experiment.
"""

from typing import List, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime
import random
import pickle
import math
import gc
import os
import shutil
import warnings
import numpy as np
import torch  # Thêm cho memory management
try:
    import psutil
except ImportError:
    psutil = None

from sampling.uncertainty import calc_image_uncertainty
from sampling.stratified import stratified_sampling, calculate_batch_diversity
from sampling.baseline import baseline_sampling
from training.trainer import YOLOTrainer
from data.loader import (
    get_all_metadata,
    get_all_metadata_cached,
    build_image_path_map,
    build_image_path_map_cached,
    filter_images_with_labels,
    YOLO_CLASS_NAMES
)
from metrics.entropy import calc_batch_entropy


# ---------------------------------------------------------------------------
# Glibc heap tuning (QUAN TRỌNG cho Kaggle: OOM Killer đọc cgroup 32.2GB)
# ---------------------------------------------------------------------------
# glibc chỉ đọc biến môi trường MALLOC_* LÚC malloc init (trước main()).
# Set os.environ["MALLOC_TRIM_THRESHOLD_"] trong cell đã chạy -> VÔ HIỆU.
# Fix duy nhất hiệu lực ngay: mallopt(M_TRIM_THRESHOLD=-1, 65536) qua ctypes
# -> glibc tự động trim heap về OS mỗi lần free, không cần gọi malloc_trim tay.
_LIBC = None
try:
    import ctypes as _ctypes
    _LIBC = _ctypes.CDLL("libc.so.6")
    _LIBC.mallopt(-1, 65536)  # M_TRIM_THRESHOLD = -1 (glibc constant)
    _LIBC.malloc_trim(0)
    print("[MALLOC] mallopt(M_TRIM_THRESHOLD=65536) + malloc_trim(0) applied", flush=True)
except Exception as _e:
    _LIBC = None
    print(f"[MALLOC] libc tuning unavailable: {_e}", flush=True)


def _read_cgroup_memory():
    """Đọc RAM đang dùng & giới hạn của cgroup (cái OOM Killer dùng)."""
    cur = mx = None
    for c, m in [("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
                  ("/sys/fs/cgroup/memory/memory.usage_in_bytes",
                   "/sys/fs/cgroup/memory/memory.limit_in_bytes")]:
        try:
            cur = int(open(c).read().strip()) / 1e9
            raw = open(m).read().strip()
            mx = int(raw) / 1e9 if raw.isdigit() else None
            break
        except Exception:
            continue
    return cur, mx


class ActiveLearningLoop:
    """
    Main Active Learning loop.
    """

    def __init__(self, config: dict, experiment_name: str = "experiment"):
        """
        Khởi tạo AL loop.
        
        Args:
            config: Dictionary chứa các tham số:
                - seed: Random seed
                - initial_budget: Số lượng mẫu ban đầu
                - al_batch_size: Số lượng mẫu thêm mỗi round
                - train_batch_size: Batch size cho YOLO training
                - num_rounds: Số vòng AL
                - epochs: Số epochs cho mỗi lần train
                - device: Thiết bị (0 cho GPU, 'cpu' cho CPU)
                - num_classes: Số lượng classes
                - stratify_fields: Các trường metadata để stratify
                - model_name: Tên YOLO model
        """
        
        self.experiment_name = experiment_name

        # Random seed - set cho cả 3 thư viện để reproducibility
        seed = config.get("seed", 42)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.initial_budget = config.get("initial_budget", 500)
        self.batch_size = config.get("al_batch_size", 500)
        self.train_batch_size = config.get("train_batch_size", 16)
        self.num_rounds = config.get("num_rounds", 5)
        self.epochs = config.get("epochs", 10)
        self.device = config.get("device", 0)
        self.num_classes = config.get("num_classes", 10)
        self.stratify_fields = config.get("stratify_fields", ["weather", "scene", "timeofday"])
        self.model_name = config.get("model_name", "yolov8n")
        self.use_baseline = config.get("use_baseline", False)

        # Inference settings (tối ưu tốc độ cho AL loop)
        # predict_batch_size: số ảnh xử lý mỗi lần gọi model
        #   - 256 nhanh, an toàn về VRAM (Kaggle T4 16GB)
        #   - 512 nhanh hơn nữa nếu đủ VRAM
        # predict_imgsz: kích thước ảnh khi tính uncertainty
        #   - 640 giống training -> nhất quán nhất (khuyến nghị)
        #   - 320 nhanh gấp 2-3x, entropy chính xác hơn một chút
        self.predict_batch_size = config.get("predict_batch_size", 256)
        # predict_workers: số dataloader workers của ultralytics khi predict.
        # DEFAULT ultralytics = 8 worker. Mỗi worker prefetch (prefetch_factor
        # batches) ảnh decode full-res (~11MB/ảnh x 256 x 2) -> với 8 worker
        # queue in-flight ~39GB, VUOT RAM T4 x1 (34GB) => OOM "Kernel died".
        # 2 worker -> ~16GB, an toàn; tốc độ giảm nhẹ (I/O-bound, không
        # ảnh hưởng GPU inference).
        self.predict_workers = config.get("predict_workers", 2)
        self.predict_imgsz = config.get("predict_imgsz", 640)

        # local_image_cache: cache ảnh vào local disk (SSD) để inference nhanh hơn
        #   - None = tắt (đọc trực tiếp từ /kaggle/input)
        #   - "data/local_images" = copy 1 lần rồi đọc từ local disk
        # Lý do: đọc ảnh từ /kaggle/input là network-backed storage, chậm hơn
        # nhiều lần so với local SSD (đo thực tế trên Kaggle T4: GPU inference
        # chỉ 2ms/ảnh nhưng tổng thời gian là 160ms/ảnh -> 98% là I/O).
        self.local_image_cache = config.get("local_image_cache", None)

        # Fixed held-out eval set
        # eval_size: số ảnh dành cho đánh giá, bị loại khỏi pool labeled vĨNH VIỄN
        #   - 0 = tắt (giữ hành vi cũ: val rút ngẫu nhiên từ labeled pool)
        #   - 1000-2000 = đủ lớn để mAP ổn định, vẫn giữ ~60k ảnh cho AL
        # eval_split_seed: seed CỐ ĐỊNH, tách biệt khỏi seed của experiment
        #   -> eval set giống hệt nhau cho mọi pipeline và seed
        self.eval_size = config.get("eval_size", 2000)
        self.eval_split_seed = config.get("eval_split_seed", 2024)
        # Thư mục chứa eval set dùng chung cho mọi round & experiment
        # (copy 1 lần duy nhất, tiết kiệm ~3.6GB disk + thời gian I/O)
        self.shared_eval_dir = config.get("shared_eval_dir", "data/processed/_shared_eval")

        # Early stopping: số epoch không cải thiện thì dừng. 0 = tắt.
        # Với dataset nhỏ (500-1500 ảnh) và chỉ 5 epochs, patience=3 là an toàn.
        self.early_stop_patience = config.get("early_stop_patience", 0)
        
        # Models và data
        self.trainer = YOLOTrainer(
            model_name=self.model_name,
            device=self.device,
            project_dir=f"experiments/runs/{experiment_name}",
            train_seed=seed,
        )
        
        self.all_metadata = []
        self.filtered_metadata = []
        self.image_path_map = {}
        
        # Labeled/Unlabeled tracking
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        self.valid_labeled_indices = set()
        self.eval_indices = []
        
        # Results tracking
        self.results_history = []
        
    def _cache_images_locally(self, metadata_list: List[dict]) -> int:
        """
        Copy ảnh từ /kaggle/input sang local disk (/kaggle/working) một lần.

        Sau khi copy, các lần predict sau (và các experiment sau trong cùng
        session) đọc từ local disk -> nhanh hơn nhiều lần.

        Returns:
            Số ảnh đã copy.
        """
        cache_root = Path(self.local_image_cache)
        cache_root.mkdir(parents=True, exist_ok=True)

        copied = 0
        skipped = 0
        for meta in metadata_list:
            src = meta.get("image_path")
            if not src:
                continue
            dst = cache_root / os.path.basename(src)
            if dst.exists():
                meta["image_path"] = str(dst)
                skipped += 1
                continue
            try:
                shutil.copy(src, dst)
                meta["image_path"] = str(dst)
                copied += 1
            except Exception:
                pass  # giữ nguyên path gốc nếu copy lỗi

        print(f"[CACHE] Local image cache: {copied} copied, {skipped} already present")
        print(f"[CACHE] Location: {cache_root}")
        return copied

    def initialize(self, metadata_path: str, image_dirs: List[str]):
        """
        Khởi tạo dataset và chọn initial samples.
        
        Args:
            metadata_path: Đường dẫn đến metadata JSON
            image_dirs: Danh sách thư mục chứa ảnh
        """
        
        print(f"[INIT] Loading metadata from {metadata_path}...")
        
        # Load metadata
        # Cache metadata + image index giữa các experiments (tiết kiệm ~3.5 phút/run)
        cache_dir = "data/cache"
        metadata_cache = f"{cache_dir}/metadata_{Path(metadata_path).stem}.pkl"
        path_cache = f"{cache_dir}/image_path_map.pkl"

        self.all_metadata = get_all_metadata_cached(metadata_path, metadata_cache)

        # Build image path map
        print("[INIT] Building image path index...")
        self.image_path_map = build_image_path_map_cached(image_dirs, path_cache)
        
        # Add image paths to metadata
        for meta in self.all_metadata:
            filename = meta.get("filename")
            if filename in self.image_path_map:
                meta["image_path"] = self.image_path_map[filename]
            else:
                meta["image_path"] = None
        
        # Filter images with labels
        self.filtered_metadata = filter_images_with_labels(self.all_metadata, self.image_path_map)
        valid_metadata = self.filtered_metadata

        # Cache ảnh vào local disk nếu được cấu hình (tăng tốc inference)
        if self.local_image_cache:
            self._cache_images_locally(valid_metadata)

        print(f"[INIT] Found {len(self.all_metadata)} images")
        print(f"[INIT] Valid samples: {len(valid_metadata)}/{len(self.all_metadata)}")

        # Initialize all as unlabeled
        self.unlabeled_indices = set(range(len(valid_metadata)))

        # Tách EVAL SET cố định ra khỏi pool (không bao giờ được label/train).
        # Dùng chung cho mọi experiment & seed => mAP comparable, không rò rỉ.
        self.eval_indices = self._build_eval_set(valid_metadata)
        eval_set = set(self.eval_indices)
        self.unlabeled_indices -= eval_set

        # Chọn initial samples bằng stratified sampling
        print(f"[INIT] Total images: {len(valid_metadata)}")
        print(f"[INIT] Eval set (held-out): {len(self.eval_indices)}")
        print(f"[INIT] Initial budget: {self.initial_budget}")

        # Chỉ chọn trong pool đã loại eval set
        pool_indices = sorted(self.unlabeled_indices)
        pool_metadata = [self.filtered_metadata[i] for i in pool_indices]

        initial_local = stratified_sampling(
            metadata_list=pool_metadata,
            uncertainty_scores={i: random.random() for i in range(len(pool_metadata))},
            batch_size=self.initial_budget,
            stratify_fields=self.stratify_fields
        )
        # Map local index -> global index
        initial_indices = [pool_indices[i] for i in initial_local]

        self.labeled_indices = set(initial_indices)
        self.unlabeled_indices -= self.labeled_indices

        # Track valid labeled samples
        for idx in initial_indices:
            meta = valid_metadata[idx]
            image_path = meta.get("image_path")
            if image_path and Path(image_path).exists():
                self.valid_labeled_indices.add(idx)

        print(f"[INIT] Selected {len(self.labeled_indices)} initial samples")

        # Calculate initial diversity
        initial_metadata = [valid_metadata[i] for i in initial_indices]
        diversity = calculate_batch_diversity(
            self.filtered_metadata,
            initial_indices,
            self.stratify_fields
        )

        print(f"[INIT] Initial batch diversity:")
        for field, entropy in diversity.items():
            print(f"       - {field}: {entropy:.4f} bits")

    def _build_eval_set(self, valid_metadata: List[Dict]) -> List[int]:
        """
        Tạo fixed held-out eval set, CỐ ĐỊNH cho mọi experiment/seed.

        Dùng seed cố định (2024 - tách biệt khỏi seed của experiment) để đảm bảo
        eval set luôn giống nhau -> so sánh mAP giữa các pipeline/seed là hợp lệ.

        Ưu điểm so với val set rút ngẫu nhiên từ labeled pool mỗi round:
        - mAP ở mọi round đo trên CÙNG tập ảnh -> comparable
        - không rò rỉ (eval set không bao giờ vào train/labeled pool)
        - val set nội bộ vẫn giữ 20% để early stopping vẫn hoạt động
        """
        eval_size = self.eval_size
        if eval_size <= 0:
            print("[INIT] Eval set disabled")
            return []

        # Sort theo filename để thứ tự deterministic
        ordered = sorted(
            range(len(valid_metadata)),
            key=lambda i: str(valid_metadata[i].get("filename", ""))
        )

        rng = random.Random(self.eval_split_seed)
        chosen = rng.sample(ordered, min(eval_size, len(ordered)))
        return sorted(chosen)
    
    def train_round(self, round_num: int) -> Dict:
        """
        Train model trên current labeled set.
        
        Args:
            round_num: Round number
        
        Returns:
            Dict chứa training metrics
        """
        
        print(f"\n[ROUND {round_num}] Training model...")
        print(f"[ROUND {round_num}] Labeled samples: {len(self.valid_labeled_indices)}")
        
        if len(self.valid_labeled_indices) == 0:
            raise ValueError("Không có ảnh nào để train!")
        
        # Prepare training data
        labeled_list = list(self.valid_labeled_indices)
        labeled_metadata = [self.filtered_metadata[i] for i in labeled_list]
        image_paths = [self.filtered_metadata[i].get("image_path") for i in labeled_list]

        # Filter out any images with None paths
        valid_pairs = [(meta, path) for meta, path in zip(labeled_metadata, image_paths) if path is not None]

        if len(valid_pairs) == 0:
            raise ValueError("Không có ảnh hợp lệ để train!")

        labeled_metadata, image_paths = zip(*valid_pairs)
        labeled_metadata = list(labeled_metadata)
        image_paths = list(image_paths)

        # Eval set cố định (nếu có) - dùng để báo cáo mAP comparable giữa các round
        eval_paths, eval_metadata = None, None
        if self.eval_indices:
            eval_metadata = [self.filtered_metadata[i] for i in self.eval_indices]
            eval_paths = [m.get("image_path") for m in eval_metadata]

        dataset_yaml = self.trainer.prepare_dataset_with_metadata(
            image_paths=image_paths,
            metadata_list=labeled_metadata,
            output_dir=f"data/processed/{self.experiment_name}/round_{round_num}",
            split_seed=42,
            eval_paths=eval_paths,
            eval_metadata=eval_metadata,
            shared_eval_dir=self.shared_eval_dir,
        )
        
        save_name = f"round_{round_num}"
        
        # train
        print(f"[ROUND {round_num}] Starting training...")
        train_results = self.trainer.train(
            dataset_yaml=dataset_yaml,
            epochs=self.epochs,
            batch_size = self.train_batch_size,
            imgsz=640,
            save_name=save_name,
            patience=self.early_stop_patience,
        )
        
        print(f"[ROUND {round_num}] Training complete!")
        print(f"[ROUND {round_num}] mAP50: {train_results['map50']:.4f}")

        # Clear CUDA cache sau khi train để tránh memory leak
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[MEMORY] GPU cache cleared")

        return train_results
    
    @staticmethod
    def _trim_memory():
        """Ép GC + CUDA + glibc trả RAM thật sự về OS (chống RSS phình).

        Kaggle OOM Killer đọc CGROUP memory (32.2GB) chứ không phải RSS
        'available' của psutil. glibc malloc giữ lại heap cache đã giải
        phóng -> RSS không về xuống -> cgroup đếm lố -> killed.
        malloc_trim(0) ép glibc trả lại toàn bộ.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if _LIBC is not None:
            try:
                _LIBC.malloc_trim(0)
            except Exception:
                pass

    def _predict_uncertainty(
        self,
        image_paths: List[str],
        indices: List[int],
        batch_size: int = None,
    ) -> Dict[int, float]:
        """
        Dự đoán uncertainty cho toàn bộ ảnh chưa label, tối ưu tốc độ.

        Tối ưu so với phiên bản cũ:
        - batch lớn hơn nhiều (mặc định 256 thay vì 32) -> giảm số lần gọi model
        - `stream=True` -> không giữ toàn bộ Results objects trong RAM
        - `half=True` (FP16) -> inference nhanh gấp đôi trên GPU
        - gc.collect() chỉ gọi mỗi ~50 batch thay vì mỗi batch
        - `rect=True` -> ít padding, nhanh hơn

        Args:
            image_paths: Danh sách đường dẫn ảnh
            indices: Index tương ứng trong metadata
            batch_size: Batch size cho inference (None = dùng config)
        """
        scores: Dict[int, float] = {}
        max_entropy = math.log2(self.num_classes)

        if batch_size is None:
            batch_size = self.predict_batch_size
        imgsz = self.predict_imgsz

        device_arg = self.device

        # CHUYEN PHUONG: doc DAY DU danh sach source trong 1 goi predict (stream).
        # Ly do: 3 lan chay 28-29/10 voi phien ban "slice 4096" deu bi
        # "Kernel died" (6.4p / 9.3p / 53s sau predict bat dau) - moi slice
        # la 1 lan goi predict() moi -> dataloader + CUDA context tao lai
        # tung slice -> bo nho tang dan den OOM. Phien ban list-don da
        # CHUNG MINH so 3.8h (log 21/10, cung T4 32GB + ultralytics 8.4.184).
        # Do do: 1 lan goi predict() duy nhat, stream=True giu RAM that thap,
        # va log RAM moi 5000 anh de giam sat.
        # half=True: ultralytics >=8.4 warn "'half' is deprecated" -> giu flag
        # (van hoat dong) va chan warning thay vi doi sang 'quantize'.
        # CHIEN LUOC RAM co GIOI HAN (Kaggle cgroup = 32.2GB):
        # - predict theo chunk 4096 anh -> dataloader chi dat hang 4096 duong dan
        # - GIẢI PHÓNG result.orig_img / results ngay khi tieu thu (mỗi Results
        #   giữ orig_img full-res ~2.8MB -> 59k Results = ~165GB, là thủ phạm chính)
        # - _trim_memory() (gc + cuda.empty_cache + malloc_trim) cuỗi mỗi chunk
        #   -> RSS thật sự hạ xuống, không phình dần cho đến khi OOM
        # - workers=0 (đọc ảnh ở main process, tránh CoW leak của multiprocessing)
        # - batch=128 giữ tốc độ GPU gần như không đổi (I/O-bound)
        # CHUNK 1024 (giảm từ 4096): giảm đỉnh RAM giữa 2 lần trim -> an toàn
        # hơn với cgroup 32.2GB khi heap phồng theo thời gian (I/O-bound nên
        # tốc độ gần như không đổi).
        PREDICT_CHUNK = 1024
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*half.*deprecated.*")
            processed = 0
            for chunk_start in range(0, len(image_paths), PREDICT_CHUNK):
                ck_paths = image_paths[chunk_start:chunk_start + PREDICT_CHUNK]
                ck_idx = indices[chunk_start:chunk_start + PREDICT_CHUNK]
                try:
                    results_iter = self.trainer.model.predict(
                        source=ck_paths,
                        batch=batch_size,
                        imgsz=imgsz,
                        stream=True,
                        verbose=False,
                        half=True,
                        device=device_arg,
                        rect=True,
                        conf=0.001,
                        iou=0.7,
                        max_det=300,
                        workers=0,      # KHÔNG dataloader workers -> tranh CoW leak
                    )
                    for k, result in enumerate(results_iter):
                        idx = ck_idx[k] if k < len(ck_idx) else None
                        if idx is not None:
                            if result is None:
                                scores[idx] = max_entropy
                            else:
                                scores[idx] = calc_image_uncertainty(result, self.num_classes)
                                # GIẢI PHÓNG ảnh gốc GIỮNG SÁT (tỗi mỗi ảnh, không đợi cuỗi chunk)
                                for attr in ("orig_img", "names", "path"):
                                    try:
                                        setattr(result, attr, None)
                                    except Exception:
                                        pass
                        del result
                    del results_iter
                    processed += len(ck_idx)
                except Exception as e:
                    print(f"[WARNING] Chunk {chunk_start}-{chunk_start+len(ck_idx)} predict "
                          f"failed ({e}) -> dùng entropy cao nhất", flush=True)
                    for idx in ck_idx:
                        scores[idx] = max_entropy
                    processed += len(ck_idx)

                # ÉP RAM hoàn trả về OS cuỗi mỗi chunk (quan trọng nhất)
                self._trim_memory()

                # LOG MEM: RSS + cgroup usage/LIMIT (cái OOM Killer dùng)
                # Log mỗi chunk -> nếu cgroup usage KHÔNG giảm giữa 2 log
                # (trim không hiệu quả) hoặc tăng dần (leak), thấy ngay trong
                # log thay vì chỉ biết khi "Kernel died".
                mem_info = ""
                try:
                    rss_gb = psutil.Process().memory_info().rss / 1e9
                    cg_cur, cg_lim = _read_cgroup_memory()
                    mem_info = f" [MEM] rss={rss_gb:.1f}GB"
                    if cg_cur is not None:
                        mem_info += f" cgroup={cg_cur:.1f}GB"
                        if cg_lim is not None:
                            mem_info += f"/limit={cg_lim:.1f}GB"
                except Exception:
                    pass
                print(f"      Processed {processed}/{len(image_paths)} images{mem_info}", flush=True)

        return scores

    def _predict_uncertainty_chunked(
        self,
        image_paths: List[str],
        indices: List[int],
        chunk_size: int = 512,
    ) -> Dict[int, float]:
        """Fallback: predict theo chunk lớn, không dùng stream."""
        scores: Dict[int, float] = {}
        max_entropy = math.log2(self.num_classes)
        imgsz = self.predict_imgsz

        for start in range(0, len(image_paths), chunk_size):
            chunk_paths = image_paths[start:start + chunk_size]
            chunk_indices = indices[start:start + chunk_size]

            try:
                batch_results = self.trainer.model.predict(
                    source=chunk_paths,
                    batch=len(chunk_paths),
                    imgsz=imgsz,
                    verbose=False,
                    half=True,
                    device=self.device,
                    rect=True,
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    workers=0,
                )
            except Exception as e:
                print(f"[WARNING] Chunk {start} failed: {e}")
                for idx in chunk_indices:
                    scores[idx] = max_entropy
                continue

            if not isinstance(batch_results, list):
                batch_results = [batch_results]

            for idx, result in zip(chunk_indices, batch_results):
                if result is None:
                    scores[idx] = max_entropy
                else:
                    scores[idx] = calc_image_uncertainty(result, self.num_classes)

            del batch_results
            if (start // chunk_size) % 4 == 0:
                self._trim_memory()
            if (start // chunk_size) % 10 == 0:
                gc.collect()
                print(f"      Processed {min(start + chunk_size, len(image_paths))}/{len(image_paths)} images", flush=True)

        return scores

    def select_batch(self, round_num: int) -> List[int]:
        """
        Chọn batch mới để label bằng stratified sampling.
        
        Args:
            round_num: Số thứ tự round
        
        Returns:
            List các indices được chọn
        """

        print(f"\n[ROUND {round_num}] Selecting next batch...")
        print(f"[ROUND {round_num}] Unlabeled samples: {len(self.unlabeled_indices)}")

        # chuyển sang list để index
        unlabeled_list = list(self.unlabeled_indices)

        # tính uncertainty cho tất cả các unlabeled images
        print(f"[ROUND {round_num}] Computing uncertainty scores...")

        uncertainty_scores = {}

        # Lấy danh sách đường dẫn ảnh hợp lệ
        valid_image_paths = []
        valid_indices = []
        
        for idx in unlabeled_list:
            meta = self.filtered_metadata[idx]
            image_path = meta.get("image_path")
            if image_path:
                valid_image_paths.append(image_path)
                valid_indices.append(idx)
        
        print(f"[ROUND {round_num}] Predicting {len(valid_image_paths)} images...")

        try:
            uncertainty_scores.update(
                self._predict_uncertainty(valid_image_paths, valid_indices)
            )
        except Exception as e:
            print(f"[WARNING] Prediction failed: {e}")
            # Fallback: random uncertainty
            for idx in valid_indices:
                uncertainty_scores[idx] = random.random()

        # Các ảnh không tồn tại có uncertainty = 0
        for idx in unlabeled_list:
            if idx not in uncertainty_scores:
                uncertainty_scores[idx] = 0.0

        # Chọn sampling method dựa trên config
        # BUGFIX (10/2026): chỉ sampling trong POOL UNLABELED, không phải
        # filtered_metadata toàn bộ. Phiên bản cũ truyền cả pool (gồm labeled
        # + eval images) vào stratified_sampling: khi 1 stratum có ít hơn
        # allocation ảnh unlabeled, code lấy TOÀN stratum (stratified.py
        # L127) -> kéo theo cả index LABELED/eval vào batch -> eval set bị
        # train (contamination) + duplicate labeled. Cách làm giống hệt
        # initialize(): pool_metadata + map local index -> global index.
        pool_indices = sorted(self.unlabeled_indices)
        pool_metadata = [self.filtered_metadata[i] for i in pool_indices]
        pool_scores = {
            i: uncertainty_scores.get(pool_indices[i], 0.0)
            for i in range(len(pool_indices))
        }

        if self.use_baseline:
            print(f"[ROUND {round_num}] Running BASELINE sampling (uncertainty-only)...")
            selected_local = baseline_sampling(
                uncertainty_scores = pool_scores,
                batch_size = min(self.batch_size, len(pool_indices)),
                metadata_list = pool_metadata,
                stratify_fields = self.stratify_fields
            )
        else:
            print(f"[ROUND {round_num}] Running STRATIFIED sampling (metadata + uncertainty)...")
            selected_local = stratified_sampling(
                metadata_list = pool_metadata,
                uncertainty_scores = pool_scores,
                batch_size = min(self.batch_size, len(pool_indices)),
                stratify_fields= self.stratify_fields
            )
        # Map local index trong pool -> global index trong filtered_metadata
        selected_indices = [pool_indices[i] for i in selected_local]

        # tính diversity của batch đã chọn
        diversity = calculate_batch_diversity(
            self.filtered_metadata,
            selected_indices,
            self.stratify_fields
        )

        print(f"[ROUND {round_num}] Selected {len(selected_indices)} samples")
        print(f"[ROUND {round_num}] Batch diversity:")

        for field, entropy in diversity.items():
            print(f"       - {field}: {entropy:.4f} bits")

        return selected_indices

    def update_labeled_set(self, new_indices: List[int]):
        """
        Cập nhật labeled và unlabeled sets.
        
        Args:
            new_indices: Indices mới được label
        """

        self.labeled_indices.update(new_indices)
        self.unlabeled_indices -= set(new_indices)

        # Validate và update valid_labeled_indices
        for idx in new_indices:
            meta = self.filtered_metadata[idx]
            image_path = meta.get("image_path")

            if image_path and Path(image_path).exists():
                self.valid_labeled_indices.add(idx)

        print(f"[UPDATE] Total labeled: {len(self.labeled_indices)}")
        print(f"[UPDATE] Remaining unlabeled: {len(self.unlabeled_indices)}")

    def save_checkpoint(self, path: str):
        """Save checkpoint (state của AL loop + đường dẫn model weights)."""
        checkpoint = {
            "labeled_indices": list(self.labeled_indices),
            "unlabeled_indices": list(self.unlabeled_indices),
            "valid_labeled_indices": list(self.valid_labeled_indices),
            "results_history": self.results_history,
            "filtered_metadata": self.filtered_metadata,
            "all_metadata": self.all_metadata,
            "eval_indices": self.eval_indices,
            # Đường dẫn tới weights của round vừa rồi để resume warm-start
            "best_model_path": self.trainer.best_model_path,
            "config": {
                "seed": self.seed,
                "use_baseline": self.use_baseline,
                "num_rounds": self.num_rounds,
                "epochs": self.epochs,
            },
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.labeled_indices = set(checkpoint["labeled_indices"])
        self.unlabeled_indices = set(checkpoint["unlabeled_indices"])
        self.valid_labeled_indices = set(checkpoint["valid_labeled_indices"])
        self.results_history = checkpoint["results_history"]
        self.filtered_metadata = checkpoint.get("filtered_metadata", [])
        self.all_metadata = checkpoint["all_metadata"]
        self.eval_indices = checkpoint.get("eval_indices", [])

        # Khôi phục model weights để warm-start từ round trước
        best_model_path = checkpoint.get("best_model_path")
        if best_model_path and Path(best_model_path).exists():
            self.trainer.best_model_path = best_model_path
            self.trainer.load_model(best_model_path)
            print(f"[CHECKPOINT] Restored model weights: {best_model_path}")
        else:
            print(f"[WARNING] Không tìm thấy weights '{best_model_path}', "
                  f"sẽ train lại từ pretrained")

        # Re-validate paths sau khi load checkpoint
        self._validate_labeled_indices()

    def _validate_labeled_indices(self):
        """Re-validate labeled indices để đảm bảo paths tồn tại."""
        valid_labeled = set()
        for idx in self.labeled_indices:
            if idx < len(self.filtered_metadata):
                meta = self.filtered_metadata[idx]
                image_path = meta.get("image_path")
                if image_path and Path(image_path).exists():
                    valid_labeled.add(idx)
                else:
                    # Xóa khỏi unlabeled nếu đã được label nhưng path không tồn tại
                    self.unlabeled_indices.discard(idx)
            else:
                self.unlabeled_indices.discard(idx)

        self.valid_labeled_indices = valid_labeled
        print(f"[VALIDATE] Valid labeled: {len(self.valid_labeled_indices)}/{len(self.labeled_indices)}")

    def run(self, metadata_path: str, image_dirs: List[str], output_dir: str = "results") -> pd.DataFrame:
        """
        Chạy full AL loop.
        
        Args:
            metadata_path: Đường dẫn metadata JSON
            image_dirs: Danh sách thư mục ảnh
            output_dir: Thư mục lưu kết quả
        
        Returns:
            DataFrame chứa kết quả mỗi round
        """
        
        # Khởi tạo
        if len(self.all_metadata) == 0:
            self.initialize(metadata_path, image_dirs)
        
        # Tạo output directory
        # Cấu trúc: results/<experiment_name>/  -> chứa results.csv + checkpoint.pkl
        # (khớp với glob của cell save/restore trong notebook:
        #  results/main_experiment/*/checkpoint.pkl)
        output_path = Path(output_dir) / self.experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Dùng cùng path format cho cả load và save
        checkpoint_path = str(output_path / "checkpoint.pkl")
        results_csv_path = str(output_path / "results.csv")
        
        print(f"\n==================================================")
        print(f"Starting Active Learning Loop")
        print(f"Rounds: {self.num_rounds}, Batch size: {self.batch_size}")
        print(f"==================================================")
        
        # initial training
        start_round = 0
        if Path(checkpoint_path).exists():
            try:
                self.load_checkpoint(checkpoint_path)
                start_round = len(self.results_history)
                print(f"[INFO] Resumed from round {start_round}")
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                start_round = 0

        # Train initial round
        if start_round == 0:
            train_results = self.train_round(round_num=0)

            # Diversity của initial batch (round 0) - quan trọng cho phân tích tương quan
            initial_diversity = calculate_batch_diversity(
                self.filtered_metadata,
                list(self.valid_labeled_indices),
                self.stratify_fields
            )

            self.results_history.append({
                "round": 0,
                "labeled_count": len(self.valid_labeled_indices),
                "map50": train_results.get("eval_map50", train_results.get("map50", 0.0)),
                "map50_95": train_results.get("eval_map50_95", train_results.get("map50_95", 0.0)),
                "val_map50": train_results.get("map50", 0.0),
                "batch_diversity": initial_diversity
            })

            # LƯU CHECKPOINT NGAY SAU ROUND 0
            self.save_checkpoint(checkpoint_path)
            print(f"[CHECKPOINT] Saved after round 0")
        
        # AL rounds
        for round_num in range(start_round + 1, self.num_rounds + 1):
            print(f"\n{'='*50}")
            print(f"ROUND {round_num}/{self.num_rounds}")
            print(f"{'='*50}")
            
            # Select next batch
            new_indices = self.select_batch(round_num)
            
            # Update labeled set
            self.update_labeled_set(new_indices)
            
            # train
            train_results = self.train_round(round_num)
            
            # Calculate diversity
            diversity = calculate_batch_diversity(
                self.filtered_metadata,
                new_indices,
                self.stratify_fields
            )
            
            # Log results
            self.results_history.append({
                "round": round_num,
                "labeled_count": len(self.valid_labeled_indices),
                "map50": train_results.get("eval_map50", train_results.get("map50", 0.0)),
                "map50_95": train_results.get("eval_map50_95", train_results.get("map50_95", 0.0)),
                "val_map50": train_results.get("map50", 0.0),
                "batch_diversity": diversity
            })
            
            # Save checkpoint
            self.save_checkpoint(checkpoint_path)
        
        # Create DataFrame
        df = pd.DataFrame(self.results_history)
        
        # Save to CSV
        df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to: {results_csv_path}")
        
        return df
