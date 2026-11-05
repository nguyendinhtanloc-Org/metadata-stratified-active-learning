"""
Active Learning Loop.
Kết hợp tất cả modules để chạy AL experiment.
"""

from typing import List, Dict
from pathlib import Path
import pandas as pd
import random
import pickle
import math
import gc
import os
import shutil
import warnings
import numpy as np
import torch
try:
    import psutil
except ImportError:
    psutil = None

from sampling.uncertainty import calc_image_uncertainty
from sampling.stratified import stratified_sampling, calculate_batch_diversity
from sampling.baseline import baseline_sampling
from training.trainer import YOLOTrainer
from data.loader import (
    get_all_metadata_cached,
    build_image_path_map_cached,
    filter_images_with_labels,
)


_LIBC = None
try:
    import ctypes as _ctypes
    _LIBC = _ctypes.CDLL("libc.so.6")
    _LIBC.mallopt(-1, 65536)
    _LIBC.malloc_trim(0)
    print("[MALLOC] mallopt(M_TRIM_THRESHOLD=65536) + malloc_trim(0) applied", flush=True)
except Exception as _e:
    _LIBC = None
    print(f"[MALLOC] libc tuning unavailable: {_e}", flush=True)


def _read_cgroup_memory():
    """Đọc RAM đang dùng & giới hạn của cgroup."""
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
    """Main Active Learning loop."""

    def __init__(self, config: dict, experiment_name: str = "experiment"):
        self.experiment_name = experiment_name

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
        self.num_rounds = config.get("num_rounds", 4)
        self.epochs = config.get("epochs", 12)
        self.warmup_epochs = config.get("warmup_epochs", 1.0)
        self.retrain_from_scratch = config.get("retrain_from_scratch", True)
        self.device = config.get("device", 0)
        self.num_classes = config.get("num_classes", 10)
        self.stratify_fields = config.get("stratify_fields", ["weather", "scene", "timeofday"])
        self.model_name = config.get("model_name", "yolov8n")
        self.use_baseline = config.get("use_baseline", False)

        # Inference settings
        self.predict_batch_size = config.get("predict_batch_size", 128)
        self.predict_workers = config.get("predict_workers", 0)
        self.predict_imgsz = config.get("predict_imgsz", 640)
        # Giải pháp A: Candidate sub-pool (None hoặc <=0 sẽ quét toàn bộ 59k ảnh)
        self.candidate_pool_size = config.get("candidate_pool_size", 15000)
        # Hybrid Top-K filtering cho stratified sampling:
        #   0  = phân tầng trên toàn bộ candidate pool (hành vi cũ)
        #   >0 = lọc Top-(batch_size * ratio) ảnh khó nhất TRƯỚC, rồi mới
        #        phân tầng bên trong tập ảnh khó (xem sampling/stratified.py)
        # Dùng để ablation: bật/tắt bằng 1 dòng config.
        self.hybrid_top_k_ratio = config.get("hybrid_top_k_ratio", 0)
        # P1: quota mode cho stratified sampling.
        #   "proportional" (cũ): quota ∝ size stratum -> giữ nguyên phân phối gốc.
        #   "inverse": quota ∝ 1/size -> ưu tiên stratum hiếm (mục tiêu coverage).
        #   "equal": quota chia đều.
        self.quota_mode = config.get("quota_mode", "proportional")
        # quota_scope: "joint" (1 tầng = tổ hợp field) | "per_field"
        # (quota riêng cho từng field -> cân bằng diversity đo theo field).
        self.quota_scope = config.get("quota_scope", "joint")

        self.local_image_cache = config.get("local_image_cache", None)
        self.eval_size = config.get("eval_size", 2000)
        self.eval_split_seed = config.get("eval_split_seed", 2024)
        self.shared_eval_dir = config.get("shared_eval_dir", "data/processed/_shared_eval")
        self.early_stop_patience = config.get("early_stop_patience", 0)

        self.trainer = YOLOTrainer(
            model_name=self.model_name,
            device=self.device,
            project_dir=f"experiments/runs/{experiment_name}",
            train_seed=seed,
            retrain_from_scratch=self.retrain_from_scratch,
            warmup_epochs=self.warmup_epochs,
        )

        self.all_metadata = []
        self.filtered_metadata = []
        self.image_path_map = {}

        self.labeled_indices = set()
        self.unlabeled_indices = set()
        self.valid_labeled_indices = set()
        self.eval_indices = []
        self.results_history = []

    def _cache_images_locally(self, metadata_list: List[dict]) -> int:
        cache_root = Path(self.local_image_cache)
        cache_root.mkdir(parents=True, exist_ok=True)
        copied = skipped = 0
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
                pass
        print(f"[CACHE] Local image cache: {copied} copied, {skipped} already present")
        return copied

    def initialize(self, metadata_path: str, image_dirs: List[str]):
        print(f"[INIT] Loading metadata from {metadata_path}...")
        cache_dir = "data/cache"
        metadata_cache = f"{cache_dir}/metadata_{Path(metadata_path).stem}.pkl"
        path_cache = f"{cache_dir}/image_path_map.pkl"

        self.all_metadata = get_all_metadata_cached(metadata_path, metadata_cache)
        print("[INIT] Building image path index...")
        self.image_path_map = build_image_path_map_cached(image_dirs, path_cache)

        for meta in self.all_metadata:
            filename = meta.get("filename")
            meta["image_path"] = self.image_path_map.get(filename, None)

        self.filtered_metadata = filter_images_with_labels(self.all_metadata, self.image_path_map)
        valid_metadata = self.filtered_metadata

        if self.local_image_cache:
            self._cache_images_locally(valid_metadata)

        print(f"[INIT] Found {len(self.all_metadata)} images")
        print(f"[INIT] Valid samples: {len(valid_metadata)}/{len(self.all_metadata)}")

        self.unlabeled_indices = set(range(len(valid_metadata)))
        self.eval_indices = self._build_eval_set(valid_metadata)
        self.unlabeled_indices -= set(self.eval_indices)

        print(f"[INIT] Total images: {len(valid_metadata)}")
        print(f"[INIT] Eval set (held-out): {len(self.eval_indices)}")
        print(f"[INIT] Initial budget: {self.initial_budget}")

        pool_indices = sorted(self.unlabeled_indices)
        pool_metadata = [self.filtered_metadata[i] for i in pool_indices]

        # Khởi tạo Round 0 với RNG riêng của seed để đồng nhất điểm xuất phát giữa 2 pipeline
        init_rng = random.Random(self.seed)
        initial_local = stratified_sampling(
            metadata_list=pool_metadata,
            uncertainty_scores={i: init_rng.random() for i in range(len(pool_metadata))},
            batch_size=self.initial_budget,
            stratify_fields=self.stratify_fields
        )
        initial_indices = [pool_indices[i] for i in initial_local]

        self.labeled_indices = set(initial_indices)
        self.unlabeled_indices -= self.labeled_indices

        for idx in initial_indices:
            meta = valid_metadata[idx]
            image_path = meta.get("image_path")
            if image_path and Path(image_path).exists():
                self.valid_labeled_indices.add(idx)

        print(f"[INIT] Selected {len(self.labeled_indices)} initial samples")
        diversity = calculate_batch_diversity(
            self.filtered_metadata,
            initial_indices,
            self.stratify_fields
        )
        print(f"[INIT] Initial batch diversity:")
        for field, entropy in diversity.items():
            print(f"       - {field}: {entropy:.4f} bits")

    def _build_eval_set(self, valid_metadata: List[Dict]) -> List[int]:
        if self.eval_size <= 0:
            return []
        ordered = sorted(
            range(len(valid_metadata)),
            key=lambda i: str(valid_metadata[i].get("filename", ""))
        )
        rng = random.Random(self.eval_split_seed)
        chosen = rng.sample(ordered, min(self.eval_size, len(ordered)))
        return sorted(chosen)

    def train_round(self, round_num: int) -> Dict:
        print(f"\n[ROUND {round_num}] Training model...")
        print(f"[ROUND {round_num}] Labeled samples: {len(self.valid_labeled_indices)}")

        if len(self.valid_labeled_indices) == 0:
            raise ValueError("Không có ảnh nào để train!")

        labeled_list = sorted(self.valid_labeled_indices)
        labeled_metadata = [self.filtered_metadata[i] for i in labeled_list]
        image_paths = [self.filtered_metadata[i].get("image_path") for i in labeled_list]

        valid_pairs = [(meta, path) for meta, path in zip(labeled_metadata, image_paths) if path is not None]
        if len(valid_pairs) == 0:
            raise ValueError("Không có ảnh hợp lệ để train!")

        labeled_metadata, image_paths = zip(*valid_pairs)
        labeled_metadata = list(labeled_metadata)
        image_paths = list(image_paths)

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
        print(f"[ROUND {round_num}] Starting training (epochs={self.epochs}, warmup={self.warmup_epochs})...")
        train_results = self.trainer.train(
            dataset_yaml=dataset_yaml,
            epochs=self.epochs,
            batch_size=self.train_batch_size,
            imgsz=640,
            save_name=save_name,
            patience=self.early_stop_patience,
        )

        print(f"[ROUND {round_num}] Training complete!")
        print(f"[ROUND {round_num}] Internal Val mAP50: {train_results['map50']:.4f} | "
              f"Fixed Eval mAP50: {train_results['eval_map50']:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[MEMORY] GPU cache cleared")

        return train_results

    @staticmethod
    def _trim_memory():
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
        scores: Dict[int, float] = {}
        max_entropy = math.log2(self.num_classes)

        if batch_size is None:
            batch_size = self.predict_batch_size
        imgsz = self.predict_imgsz
        device_arg = self.device

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
                        conf=0.05,       # Lọc ngay box nhiễu < 0.05 từ tầng NMS để chạy nhanh & chuẩn hơn
                        iou=0.7,
                        max_det=300,
                        workers=0,
                    )
                    for k, result in enumerate(results_iter):
                        idx = ck_idx[k] if k < len(ck_idx) else None
                        if idx is not None:
                            if result is None:
                                scores[idx] = max_entropy * 0.5
                            else:
                                scores[idx] = calc_image_uncertainty(result, self.num_classes)
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
                          f"failed ({e}) -> dùng entropy trung bình", flush=True)
                    for idx in ck_idx:
                        scores[idx] = max_entropy * 0.5
                    processed += len(ck_idx)

                self._trim_memory()

                if processed % 3072 == 0 or processed == len(image_paths):
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

    def select_batch(self, round_num: int) -> List[int]:
        print(f"\n[ROUND {round_num}] Selecting next batch...")
        print(f"[ROUND {round_num}] Total unlabeled available: {len(self.unlabeled_indices)}")

        all_unlabeled_sorted = sorted(self.unlabeled_indices)

        # GIẢI PHÁP A: Candidate Sub-pool Sampling (Tăng tốc gấp 4 lần & tạo phương sai tự nhiên theo seed)
        if self.candidate_pool_size and 0 < self.candidate_pool_size < len(all_unlabeled_sorted):
            round_rng = random.Random(self.seed * 1000 + round_num)
            candidate_indices = sorted(round_rng.sample(all_unlabeled_sorted, self.candidate_pool_size))
            print(f"[ROUND {round_num}] Sub-sampled candidate pool: {len(candidate_indices)} images (seed={self.seed}, round={round_num})")
        else:
            candidate_indices = all_unlabeled_sorted

        print(f"[ROUND {round_num}] Computing uncertainty scores...")
        uncertainty_scores = {}
        valid_image_paths = []
        valid_indices = []

        for idx in candidate_indices:
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
            for idx in valid_indices:
                uncertainty_scores[idx] = random.random()

        for idx in candidate_indices:
            if idx not in uncertainty_scores:
                uncertainty_scores[idx] = 0.0

        pool_indices = candidate_indices
        pool_metadata = [self.filtered_metadata[i] for i in pool_indices]
        pool_scores = {
            i: uncertainty_scores.get(pool_indices[i], 0.0)
            for i in range(len(pool_indices))
        }

        if self.use_baseline:
            print(f"[ROUND {round_num}] Running BASELINE sampling (uncertainty-only)...")
            selected_local = baseline_sampling(
                uncertainty_scores=pool_scores,
                batch_size=min(self.batch_size, len(pool_indices)),
                metadata_list=pool_metadata,
                stratify_fields=self.stratify_fields
            )
        else:
            print(f"[ROUND {round_num}] Running STRATIFIED sampling (metadata + uncertainty)"
                  + (f" [HYBRID top-{self.hybrid_top_k_ratio}x]" if self.hybrid_top_k_ratio else "")
                  + f" [quota={self.quota_mode}/{self.quota_scope}]")
            selected_local = stratified_sampling(
                metadata_list=pool_metadata,
                uncertainty_scores=pool_scores,
                batch_size=min(self.batch_size, len(pool_indices)),
                stratify_fields=self.stratify_fields,
                hybrid_top_k_ratio=self.hybrid_top_k_ratio,
                quota_mode=self.quota_mode,
                quota_scope=self.quota_scope
            )

        selected_indices = [pool_indices[i] for i in selected_local]
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
        self.labeled_indices.update(new_indices)
        self.unlabeled_indices -= set(new_indices)

        for idx in new_indices:
            meta = self.filtered_metadata[idx]
            image_path = meta.get("image_path")
            if image_path and Path(image_path).exists():
                self.valid_labeled_indices.add(idx)

        print(f"[UPDATE] Total labeled: {len(self.labeled_indices)}")
        print(f"[UPDATE] Remaining unlabeled: {len(self.unlabeled_indices)}")

    def save_checkpoint(self, path: str):
        checkpoint = {
            "labeled_indices": list(self.labeled_indices),
            "unlabeled_indices": list(self.unlabeled_indices),
            "valid_labeled_indices": list(self.valid_labeled_indices),
            "results_history": self.results_history,
            "filtered_metadata": self.filtered_metadata,
            "all_metadata": self.all_metadata,
            "eval_indices": self.eval_indices,
            "best_model_path": self.trainer.best_model_path,
            "config": {
                "seed": self.seed,
                "use_baseline": self.use_baseline,
                "num_rounds": self.num_rounds,
                "epochs": self.epochs,
                "hybrid_top_k_ratio": self.hybrid_top_k_ratio,
                "quota_mode": self.quota_mode,
                "quota_scope": self.quota_scope,
                "candidate_pool_size": self.candidate_pool_size,
                "retrain_from_scratch": self.retrain_from_scratch,
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str):
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.labeled_indices = set(checkpoint["labeled_indices"])
        self.unlabeled_indices = set(checkpoint["unlabeled_indices"])
        self.valid_labeled_indices = set(checkpoint["valid_labeled_indices"])
        self.results_history = checkpoint["results_history"]
        self.filtered_metadata = checkpoint.get("filtered_metadata", [])
        self.all_metadata = checkpoint["all_metadata"]
        self.eval_indices = checkpoint.get("eval_indices", [])
        # Giữ đúng protocol khi resume (tránh lệch giữa các phiên chạy)
        ck_cfg = checkpoint.get("config", {})
        for key in ("hybrid_top_k_ratio", "candidate_pool_size", "retrain_from_scratch",
                    "quota_mode", "quota_scope"):
            if key in ck_cfg:
                setattr(self, key, ck_cfg[key])

        best_model_path = checkpoint.get("best_model_path")
        if best_model_path and Path(best_model_path).exists():
            self.trainer.best_model_path = best_model_path
            self.trainer.load_model(best_model_path)
            print(f"[CHECKPOINT] Restored model weights: {best_model_path}")
        else:
            print(f"[WARNING] Không tìm thấy weights '{best_model_path}', sẽ train lại từ pretrained")

        self._validate_labeled_indices()

    def _validate_labeled_indices(self):
        valid_labeled = set()
        for idx in self.labeled_indices:
            if idx < len(self.filtered_metadata):
                meta = self.filtered_metadata[idx]
                image_path = meta.get("image_path")
                if image_path and Path(image_path).exists():
                    valid_labeled.add(idx)
                else:
                    self.unlabeled_indices.discard(idx)
            else:
                self.unlabeled_indices.discard(idx)

        self.valid_labeled_indices = valid_labeled
        print(f"[VALIDATE] Valid labeled: {len(self.valid_labeled_indices)}/{len(self.labeled_indices)}")

    def run(self, metadata_path: str, image_dirs: List[str], output_dir: str = "results") -> pd.DataFrame:
        if len(self.all_metadata) == 0:
            self.initialize(metadata_path, image_dirs)

        output_path = Path(output_dir) / self.experiment_name
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = str(output_path / "checkpoint.pkl")
        results_csv_path = str(output_path / "results.csv")

        # BƯỚC 2 (domain-specific evaluation): dump metadata của eval set
        # CỐ ĐỊNH (2000 ảnh, seed 2024) ra CSV -> notebook join với
        # predictions.csv của ultralytics theo 'filename' để tính mAP
        # riêng theo domain (night / adverse weather / rare scene).
        # Idempotent: mọi experiment đều ghi cùng nội dung (cùng eval set).
        if self.eval_indices:
            import pandas as pd
            em = [self.filtered_metadata[i] for i in self.eval_indices]
            pd.DataFrame([
                {
                    "filename": m.get("filename"),
                    "image_path": m.get("image_path"),
                    "weather": m.get("weather", "unknown"),
                    "scene": m.get("scene", "unknown"),
                    "timeofday": m.get("timeofday", "unknown"),
                }
                for m in em
            ]).to_csv(str(Path(output_dir) / "_shared_eval_metadata.csv"), index=False)
            print(f"[EVAL-META] {len(em)} rows -> {Path(output_dir) / '_shared_eval_metadata.csv'}")

        print(f"\n==================================================")
        print(f"Starting Active Learning Loop")
        print(f"Rounds: {self.num_rounds}, Batch size: {self.batch_size}")
        print(f"==================================================")

        # BUGFIX: Xác định đúng round tiếp theo cần chạy từ results_history
        next_round = 0
        if Path(checkpoint_path).exists():
            try:
                self.load_checkpoint(checkpoint_path)
                next_round = len(self.results_history)
                print(f"[INFO] Resumed! Next round to run: {next_round}")
            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint: {e}")
                next_round = 0

        if next_round == 0:
            train_results = self.train_round(round_num=0)
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
                "batch_diversity": initial_diversity,
                "per_class_ap50": train_results.get("per_class_ap50", {}),
            })
            self.save_checkpoint(checkpoint_path)
            print(f"[CHECKPOINT] Saved after round 0")
            next_round = 1

        for round_num in range(next_round, self.num_rounds + 1):
            print(f"\n{'='*50}")
            print(f"ROUND {round_num}/{self.num_rounds}")
            print(f"{'='*50}")

            new_indices = self.select_batch(round_num)
            self.update_labeled_set(new_indices)
            train_results = self.train_round(round_num)

            diversity = calculate_batch_diversity(
                self.filtered_metadata,
                new_indices,
                self.stratify_fields
            )

            self.results_history.append({
                "round": round_num,
                "labeled_count": len(self.valid_labeled_indices),
                "map50": train_results.get("eval_map50", train_results.get("map50", 0.0)),
                "map50_95": train_results.get("eval_map50_95", train_results.get("map50_95", 0.0)),
                "val_map50": train_results.get("map50", 0.0),
                "batch_diversity": diversity,
                "per_class_ap50": train_results.get("per_class_ap50", {}),
            })

            self.save_checkpoint(checkpoint_path)
            # Lưu luôn CSV tạm sau mỗi round để an toàn tuyệt đối
            pd.DataFrame(self.results_history).to_csv(results_csv_path, index=False)

        df = pd.DataFrame(self.results_history)
        df.to_csv(results_csv_path, index=False)
        print(f"\nResults saved to: {results_csv_path}")
        return df