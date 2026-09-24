"""
Active Learning Loop.
Kết hợp tất cả modules để chạy AL experiment.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd 
from datetime import datetime
from random import sample
import random
import pickle
import math

from sampling.uncertainty import calc_image_uncertainty
from sampling.stratified import stratified_sampling, calculate_batch_diversity
from training.trainer import YOLOTrainer
from data.loader import (
    get_all_metadata,
    build_image_path_map,
    filter_images_with_labels
)
from metrics.entropy import calc_batch_entropy

class ActiveLearningLoop:
    """
    Active Learning Loop cho object detection với metadata stratification.
    """

    def __init__(
        self,
        config: Dict,
        experiment_name: str = "al_experiment"
    ):
        """
        Khởi tạo AL loop.
        
        Args:
            config: Dict chứa các tham số cấu hình
            experiment_name: Tên experiment để lưu kết quả
        """

        self.config = config
        self.experiment_name = experiment_name
        self.seed = config.get("seed", 42)

        # các tham số từ config
        self.initial_budget = config.get("initial_budget", 1000)
        self.batch_size = config.get("batch_size", 500)
        self.num_rounds = config.get("num_rounds", 10)
        self.num_classes = config.get("num_classes", 10)
        self.stratify_fields = config.get("stratify_fields", ["weather", "scene", "timeofday"])

        # components
        self.trainer = YOLOTrainer(
            model_name = config.get("model_name", "yolov8n"),
            device = config.get("device", "cuda"),
            project_dir = f"experiments/runs/{experiment_name}"
        )

        # state
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        self.all_metadata = []
        self.history = []
        self.metadata_path = None

    def initialize(self, metadata_path: str, image_dirs: List[str]):
        """
        Khởi tạo: Load metadata, tìm image paths, và chọn initial batch.

        Args:
            metadata_path: Đường dẫn đến annotations JSON
            image_dirs: Danh sách thư mục base chứa ảnh
        """

        random.seed(self.seed)
        self.metadata_path = metadata_path

        print(f"[INIT] Loading metadata from {metadata_path}...")

        # Load metadata
        raw_metadata = get_all_metadata(metadata_path)

        print(f"[INIT] Building image path index...")
        # Build image path map (tìm ảnh trong nhiều thư mục)
        image_path_map = build_image_path_map(image_dirs)
        print(f"[INIT] Found {len(image_path_map)} images")

        # Filter: chỉ giữ samples có ảnh + detection labels
        self.all_metadata = filter_images_with_labels(raw_metadata, image_path_map)
        print(f"[INIT] Valid samples: {len(self.all_metadata)}/{len(raw_metadata)}")

        # tất cả các indicies ban đầu đều chưa có label
        self.unlabeled_indices = set(range(len(self.all_metadata)))
        self.labeled_indices = set()

        print(f"[INIT] Total images: {len(self.all_metadata)}")
        print(f"[INIT] Initial budget: {self.initial_budget}")

        # chọn initial batch ngẫu nhiên (random sampling)
        initial_indices = sample(
            list(self.unlabeled_indices),
            min(self.initial_budget, len(self.unlabeled_indices))
        )

        # chuyển sang labeled
        self.labeled_indices.update(initial_indices)
        self.unlabeled_indices -= set(initial_indices)

        print(f"[INIT] Selected {len(initial_indices)} initial samples")

        # Log initial selection diversity
        initial_metadata = [self.all_metadata[i] for i in initial_indices]

        print("[INIT] Initial batch diversity:")

        for field in self.stratify_fields:
            entropy = calc_batch_entropy(initial_metadata, field)
            print(f"       - {field}: {entropy:.4f} bits")

    def train_round(self, round_num: int) -> Dict:
        """
        Train model trên labeled data của round hiện tại.

        Args:
            round_num: Số thứ tự round

        Returns:
            Dict chứa training metrics
        """

        print(f"\n[ROUND {round_num}] Training model...")
        print(f"[ROUND {round_num}] Labeled samples: {len(self.labeled_indices)}")

        # lấy danh sách đã label (sử dụng image_path đã được resolve)
        labeled_images = []
        labeled_metadata = []

        for i in self.labeled_indices:
            meta = self.all_metadata[i]
            image_path = meta.get("image_path")
            if image_path and Path(image_path).exists():
                labeled_images.append(image_path)
                labeled_metadata.append(meta)

        if len(labeled_images) == 0:
            raise ValueError("Không có ảnh nào để train!")

        # Prepare dataset - truyền labels từ metadata thay vì đọc từ file
        dataset_yaml = self.trainer.prepare_dataset_with_metadata(
            image_paths = labeled_images,
            metadata_list = labeled_metadata,
            output_dir = f"data/processed/{self.experiment_name}/round_{round_num}"
        )

        # train
        save_name = f"round_{round_num}"
        train_results = self.trainer.train(
            dataset_yaml = dataset_yaml,
            epochs = self.config.get("epochs", 10),
            batch_size = self.config.get("batch_size", 16),
            imgsz = self.config.get("imgsz", 640),
            save_name = save_name
        )

        print(f"[ROUND {round_num}] Training complete!")
        print(f"[ROUND {round_num}] mAP50: {train_results['map50']:.4f}")

        return train_results

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

        for idx in unlabeled_list:
            meta = self.all_metadata[idx]
            image_path = meta.get("image_path")

            if not image_path or not Path(image_path).exists():
               uncertainty_scores[idx] = 0.0
               continue

            try:
                result = self.trainer.predict(str(image_path), verbose=False)
                if result is None:
                    # Model không predict được → uncertainty cao nhất
                    uncertainty_scores[idx] = math.log2(self.num_classes)
                else:
                    # Lấy result đầu tiên (nếu là list)
                    if isinstance(result, list):
                        result = result[0]
                    uncertainty_scores[idx] = calc_image_uncertainty(result, self.num_classes)
            except FileNotFoundError:
                print(f"[WARNING] Image not found: {image_path}")
                uncertainty_scores[idx] = 0.0
            except RuntimeError as e:
                # CUDA OOM hoặc lỗi nghiêm trọng
                print(f"[ERROR] Runtime error predicting {image_path}: {e}")
                raise  # Re-raise để không continue âm thầm
            except Exception as e:
                print(f"[WARNING] Failed to predict {image_path}: {e}")
                uncertainty_scores[idx] = 0.0

        # startify sampling
        print(f"[ROUND {round_num}] Running stratified sampling...")

        selected_indices = stratified_sampling(
            metadata_list = self.all_metadata,
            uncertainty_scores = uncertainty_scores,
            batch_size = min(self.batch_size, len(unlabeled_list)),
            stratify_fields= self.stratify_fields
        )

        # tính diversity của batch đã chọn
        diversity = calculate_batch_diversity(
            self.all_metadata,
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

        print(f"[UPDATE] Total labeled: {len(self.labeled_indices)}")
        print(f"[UPDATE] Remaining unlabeled: {len(self.unlabeled_indices)}")

    def run(self, metadata_path: str, image_dir: str) -> pd.DataFrame:
        """
        Chạy toàn bộ Active Learning loop.
        
        Args:
            metadata_path: Đường dẫn đến annotations JSON
            image_dir: Thư mục chứa ảnh
        
        Returns:
            DataFrame chứa lịch sử kết quả các rounds
        """

        # Kiểm tra checkpoint có tồn tại không
        checkpoint_path = f"checkpoints/{self.experiment_name}.pkl"
        start_round = 0

        if Path(checkpoint_path).exists():
            print(f"[INFO] Found checkpoint, loading...")
            self.load_checkpoint(checkpoint_path)
            print(f"[INFO] Resumed with {len(self.labeled_indices)} labeled samples")
    
            # Xác định round bắt đầu từ checkpoint
            if self.history:
                start_round = self.history[-1]["round"] + 1
                print(f"[INFO] Resuming from round {start_round}")
        else:
            self.initialize(metadata_path, image_dir)

        print(f"\n{'='*50}")
        print(f"Starting Active Learning Loop")
        print(f"Rounds: {self.num_rounds}, Batch size: {self.batch_size}")
        print(f"{'='*50}")

        # initial training
        # Chỉ train và log nếu bắt đầu mới hoàn toàn
        if start_round == 0:
            round_num = 0
            train_results = self.train_round(round_num)
            
            self.history.append({
                "round": round_num,
                "labeled_count": len(self.labeled_indices),
                "map50": train_results.get("map50", 0.0),
                "map50_95": train_results.get("map50_95", 0.0),
                "batch_diversity": {}
            })

        # main loop
        for round_num in range(start_round + 1, self.num_rounds + 1):
            print(f"\n{'='*50}")
            print(f"ROUND {round_num}/{self.num_rounds}")
            print(f"{'='*50}")

            # select batch
            new_indices = self.select_batch(round_num)

            if len(new_indices) == 0:
                print("[INFO] No more samples to label. Stopping.")
                break

            # update sets
            self.update_labeled_set(new_indices)

            # train
            train_results = self.train_round(round_num)

            # log
            diversity = calculate_batch_diversity(
                self.all_metadata,
                list(self.labeled_indices),
                self.stratify_fields
            )

            self.history.append({
                "round": round_num,
                "labeled_count": len(self.labeled_indices),
                "map50": train_results.get("map50", 0.0),
                "map50_95": train_results.get("map50_95", 0.0),
                "batch_diversity": diversity
            })

        # create results Dataframe
        df = pd.DataFrame(self.history)

        # Save results
        output_path = f"results/{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        Path("results").mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"\n{'='*50}")
        print("EXPERIMENT COMPLETE!")
        print(f"Results saved to: {output_path}")
        print(f"{'='*50}")

        self.save_checkpoint(f"checkpoints/{self.experiment_name}_round_{round_num}.pkl")
        
        return df

    def save_checkpoint(self, path: str):
        """Lưu checkpoint để có thể resume."""
        
        checkpoint = {
            "labeled_indices": list(self.labeled_indices),
            "unlabeled_indices": list(self.unlabeled_indices),
            "history": self.history,
            "seed": self.seed,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self, path: str):
        """Load checkpoint để resume."""
        
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.labeled_indices = set(checkpoint["labeled_indices"])
        self.unlabeled_indices = set(checkpoint["unlabeled_indices"])
        self.history = checkpoint["history"]
        self.seed = checkpoint.get("seed", 42)


        
