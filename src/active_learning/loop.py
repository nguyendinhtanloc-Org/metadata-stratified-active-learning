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

from sampling.uncertainty import calc_image_uncertainty
from sampling.stratified import stratified_sampling, calculate_batch_diversity
from sampling.baseline import baseline_sampling
from training.trainer import YOLOTrainer
from data.loader import (
    get_all_metadata,
    build_image_path_map,
    filter_images_with_labels,
    YOLO_CLASS_NAMES
)
from metrics.entropy import calc_batch_entropy


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
        
        # Random seed
        seed = config.get("seed", 42)
        random.seed(seed)
        
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
        
        # Models và data
        self.trainer = YOLOTrainer(
            model_name=self.model_name,
            device=self.device,
            project_dir=f"experiments/runs/{experiment_name}"
        )
        
        self.all_metadata = []
        self.image_path_map = {}
        
        # Labeled/Unlabeled tracking
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        self.valid_labeled_indices = set()
        
        # Results tracking
        self.results_history = []
        
    def initialize(self, metadata_path: str, image_dirs: List[str]):
        """
        Khởi tạo dataset và chọn initial samples.
        
        Args:
            metadata_path: Đường dẫn đến metadata JSON
            image_dirs: Danh sách thư mục chứa ảnh
        """
        
        print(f"[INIT] Loading metadata from {metadata_path}...")
        
        # Load metadata
        self.all_metadata = get_all_metadata(metadata_path)
        
        # Build image path map
        print("[INIT] Building image path index...")
        self.image_path_map = build_image_path_map(image_dirs)
        
        # Add image paths to metadata
        for meta in self.all_metadata:
            filename = meta.get("filename")
            if filename in self.image_path_map:
                meta["image_path"] = self.image_path_map[filename]
            else:
                meta["image_path"] = None
        
        # Filter images with labels
        valid_metadata = filter_images_with_labels(self.all_metadata, self.image_path_map)
        
        print(f"[INIT] Found {len(self.all_metadata)} images")
        print(f"[INIT] Valid samples: {len(valid_metadata)}/{len(self.all_metadata)}")
        
        # Initialize all as unlabeled
        self.unlabeled_indices = set(range(len(valid_metadata)))
        
        # Chọn initial samples bằng stratified sampling
        print(f"[INIT] Total images: {len(valid_metadata)}")
        print(f"[INIT] Initial budget: {self.initial_budget}")
        
        initial_indices = stratified_sampling(
            metadata_list=valid_metadata,
            uncertainty_scores={i: random.random() for i in range(len(valid_metadata))},
            batch_size=self.initial_budget,
            stratify_fields=self.stratify_fields
        )
        
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
            valid_metadata,
            initial_indices,
            self.stratify_fields
        )
        
        print(f"[INIT] Initial batch diversity:")
        for field, entropy in diversity.items():
            print(f"       - {field}: {entropy:.4f} bits")
    
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
        labeled_metadata = [self.all_metadata[i] for i in labeled_list]
        image_paths = [self.all_metadata[i].get("image_path") for i in labeled_list]
        
        dataset_yaml = self.trainer.prepare_dataset_with_metadata(
            image_paths=image_paths,
            metadata_list=labeled_metadata,
            output_dir=f"data/processed/{self.experiment_name}/round_{round_num}",
            split_seed=42
        )
        
        save_name = f"round_{round_num}"
        
        # train
        print(f"[ROUND {round_num}] Starting training...")
        train_results = self.trainer.train(
            dataset_yaml=dataset_yaml,
            epochs=self.epochs,
            batch_size = self.train_batch_size,
            imgsz=640,
            save_name=save_name
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

        # Lấy danh sách đường dẫn ảnh hợp lệ
        valid_image_paths = []
        valid_indices = []
        
        for idx in unlabeled_list:
            meta = self.all_metadata[idx]
            image_path = meta.get("image_path")
            if image_path and Path(image_path).exists():
                valid_image_paths.append(image_path)
                valid_indices.append(idx)
        
        print(f"[ROUND {round_num}] Predicting {len(valid_image_paths)} images...")
        
        # Compute uncertainty in small batches - XỬ LÝ NGAY, KHÔNG LƯU
        batch_size = 4  # Nhỏ hơn để tránh OOM
        
        try:
            for batch_start in range(0, len(valid_image_paths), batch_size):
                batch_end = min(batch_start + batch_size, len(valid_image_paths))
                batch_paths = valid_image_paths[batch_start:batch_end]
                batch_indices = valid_indices[batch_start:batch_end]
                
                # Predict batch
                batch_results = self.trainer.model.predict(
                    source=batch_paths,
                    batch=len(batch_paths),
                    verbose=False
                )
                
                # Process results immediately
                for idx, result in zip(batch_indices, batch_results):
                    if result is None:
                        uncertainty_scores[idx] = math.log2(self.num_classes)
                    else:
                        if isinstance(result, list):
                            result = result[0]
                        uncertainty_scores[idx] = calc_image_uncertainty(result, self.num_classes)
                
                # Clear GPU memory after each small batch
                del batch_results
                gc.collect()
                if batch_start % 2000 == 0 and batch_start > 0:
                    print(f"      Processed {batch_start}/{len(valid_image_paths)} images")
            
            print(f"      Processed {len(valid_image_paths)}/{len(valid_image_paths)} images")
            
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
        if self.use_baseline:
            print(f"[ROUND {round_num}] Running BASELINE sampling (uncertainty-only)...")
            selected_indices = baseline_sampling(
                uncertainty_scores = uncertainty_scores,
                batch_size = min(self.batch_size, len(unlabeled_list)),
                metadata_list = self.all_metadata,
                stratify_fields = self.stratify_fields
            )
        else:
            print(f"[ROUND {round_num}] Running STRATIFIED sampling (metadata + uncertainty)...")
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

        # Validate và update valid_labeled_indices
        for idx in new_indices:
            meta = self.all_metadata[idx]
            image_path = meta.get("image_path")

            if image_path and Path(image_path).exists():
                self.valid_labeled_indices.add(idx)

        print(f"[UPDATE] Total labeled: {len(self.labeled_indices)}")
        print(f"[UPDATE] Remaining unlabeled: {len(self.unlabeled_indices)}")

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        checkpoint = {
            "labeled_indices": list(self.labeled_indices),
            "unlabeled_indices": list(self.unlabeled_indices),
            "valid_labeled_indices": list(self.valid_labeled_indices),
            "results_history": self.results_history,
            "all_metadata": self.all_metadata
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
        self.all_metadata = checkpoint["all_metadata"]

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
        if Path(checkpoint_path).exists() and len(self.results_history) > 0:
            try:
                self.load_checkpoint(checkpoint_path)
                start_round = len(self.results_history)
                print(f"[INFO] Resumed from round {start_round}")
            except:
                pass
        
        # Train initial round
        if start_round == 0:
            train_results = self.train_round(round_num=0)
            
            self.results_history.append({
                "round": 0,
                "labeled_count": len(self.valid_labeled_indices),
                "map50": train_results.get("map50", 0.0),
                "map50_95": train_results.get("map50_95", 0.0),
                "batch_diversity": {}
            })
            
            self.save_checkpoint(checkpoint_path)
        
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
                self.all_metadata,
                new_indices,
                self.stratify_fields
            )
            
            # Log results
            self.results_history.append({
                "round": round_num,
                "labeled_count": len(self.valid_labeled_indices),
                "map50": train_results.get("map50", 0.0),
                "map50_95": train_results.get("map50_95", 0.0),
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
