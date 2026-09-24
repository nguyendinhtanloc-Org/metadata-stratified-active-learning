"""
Script chạy Active Learning Experiment.
Dùng cho cả local và Kaggle.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from active_learning.loop import ActiveLearningLoop
from sampling.baseline import baseline_sampling
from utils.config import DEFAULT_CONFIG

def run_experiment(
    experiment_name: str,
    pipeline: str = "stratified",
    num_rounds: int = 10,
    batch_size: int = 500,
    initial_budget: int = 1000,
    device: str = "cuda",
    epochs: int = 10,
    seed: int = 42
):
    """
    Chạy experiment với config chỉ định.

    Args:
        experiment_name: Tên experiment
        pipeline: "stratified" hoặc "baseline"
        num_rounds: Số vòng AL
        batch_size: Số ảnh thêm mỗi vòng
        initial_budget: Số ảnh ban đầu
        device: "cuda" hoặc "cpu"
    """

    # load config
    config = DEFAULT_CONFIG.copy()

    # override with arguments
    config["experiment_name"] = experiment_name
    config["num_rounds"] = num_rounds
    config["batch_size"] = batch_size
    config["initial_budget"] = initial_budget
    config["device"] = device
    config["epochs"] = epochs
    config["seed"] = seed

    # Paths - điều chỉnh theo environment
    METADATA_PATH = "data/raw/annotations/bdd100k_labels_images_train.json"
    IMAGE_DIR = "data/raw/images"

    print(f"=" * 50)
    print(f"Running Experiment: {experiment_name}")
    print(f"Pipeline: {pipeline}")
    print(f"Device: {device}")
    print(f"=" * 50)

    # Monkey-patch stratified_sampling nếu là baseline
    if pipeline == "baseline":
        import active_learning.loop as loop_module
        loop_module.stratified_sampling = baseline_sampling
        print("[INFO] Using baseline (uncertainty-only) sampling")

    # Initialize AL loop
    al_loop = ActiveLearningLoop(config, experiment_name)

    # Run
    results_df = al_loop.run(
        metadata_path=METADATA_PATH,
        image_dir=IMAGE_DIR
    )
    
    print("\nFinal Results:")
    print(results_df)
    
    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AL Experiment")
    parser.add_argument("--name", type=str, default="experiment", help="Experiment name")
    parser.add_argument("--pipeline", type=str, default="stratified", 
                       choices=["stratified", "baseline"], help="Sampling pipeline")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--batch", type=int, default=500, help="Batch size")
    parser.add_argument("--budget", type=int, default=1000, help="Initial budget")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")   
    parser.add_argument("--seed", type=int, default=42, help="Random seed") 
    
    args = parser.parse_args()
    
    run_experiment(
        experiment_name=args.name,
        pipeline=args.pipeline,
        num_rounds=args.rounds,
        batch_size=args.batch,
        initial_budget=args.budget,
        device=args.device,
        epochs=args.epochs,
        seed=args.seed
    )

    
