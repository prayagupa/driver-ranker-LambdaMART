"""CLI entrypoint for training pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.train import train_local
from training.evaluate import evaluate_model
from training.export import export_to_onnx, export_to_lightgbm_native
from training.dataset import FEATURE_COLS, load_training_data, train_test_split_by_request


def main():
    parser = argparse.ArgumentParser(description="Train rider-driver matching model")
    parser.add_argument("--data", default="data/training_set.parquet", help="Path to training data")
    parser.add_argument("--rounds", type=int, default=100, help="Number of boosting rounds")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX format")
    parser.add_argument("--export-native", action="store_true", help="Export to LightGBM native format")
    parser.add_argument("--output-dir", default="model", help="Output directory for model")
    args = parser.parse_args()

    print("=" * 60)
    print("RIDER-DRIVER MATCHING MODEL - TRAINING PIPELINE")
    print("=" * 60)

    # Train
    print(f"\n[1/3] Training model ({args.rounds} rounds)...")
    model = train_local(data_path=args.data, num_boost_round=args.rounds)
    print("      ✓ Training complete")

    # Evaluate
    print("\n[2/3] Evaluating model...")
    df = load_training_data(args.data)
    _, test_df = train_test_split_by_request(df)
    metrics = evaluate_model(model, test_df, FEATURE_COLS, k=5)
    for name, value in metrics.items():
        print(f"      {name}: {value:.4f}")

    # Export
    print("\n[3/3] Exporting model...")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.export_onnx:
        export_to_onnx(model, len(FEATURE_COLS), f"{args.output_dir}/ranker.onnx")
    if args.export_native or not args.export_onnx:
        export_to_lightgbm_native(model, f"{args.output_dir}/ranker.txt")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
