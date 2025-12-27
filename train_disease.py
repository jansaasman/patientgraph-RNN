#!/usr/bin/env python
"""
Generic disease prediction trainer.

Usage:
    python train_disease.py configs/heart_failure.yaml
    python train_disease.py configs/nephropathy.yaml
    python train_disease.py configs/atrial_fibrillation.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.disease_config import DiseaseConfig
from src.generic_trainer import DiseasePredictor


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_disease.py <config.yaml>")
        print("\nAvailable configs:")
        config_dir = Path("configs")
        if config_dir.exists():
            for f in config_dir.glob("*.yaml"):
                print(f"  {f}")
        sys.exit(1)

    config_path = sys.argv[1]

    if not Path(config_path).exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Load configuration
    config = DiseaseConfig.from_yaml(config_path)
    print(f"Loaded config: {config.display_name}")
    print(f"  Case filters: {config.case_condition_filters}")
    print(f"  Control risk filters: {config.control_risk_filters}")

    # Create predictor and train
    predictor = DiseasePredictor(config)
    results = predictor.train()

    print(f"\nFinal Results:")
    print(f"  AUROC: {results['test_auroc']:.4f}")
    print(f"  AUPRC: {results['test_auprc']:.4f}")


if __name__ == "__main__":
    main()
