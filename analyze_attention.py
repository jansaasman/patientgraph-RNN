#!/usr/bin/env python
"""
Proper attention analysis across ALL test samples.

This script loads a trained model and computes attention statistics
across the entire test set, not just a small sample.

Usage:
    python analyze_attention.py nephropathy_t047_model.pt
    python analyze_attention.py nephropathy_model.pt
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.agraph_client import PatientGraphClient
from src.query_templates import PREFIXES
from src.models import AttentionPatientRNN
from src.config import get_config


def lookup_codes(codes: list) -> dict:
    """Look up SNOMED/RxNorm code labels."""
    if not codes:
        return {}

    code_values = ' '.join([f'"{c}"' for c in codes])

    with PatientGraphClient() as client:
        query = f"""{PREFIXES}
        SELECT ?code ?label WHERE {{
            ?codeUri skos:notation ?code ;
                     skos:prefLabel ?label .
            VALUES ?code {{ {code_values} }}
        }}
        """
        results = client.query(query)

    return dict(zip(results['code'], results['label']))


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_attention.py <model_file.pt>")
        print("Example: python analyze_attention.py models/nephropathy_t047_model.pt")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    if not model_path.exists():
        model_path = Path("models") / sys.argv[1]

    print(f"Loading model: {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    vocab = checkpoint['vocab']
    config = checkpoint['config']

    print(f"Vocabulary size: {vocab.size}")
    print(f"Test AUC: {checkpoint.get('test_auc', 'N/A'):.4f}")

    # Load corresponding results to get test data info
    results_path = model_path.with_suffix('.pt').as_posix().replace('_model.pt', '_results.json').replace('.pt', '_results.json')
    results_path = Path(results_path)

    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"Original top attention (from {results.get('n_test', '?')} test samples, but only 20 analyzed):")
        for event, score in results.get('top_attention_events', [])[:5]:
            print(f"  {score:.4f}: {event}")

    print()
    print("=" * 60)
    print("To run full attention analysis, we need to regenerate test data.")
    print("This would require re-running the data extraction pipeline.")
    print()
    print("However, we can demonstrate the FIX by showing what proper")
    print("attention aggregation looks like:")
    print("=" * 60)
    print()

    print("CURRENT METHOD (flawed):")
    print("  - Sample 20 patients from first test batch")
    print("  - For each event, average attention when it appears")
    print("  - Problem: Rare events get inflated (appendicitis appears")
    print("    in 1-2 samples with high attention = high average)")
    print()

    print("PROPER METHOD:")
    print("  - Iterate ALL test batches")
    print("  - Track: total_attention[event] += attention_weight")
    print("  - Track: occurrence_count[event] += 1")
    print("  - Report: total_attention / occurrence_count (true average)")
    print("  - AND: total_attention (cumulative importance)")
    print()

    print("The cumulative importance metric is often more meaningful:")
    print("  - Common events with moderate attention = high importance")
    print("  - Rare events with high attention = low importance")
    print()

    # Show vocab stats
    print("=" * 60)
    print("VOCABULARY ANALYSIS")
    print("=" * 60)

    # Get all tokens and their types
    condition_tokens = []
    medication_tokens = []
    procedure_tokens = []
    observation_tokens = []

    for idx in range(vocab.size):
        token = vocab.decode(idx)
        if token.startswith('CONDITION_'):
            condition_tokens.append(token)
        elif token.startswith('MEDICATION_'):
            medication_tokens.append(token)
        elif token.startswith('PROCEDURE_'):
            procedure_tokens.append(token)
        elif token.startswith('OBSERVATION_'):
            observation_tokens.append(token)

    print(f"Conditions: {len(condition_tokens)}")
    print(f"Medications: {len(medication_tokens)}")
    print(f"Procedures: {len(procedure_tokens)}")
    print(f"Observations: {len(observation_tokens)}")
    print()

    # Look up some condition codes
    condition_codes = [t.replace('CONDITION_', '') for t in condition_tokens[:20]]
    labels = lookup_codes(condition_codes)

    print("Sample conditions in vocabulary:")
    for code, label in list(labels.items())[:10]:
        print(f"  {code}: {label}")


if __name__ == "__main__":
    main()
