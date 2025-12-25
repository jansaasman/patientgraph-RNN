#!/usr/bin/env python
"""
Test script to verify PG-RNN infrastructure works with PatientGraph.

Run from the project root:
    python test_infrastructure.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.agraph_client import PatientGraphClient
from src.query_templates import patient_base_query, event_sequence_query, vocabulary_query
from src.data_extractor import DataExtractor


def test_config():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")
    config = get_config()

    print(f"Repository: {config.agraph.repository}")
    print(f"Host: {config.agraph.host}:{config.agraph.port}")
    print(f"Namespaces: {len(config.namespaces)} registered")
    print(f"Max sequence length: {config.sequence.max_length}")
    print(f"Model type: {config.model.type}")
    print("Config: OK")


def test_connection():
    """Test AllegroGraph connection."""
    print("\n=== Testing Connection ===")

    with PatientGraphClient() as client:
        info = client.test_connection()
        print(f"Connected: {info['connected']}")
        print(f"Repository: {info['repository']}")
        print(f"Triple count: {info['triple_count']:,}")
        print("Connection: OK")


def test_patient_query():
    """Test patient base query."""
    print("\n=== Testing Patient Query ===")

    with PatientGraphClient() as client:
        query = patient_base_query(limit=5)
        df = client.query(query)
        print(f"Query returned {len(df)} patients")
        print(f"Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"Sample patient: {df.iloc[0]['patientId']}")
        print("Patient query: OK")


def test_event_query():
    """Test event sequence query."""
    print("\n=== Testing Event Query ===")

    with PatientGraphClient() as client:
        # First get a patient ID
        patient_query = patient_base_query(limit=1)
        patients_df = client.query(patient_query)

        if len(patients_df) == 0:
            print("No patients found!")
            return

        patient_id = patients_df.iloc[0]['patientId']
        print(f"Testing with patient: {patient_id}")

        # Get events for this patient
        query = event_sequence_query(patient_ids=[patient_id], limit=50)
        events_df = client.query(query)
        print(f"Query returned {len(events_df)} events")

        if len(events_df) > 0:
            print(f"Event types: {events_df['eventType'].value_counts().to_dict()}")
        print("Event query: OK")


def test_vocabulary_query():
    """Test vocabulary query."""
    print("\n=== Testing Vocabulary Query ===")

    with PatientGraphClient() as client:
        query = vocabulary_query(limit=20)
        vocab_df = client.query(query)
        print(f"Query returned {len(vocab_df)} codes")

        if len(vocab_df) > 0:
            print(f"Event types in vocab: {vocab_df['eventType'].unique().tolist()}")
            print(f"Top code: {vocab_df.iloc[0]['code']} (freq: {vocab_df.iloc[0]['frequency']})")
        print("Vocabulary query: OK")


def test_data_extractor():
    """Test the DataExtractor class."""
    print("\n=== Testing DataExtractor ===")

    with DataExtractor() as extractor:
        info = extractor.test_connection()
        print(f"Connected: {info['triple_count']:,} triples")

        # Extract a few patients
        patients_df = extractor.extract_patients(limit=10, cache=False)
        print(f"Extracted {len(patients_df)} patients")

        if len(patients_df) > 0:
            # Get events for first patient
            patient_ids = patients_df['patientId'].tolist()[:3]
            events_df = extractor.extract_event_sequences(
                patient_ids=patient_ids,
                cache=False
            )
            print(f"Extracted {len(events_df)} events for {len(patient_ids)} patients")

        print("DataExtractor: OK")


def test_preprocessor():
    """Test sequence preprocessing."""
    print("\n=== Testing Preprocessor ===")

    try:
        import torch
        from src.sequence_preprocessor import SequencePreprocessor, EventVocabulary
    except ImportError:
        print("PyTorch not installed, skipping preprocessor test")
        return

    # Create a mock vocabulary
    import pandas as pd
    vocab_df = pd.DataFrame({
        'eventType': ['CONDITION', 'CONDITION', 'MEDICATION'],
        'code': ['123', '456', '789'],
        'frequency': [100, 50, 75]
    })

    preprocessor = SequencePreprocessor(min_frequency=1)
    preprocessor.fit(vocab_df)
    print(f"Vocabulary size: {preprocessor.vocab_size}")

    # Create mock events
    events_df = pd.DataFrame({
        'patientId': ['P1', 'P1', 'P1', 'P2', 'P2'],
        'eventType': ['CONDITION', 'MEDICATION', 'CONDITION', 'MEDICATION', 'CONDITION'],
        'eventCode': ['123', '789', '456', '789', '123'],
        'eventDateTime': pd.date_range('2020-01-01', periods=5, freq='D')
    })

    # Transform
    data = preprocessor.transform(events_df)
    print(f"Processed {len(data['patient_ids'])} patients")
    print(f"Sequence lengths: {data['lengths']}")

    # Convert to tensors
    tensors = preprocessor.to_tensors(data)
    print(f"Sequence tensor shape: {tensors['sequences'].shape}")

    print("Preprocessor: OK")


def test_models():
    """Test model creation."""
    print("\n=== Testing Models ===")

    try:
        import torch
        from src.models import create_model, PatientLSTM, PatientGRU
    except ImportError:
        print("PyTorch not installed, skipping model test")
        return

    vocab_size = 1000

    # Test LSTM
    model = create_model('lstm', vocab_size)
    print(f"LSTM model created: {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    seq_len = 50
    sequences = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([50, 40, 30, 20])

    with torch.no_grad():
        output = model(sequences, lengths)

    print(f"Output shape: {output.shape}")
    print("Models: OK")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PG-RNN Infrastructure Test")
    print("=" * 60)

    tests = [
        ("Configuration", test_config),
        ("Connection", test_connection),
        ("Patient Query", test_patient_query),
        ("Event Query", test_event_query),
        ("Vocabulary Query", test_vocabulary_query),
        ("DataExtractor", test_data_extractor),
        ("Preprocessor", test_preprocessor),
        ("Models", test_models),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n!!! {name} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
