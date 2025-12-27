#!/usr/bin/env python
"""
Predict disease risk for a single patient.

Usage:
    python predict_patient.py <patient_id> --model <model_path>
    python predict_patient.py "Aaron Flatley" --model models/heart_failure_model.pt
    python predict_patient.py "John Smith" --model models/nephropathy_model.pt

Example:
    python predict_patient.py "patient-123" --model models/heart_failure_model.pt
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import torch

from src.agraph_client import PatientGraphClient
from src.query_templates import PREFIXES
from src.models import AttentionPatientRNN
from src.config import get_config


def get_patient_events_t047(client: PatientGraphClient, patient_id: str) -> pd.DataFrame:
    """Get all events for a single patient, filtering conditions to T047 only."""
    T047_URI = "https://uts.nlm.nih.gov/uts/umls/semantic-network/T047"

    query = f"""{PREFIXES}
    SELECT DISTINCT
        ?patientId
        ?eventDateTime
        ?eventType
        ?eventCode
        ?eventLabel
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId .
        FILTER(?patientId = "{patient_id}")

        {{
            ?patient ns28:patientCondition ?event .
            ?event ns28:startDateTime ?eventDateTime ;
                   ns28:code ?codeUri .
            ?codeUri skos:notation ?eventCode ;
                     a <{T047_URI}> .
            OPTIONAL {{ ?codeUri skos:prefLabel ?eventLabel }}
            BIND("CONDITION" AS ?eventType)
        }}
        UNION
        {{
            ?patient ns28:patientMedication ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {{
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
                ?codeUri skos:prefLabel ?eventLabel .
            }}
            BIND("MEDICATION" AS ?eventType)
        }}
        UNION
        {{
            ?patient ns28:patientProcedure ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {{
                ?event ns28:code ?eventCode .
            }}
            BIND("PROCEDURE" AS ?eventType)
        }}
        UNION
        {{
            ?patient ns28:patientObservation ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {{
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
                ?codeUri skos:prefLabel ?eventLabel .
            }}
            BIND("OBSERVATION" AS ?eventType)
        }}
    }}
    ORDER BY ?eventDateTime
    """
    return client.query(query)


def lookup_codes(client: PatientGraphClient, codes: list) -> dict:
    """Look up code labels."""
    if not codes:
        return {}
    code_values = ' '.join([f'"{c}"' for c in codes])
    query = f"""{PREFIXES}
    SELECT ?code ?label WHERE {{
        ?codeUri skos:notation ?code ;
                 skos:prefLabel ?label .
        VALUES ?code {{ {code_values} }}
    }}
    """
    results = client.query(query)
    return dict(zip(results['code'], results['label']))


def predict_single_patient(
    patient_id: str,
    model_path: str,
    show_attention: bool = True,
    top_k: int = 10
):
    """
    Predict disease risk for a single patient.

    Returns:
        dict with prediction probability, risk level, and top attention events
    """
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model file not found: {model_path}")
        return None

    print(f"Loading model: {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    vocab = checkpoint['vocab']
    config = checkpoint['config']

    # Get disease info from checkpoint if available
    disease_config = checkpoint.get('disease_config', {})
    disease_name = disease_config.get('display_name', 'Disease')

    print(f"Disease: {disease_name}")

    # Determine model architecture from checkpoint or config
    # Check if it's a smaller model (for small datasets like AF)
    model_state = checkpoint['model_state_dict']

    # Infer hidden size from the model state
    for key in model_state.keys():
        if 'lstm.weight_hh_l0' in key:
            hidden_size = model_state[key].shape[1]
            break
    else:
        hidden_size = config.model.hidden_size

    # Infer number of layers
    num_layers = sum(1 for k in model_state.keys() if 'lstm.weight_hh_l' in k and 'reverse' not in k)
    if num_layers == 0:
        num_layers = config.model.num_layers

    # Infer embedding dim
    embedding_dim = model_state['embedding.weight'].shape[1]

    # Recreate model architecture
    model = AttentionPatientRNN(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0  # No dropout during inference
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nFetching events for patient: {patient_id}")

    # Get patient events (T047 filtered, matching training)
    with PatientGraphClient() as client:
        events_df = get_patient_events_t047(client, patient_id)

    if len(events_df) == 0:
        print(f"ERROR: No events found for patient {patient_id}")
        return None

    print(f"Found {len(events_df)} events")

    # Sort by time
    events_df['eventDateTime'] = pd.to_datetime(events_df['eventDateTime'])
    events_df = events_df.sort_values('eventDateTime')

    # Show event summary
    event_counts = events_df['eventType'].value_counts()
    print("\nEvent breakdown:")
    for etype, count in event_counts.items():
        print(f"  {etype}: {count}")

    # Encode events
    sequence = []
    event_info = []  # Store info for attention display

    for _, row in events_df.iterrows():
        event_type = row['eventType']
        code = str(row['eventCode']) if pd.notna(row['eventCode']) else 'UNK'
        idx = vocab.encode(event_type, code)
        sequence.append(idx)
        event_info.append({
            'type': event_type,
            'code': code,
            'label': row.get('eventLabel', ''),
            'date': row['eventDateTime'],
            'token_id': idx
        })

    # Truncate if needed
    max_len = config.sequence.max_length
    if len(sequence) > max_len:
        print(f"\nNote: Truncating sequence from {len(sequence)} to {max_len} events")
        sequence = sequence[:max_len]
        event_info = event_info[:max_len]

    # Prepare tensors
    seq_tensor = torch.LongTensor([sequence])
    len_tensor = torch.LongTensor([len(sequence)])

    # Get prediction
    with torch.no_grad():
        logits = model(seq_tensor, len_tensor)
        probability = torch.sigmoid(logits).item()
        attention = model.get_attention_weights(seq_tensor, len_tensor)[0].numpy()

    # Determine risk level
    if probability < 0.3:
        risk_level = "LOW"
    elif probability < 0.6:
        risk_level = "MODERATE"
    else:
        risk_level = "HIGH"

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Patient ID: {patient_id}")
    print(f"{disease_name} Risk Probability: {probability:.1%}")
    print(f"Risk Level: {risk_level}")
    print("=" * 60)

    if show_attention and len(event_info) > 0:
        print(f"\nTop {top_k} events by attention (what the model focused on):")
        print("-" * 60)

        # Get top attention indices
        attn_valid = attention[:len(sequence)]
        top_indices = np.argsort(attn_valid)[::-1][:top_k]

        # Look up any unknown codes
        codes_to_lookup = []
        for idx in top_indices:
            info = event_info[idx]
            if info['code'] != 'UNK' and not info.get('label'):
                codes_to_lookup.append(info['code'])

        if codes_to_lookup:
            with PatientGraphClient() as client:
                code_labels = lookup_codes(client, codes_to_lookup)
        else:
            code_labels = {}

        for rank, idx in enumerate(top_indices, 1):
            info = event_info[idx]
            attn_score = attn_valid[idx]

            # Get label
            label = info.get('label') or code_labels.get(info['code'], '')
            if not label and info['code'] != 'UNK':
                label = f"(code: {info['code']})"

            date_str = info['date'].strftime('%Y-%m-%d')
            print(f"  {rank:2d}. [{attn_score:.3f}] {info['type']:12s} | {date_str} | {label[:50]}")

    return {
        'patient_id': patient_id,
        'disease': disease_name,
        'probability': probability,
        'risk_level': risk_level,
        'n_events': len(events_df),
        'top_attention': [(event_info[i], attention[i]) for i in top_indices] if show_attention else None
    }


def main():
    parser = argparse.ArgumentParser(description='Predict disease risk for a patient')
    parser.add_argument('patient_id', nargs='?', help='Patient ID to predict')
    parser.add_argument('--model', required=False, default='models/heart_failure_model.pt',
                        help='Path to trained model file')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top attention events to show')
    parser.add_argument('--no-attention', action='store_true', help='Skip attention analysis')

    args = parser.parse_args()

    if not args.patient_id:
        parser.print_help()
        print("\n\nExample usage:")
        print("  python predict_patient.py 'Aaron Flatley' --model models/heart_failure_model.pt")
        print("  python predict_patient.py 'John Smith' --model models/nephropathy_model.pt")
        return

    result = predict_single_patient(
        patient_id=args.patient_id,
        model_path=args.model,
        show_attention=not args.no_attention,
        top_k=args.top_k
    )

    if result:
        print(f"\nPrediction complete")


if __name__ == "__main__":
    main()
