#!/usr/bin/env python
"""
Predict nephropathy risk for a single patient.

Usage:
    python predict_patient.py <patient_id>
    python predict_patient.py <patient_id> --model nephropathy_t047_model.pt
    python predict_patient.py --list  # List available diabetic patients

Example:
    python predict_patient.py "patient-123"
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


def get_patient_events(client: PatientGraphClient, patient_id: str, t047_only: bool = False) -> pd.DataFrame:
    """Get all events for a single patient."""

    if t047_only:
        # T047 conditions only
        condition_block = f"""
            ?patient ns28:patientCondition ?event .
            ?event ns28:startDateTime ?eventDateTime .
            ?event ns28:code ?codeUri .
            ?codeUri skos:notation ?eventCode .
            ?codeUri a <https://uts.nlm.nih.gov/uts/umls/semantic-network/T047> .
            BIND("CONDITION" AS ?eventType)
        """
    else:
        # All conditions
        condition_block = """
            ?patient ns28:patientCondition ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
            }
            BIND("CONDITION" AS ?eventType)
        """

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
            {condition_block}
            OPTIONAL {{ ?codeUri skos:prefLabel ?eventLabel }}
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


def get_diabetic_patients(client: PatientGraphClient, limit: int = 20) -> pd.DataFrame:
    """Get list of diabetic patients for testing."""
    query = f"""{PREFIXES}
    SELECT DISTINCT ?patientId (SAMPLE(?label) as ?diabetesType)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?condition .

        ?condition ns28:code ?code .
        ?code skos:prefLabel ?label .

        FILTER(
            CONTAINS(LCASE(?label), "diabetes") ||
            CONTAINS(LCASE(?label), "diabetic")
        )
    }}
    GROUP BY ?patientId
    LIMIT {limit}
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
    model_path: str = "models/nephropathy_t047_model.pt",
    show_attention: bool = True,
    top_k: int = 10
):
    """
    Predict nephropathy risk for a single patient.

    Returns:
        dict with prediction probability, risk level, and top attention events
    """
    model_path = Path(model_path)
    if not model_path.exists():
        model_path = Path("models") / model_path.name

    # Determine if T047 model
    t047_only = "t047" in model_path.name.lower()

    print(f"Loading model: {model_path}")
    print(f"Condition filter: {'T047 only' if t047_only else 'All conditions'}")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    vocab = checkpoint['vocab']
    config = checkpoint['config']

    # Recreate model architecture
    model = AttentionPatientRNN(
        vocab_size=len(vocab),
        embedding_dim=config.model.embedding_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"\nFetching events for patient: {patient_id}")

    # Get patient events
    with PatientGraphClient() as client:
        events_df = get_patient_events(client, patient_id, t047_only=t047_only)

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
    print(f"Nephropathy Risk Probability: {probability:.1%}")
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
            print(f"  {rank:2d}. [{attn_score:.3f}] {info['type']:12s} | {date_str} | {label[:40]}")

    return {
        'patient_id': patient_id,
        'probability': probability,
        'risk_level': risk_level,
        'n_events': len(events_df),
        'top_attention': [(event_info[i], attention[i]) for i in top_indices] if show_attention else None
    }


def main():
    parser = argparse.ArgumentParser(description='Predict nephropathy risk for a patient')
    parser.add_argument('patient_id', nargs='?', help='Patient ID to predict')
    parser.add_argument('--model', default='nephropathy_t047_model.pt', help='Model file to use')
    parser.add_argument('--list', action='store_true', help='List available diabetic patients')
    parser.add_argument('--top-k', type=int, default=10, help='Number of top attention events to show')
    parser.add_argument('--no-attention', action='store_true', help='Skip attention analysis')

    args = parser.parse_args()

    if args.list:
        print("Fetching diabetic patients...")
        with PatientGraphClient() as client:
            patients = get_diabetic_patients(client, limit=30)
        print(f"\nFound {len(patients)} diabetic patients:\n")
        for _, row in patients.iterrows():
            print(f"  {row['patientId']}: {row['diabetesType']}")
        return

    if not args.patient_id:
        parser.print_help()
        print("\n\nExample usage:")
        print("  python predict_patient.py --list")
        print("  python predict_patient.py 'patient-abc123'")
        return

    result = predict_single_patient(
        patient_id=args.patient_id,
        model_path=args.model,
        show_attention=not args.no_attention,
        top_k=args.top_k
    )

    if result:
        print(f"\n✓ Prediction complete")


if __name__ == "__main__":
    main()
