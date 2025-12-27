#!/usr/bin/env python
"""
Predict disease risk for a single patient.

Usage:
    python predict_patient.py <patient_id> --model <model_path>
    python predict_patient.py "Aaron Flatley" --model models/heart_failure_model.pt
    python predict_patient.py "John Smith" --model models/afib_model.pt --date 2020-01-01

Examples:
    # Check current risk (or diagnosed status)
    python predict_patient.py "patient-123" --model models/heart_failure_model.pt

    # Point-in-time: what was the risk on a specific date?
    python predict_patient.py "patient-123" --model models/afib_model.pt --date 2018-06-15
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


def get_disease_diagnosis_date(client: PatientGraphClient, patient_id: str,
                                disease_filters: list) -> tuple:
    """
    Check if patient has been diagnosed with the disease.

    Returns:
        (diagnosis_date, condition_label) if diagnosed, (None, None) otherwise
    """
    # Build filter for disease conditions
    filter_conditions = " || ".join([
        f'CONTAINS(LCASE(?label), "{f.lower()}")'
        for f in disease_filters
    ])

    query = f"""{PREFIXES}
    SELECT ?diagDate ?label
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?condition .
        FILTER(?patientId = "{patient_id}")

        ?condition ns28:code ?code ;
                   ns28:startDateTime ?diagDate .
        ?code skos:prefLabel ?label .

        FILTER({filter_conditions})
    }}
    ORDER BY ?diagDate
    LIMIT 1
    """
    result = client.query(query)

    if len(result) > 0:
        diag_date = pd.to_datetime(result.iloc[0]['diagDate'])
        label = result.iloc[0]['label']
        return diag_date, label
    return None, None


def get_patient_events_t047(client: PatientGraphClient, patient_id: str,
                            cutoff_date: datetime = None) -> pd.DataFrame:
    """Get events for a patient, filtering conditions to T047 only."""
    T047_URI = "https://uts.nlm.nih.gov/uts/umls/semantic-network/T047"

    # Add date filter if specified
    date_filter = ""
    if cutoff_date:
        cutoff_str = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S')
        date_filter = f'FILTER(?eventDateTime < "{cutoff_str}"^^xsd:dateTime)'

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
            {date_filter}
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
            {date_filter}
        }}
        UNION
        {{
            ?patient ns28:patientProcedure ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {{
                ?event ns28:code ?eventCode .
            }}
            BIND("PROCEDURE" AS ?eventType)
            {date_filter}
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
            {date_filter}
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
    prediction_date: datetime = None,
    show_attention: bool = True,
    top_k: int = 10
):
    """
    Predict disease risk for a single patient.

    Args:
        patient_id: Patient identifier
        model_path: Path to trained model file
        prediction_date: Optional date for point-in-time prediction
        show_attention: Whether to show attention weights
        top_k: Number of top attention events to show

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

    # Get disease info from checkpoint
    disease_config = checkpoint.get('disease_config', {})
    disease_name = disease_config.get('display_name', 'Disease')
    disease_filters = disease_config.get('case_condition_filters', [])

    print(f"Disease: {disease_name}")

    if prediction_date:
        print(f"Prediction date: {prediction_date.strftime('%Y-%m-%d')}")

    # =========================================================================
    # Step 1: Check if patient already has the disease
    # =========================================================================
    print(f"\nChecking diagnosis status for: {patient_id}")

    with PatientGraphClient() as client:
        diagnosis_date, diagnosis_label = get_disease_diagnosis_date(
            client, patient_id, disease_filters
        )

    if diagnosis_date:
        # Check if diagnosed before the prediction date
        check_date = prediction_date if prediction_date else datetime.now()

        if diagnosis_date.tz_localize(None) <= pd.Timestamp(check_date):
            # Patient already has the disease
            print("\n" + "=" * 60)
            print("PREDICTION RESULT")
            print("=" * 60)
            print(f"Patient ID: {patient_id}")
            print(f"{disease_name} Status: DIAGNOSED")
            print(f"Diagnosis Date: {diagnosis_date.strftime('%Y-%m-%d')}")
            print(f"Condition: {diagnosis_label}")
            print(f"Probability: 100.0%")
            print("=" * 60)

            return {
                'patient_id': patient_id,
                'disease': disease_name,
                'probability': 1.0,
                'risk_level': 'DIAGNOSED',
                'diagnosis_date': diagnosis_date,
                'diagnosis_label': diagnosis_label
            }
        else:
            print(f"  Note: Patient diagnosed on {diagnosis_date.strftime('%Y-%m-%d')} "
                  f"(after prediction date)")

    else:
        print("  Not diagnosed with this condition")

    # =========================================================================
    # Step 2: Get patient events (up to prediction_date if specified)
    # =========================================================================
    print(f"\nFetching events for patient: {patient_id}")

    with PatientGraphClient() as client:
        events_df = get_patient_events_t047(client, patient_id, prediction_date)

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

    # =========================================================================
    # Step 3: Build model and run prediction
    # =========================================================================
    model_state = checkpoint['model_state_dict']

    # Infer model architecture from saved weights
    for key in model_state.keys():
        if 'lstm.weight_hh_l0' in key:
            hidden_size = model_state[key].shape[1]
            break
    else:
        hidden_size = config.model.hidden_size

    num_layers = sum(1 for k in model_state.keys()
                     if 'lstm.weight_hh_l' in k and 'reverse' not in k)
    if num_layers == 0:
        num_layers = config.model.num_layers

    embedding_dim = model_state['embedding.weight'].shape[1]

    model = AttentionPatientRNN(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.0
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Encode events
    sequence = []
    event_info = []

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

    # =========================================================================
    # Step 4: Display results
    # =========================================================================
    print("\n" + "=" * 60)
    if prediction_date:
        print(f"PREDICTION RESULT (as of {prediction_date.strftime('%Y-%m-%d')})")
    else:
        print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Patient ID: {patient_id}")
    print(f"{disease_name} Risk Probability: {probability:.1%}")
    print(f"Risk Level: {risk_level}")

    # Add note if patient was later diagnosed
    if diagnosis_date and prediction_date:
        if diagnosis_date.tz_localize(None) > pd.Timestamp(prediction_date):
            print(f"Note: Patient was later diagnosed on {diagnosis_date.strftime('%Y-%m-%d')}")

    print("=" * 60)

    if show_attention and len(event_info) > 0:
        print(f"\nTop {top_k} events by attention (what the model focused on):")
        print("-" * 60)

        attn_valid = attention[:len(sequence)]
        top_indices = np.argsort(attn_valid)[::-1][:top_k]

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
        'prediction_date': prediction_date,
        'diagnosis_date': diagnosis_date,
        'top_attention': [(event_info[i], attention[i]) for i in top_indices] if show_attention else None
    }


def main():
    parser = argparse.ArgumentParser(description='Predict disease risk for a patient')
    parser.add_argument('patient_id', nargs='?', help='Patient ID to predict')
    parser.add_argument('--model', required=False, default='models/heart_failure_model.pt',
                        help='Path to trained model file')
    parser.add_argument('--date', type=str, default=None,
                        help='Prediction date (YYYY-MM-DD) for point-in-time prediction')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top attention events to show')
    parser.add_argument('--no-attention', action='store_true',
                        help='Skip attention analysis')

    args = parser.parse_args()

    if not args.patient_id:
        parser.print_help()
        print("\n\nExample usage:")
        print("  # Check current status")
        print("  python predict_patient.py 'Aaron Flatley' --model models/heart_failure_model.pt")
        print("")
        print("  # Point-in-time prediction")
        print("  python predict_patient.py 'Aaron Flatley' --model models/afib_model.pt --date 2018-01-01")
        return

    # Parse date if provided
    prediction_date = None
    if args.date:
        try:
            prediction_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"ERROR: Invalid date format '{args.date}'. Use YYYY-MM-DD")
            return

    result = predict_single_patient(
        patient_id=args.patient_id,
        model_path=args.model,
        prediction_date=prediction_date,
        show_attention=not args.no_attention,
        top_k=args.top_k
    )

    if result:
        print(f"\nPrediction complete")


if __name__ == "__main__":
    main()
