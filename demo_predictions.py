#!/usr/bin/env python
"""
Demo helper for disease prediction.

Selects random patients from cases and controls, runs predictions,
and outputs results as a markdown table and optionally CSV.

Usage:
    python demo_predictions.py configs/heart_failure.yaml
    python demo_predictions.py configs/heart_failure.yaml --count 20
    python demo_predictions.py configs/heart_failure.yaml --output results.csv
"""

import sys
import argparse
import csv
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import torch

from src.disease_config import DiseaseConfig
from src.agraph_client import PatientGraphClient
from src.query_templates import PREFIXES
from src.models import AttentionPatientRNN
from src.config import get_config


def get_random_cases(client: PatientGraphClient, disease_filters: list, count: int) -> pd.DataFrame:
    """Get random patients with the disease."""
    filter_conditions = " || ".join([
        f'CONTAINS(LCASE(?label), "{f.lower()}")'
        for f in disease_filters
    ])

    query = f"""{PREFIXES}
    SELECT ?patientId (MIN(?diagDate) AS ?diagnosisDate) (SAMPLE(?condLabel) AS ?condition)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?cond .

        ?cond ns28:code ?code ;
              ns28:startDateTime ?diagDate .
        ?code skos:prefLabel ?label .
        BIND(?label AS ?condLabel)

        FILTER({filter_conditions})
    }}
    GROUP BY ?patientId
    ORDER BY RAND()
    LIMIT {count}
    """
    return client.query(query)


def get_random_controls(client: PatientGraphClient, disease_filters: list,
                        risk_filters: list, count: int) -> pd.DataFrame:
    """Get random at-risk patients without the disease."""
    risk_conditions = " || ".join([
        f'CONTAINS(LCASE(?label), "{f.lower()}")'
        for f in risk_filters
    ])

    disease_conditions = " || ".join([
        f'CONTAINS(LCASE(?diseaseLabel), "{f.lower()}")'
        for f in disease_filters
    ])

    query = f"""{PREFIXES}
    SELECT ?patientId (MAX(?eventDate) AS ?lastEventDate)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?cond .

        ?cond ns28:code ?code ;
              ns28:startDateTime ?eventDate .
        ?code skos:prefLabel ?label .

        # Has risk factors
        FILTER({risk_conditions})

        # Does NOT have the disease
        FILTER NOT EXISTS {{
            ?patient ns28:patientCondition ?diseaseCond .
            ?diseaseCond ns28:code ?diseaseCode .
            ?diseaseCode skos:prefLabel ?diseaseLabel .
            FILTER({disease_conditions})
        }}
    }}
    GROUP BY ?patientId
    ORDER BY RAND()
    LIMIT {count}
    """
    return client.query(query)


def get_patient_events_t047(client: PatientGraphClient, patient_id: str,
                            cutoff_date: datetime = None) -> pd.DataFrame:
    """Get events for a patient, filtering conditions to T047 only."""
    T047_URI = "https://uts.nlm.nih.gov/uts/umls/semantic-network/T047"

    date_filter = ""
    if cutoff_date:
        cutoff_str = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S')
        date_filter = f'FILTER(?eventDateTime < "{cutoff_str}"^^xsd:dateTime)'

    query = f"""{PREFIXES}
    SELECT DISTINCT ?eventDateTime ?eventType ?eventCode ?eventLabel
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
            OPTIONAL {{ ?event ns28:code ?eventCode }}
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


def predict_patient(patient_id: str, prediction_date: datetime,
                    model, vocab, config, client: PatientGraphClient) -> dict:
    """Run prediction for a patient at a specific date."""
    events_df = get_patient_events_t047(client, patient_id, prediction_date)

    if len(events_df) == 0:
        return {'probability': None, 'top_predictors': []}

    events_df['eventDateTime'] = pd.to_datetime(events_df['eventDateTime'])
    events_df = events_df.sort_values('eventDateTime')

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
            'label': row.get('eventLabel', '') or ''
        })

    # Truncate if needed
    max_len = config.sequence.max_length
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
        event_info = event_info[:max_len]

    if len(sequence) == 0:
        return {'probability': None, 'top_predictors': []}

    # Prepare tensors
    seq_tensor = torch.LongTensor([sequence])
    len_tensor = torch.LongTensor([len(sequence)])

    # Get prediction
    with torch.no_grad():
        logits = model(seq_tensor, len_tensor)
        probability = torch.sigmoid(logits).item()
        attention = model.get_attention_weights(seq_tensor, len_tensor)[0].numpy()

    # Get top predictors
    attn_valid = attention[:len(sequence)]
    top_indices = np.argsort(attn_valid)[::-1][:3]

    top_predictors = []
    for idx in top_indices:
        info = event_info[idx]
        label = info['label']
        if not label:
            label = f"{info['type']}_{info['code']}"
        # Shorten label
        if len(label) > 25:
            label = label[:22] + "..."
        top_predictors.append(label)

    return {
        'probability': probability,
        'top_predictors': top_predictors
    }


def run_demo(config_path: str, count: int = 10, output_csv: str = None,
             show_table: bool = True):
    """Run the demo."""
    # Load config
    config = DiseaseConfig.from_yaml(config_path)
    print(f"Disease: {config.display_name}")
    print(f"Case filters: {config.case_condition_filters}")
    print(f"Control filters: {config.control_risk_filters}")

    # Auto-detect model path
    config_name = Path(config_path).stem
    model_path = Path("models") / f"{config_name}_model.pt"

    if not model_path.exists():
        # Try with _matched suffix
        model_path = Path("models") / f"{config_name}_matched_model.pt"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Train a model first: python train_disease.py " + config_path)
        return

    print(f"Model: {model_path}")

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    vocab = checkpoint['vocab']
    app_config = checkpoint['config']

    model_state = checkpoint['model_state_dict']

    # Infer model architecture
    for key in model_state.keys():
        if 'lstm.weight_hh_l0' in key:
            hidden_size = model_state[key].shape[1]
            break
    else:
        hidden_size = app_config.model.hidden_size

    num_layers = sum(1 for k in model_state.keys()
                     if 'lstm.weight_hh_l' in k and 'reverse' not in k)
    if num_layers == 0:
        num_layers = app_config.model.num_layers

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

    # Get cases and controls
    print(f"\nFetching {count} cases and {count} controls...")

    with PatientGraphClient() as client:
        cases_df = get_random_cases(client, config.case_condition_filters, count)
        controls_df = get_random_controls(
            client, config.case_condition_filters, config.control_risk_filters, count
        )

    print(f"Found {len(cases_df)} cases, {len(controls_df)} controls")

    results = []

    # Process cases
    print("\nProcessing cases...")
    with PatientGraphClient() as client:
        for _, row in cases_df.iterrows():
            patient_id = row['patientId']
            diagnosis_date = pd.to_datetime(row['diagnosisDate'])
            condition = row.get('condition', config.display_name)

            # Prediction date = 6 months before diagnosis
            prediction_date = diagnosis_date - timedelta(days=182)

            pred = predict_patient(patient_id, prediction_date, model, vocab, app_config, client)

            if pred['probability'] is not None:
                results.append({
                    'patient_id': patient_id,
                    'group': 'case',
                    'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                    'risk_probability': pred['probability'],
                    'outcome': f"Diagnosed {diagnosis_date.strftime('%Y-%m-%d')}",
                    'top_predictors': ", ".join(pred['top_predictors'])
                })
                print(f"  {patient_id}: {pred['probability']:.1%}")

    # Process controls
    print("\nProcessing controls...")
    with PatientGraphClient() as client:
        for _, row in controls_df.iterrows():
            patient_id = row['patientId']
            last_event = pd.to_datetime(row['lastEventDate'])

            # Prediction date = last event date
            prediction_date = last_event

            pred = predict_patient(patient_id, prediction_date, model, vocab, app_config, client)

            if pred['probability'] is not None:
                results.append({
                    'patient_id': patient_id,
                    'group': 'control',
                    'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                    'risk_probability': pred['probability'],
                    'outcome': 'None',
                    'top_predictors': ", ".join(pred['top_predictors'])
                })
                print(f"  {patient_id}: {pred['probability']:.1%}")

    # Sort by probability descending
    results.sort(key=lambda x: x['risk_probability'], reverse=True)

    # Output markdown table
    if show_table:
        print("\n" + "=" * 80)
        print(f"## {config.display_name} Prediction Demo")
        print("=" * 80)

        # Cases table
        cases = [r for r in results if r['group'] == 'case']
        cases.sort(key=lambda x: x['risk_probability'], reverse=True)

        print(f"\n### Cases (patients with {config.display_name})")
        print(f"Prediction made 6 months before diagnosis\n")
        print("| Patient | Pred. Date | Risk % | Outcome | Top Predictors |")
        print("|---------|------------|--------|---------|----------------|")
        for r in cases:
            patient_short = r['patient_id'][:20] if len(r['patient_id']) > 20 else r['patient_id']
            predictors = r['top_predictors'][:40] + "..." if len(r['top_predictors']) > 40 else r['top_predictors']
            print(f"| {patient_short} | {r['prediction_date']} | {r['risk_probability']:.1%} | {r['outcome']} | {predictors} |")

        # Controls table
        controls = [r for r in results if r['group'] == 'control']
        controls.sort(key=lambda x: x['risk_probability'], reverse=True)

        print(f"\n### Controls (at-risk, no {config.display_name})")
        print(f"Prediction made at last event date\n")
        print("| Patient | Pred. Date | Risk % | Outcome | Top Predictors |")
        print("|---------|------------|--------|---------|----------------|")
        for r in controls:
            patient_short = r['patient_id'][:20] if len(r['patient_id']) > 20 else r['patient_id']
            predictors = r['top_predictors'][:40] + "..." if len(r['top_predictors']) > 40 else r['top_predictors']
            print(f"| {patient_short} | {r['prediction_date']} | {r['risk_probability']:.1%} | {r['outcome']} | {predictors} |")

    # Output CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'patient_id', 'group', 'prediction_date', 'risk_probability', 'outcome', 'top_predictors'
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to: {output_csv}")

    # Summary stats
    case_probs = [r['risk_probability'] for r in results if r['group'] == 'case']
    control_probs = [r['risk_probability'] for r in results if r['group'] == 'control']

    print(f"\n### Summary")
    if case_probs:
        print(f"Cases: mean={np.mean(case_probs):.1%}, min={np.min(case_probs):.1%}, max={np.max(case_probs):.1%}")
    if control_probs:
        print(f"Controls: mean={np.mean(control_probs):.1%}, min={np.min(control_probs):.1%}, max={np.max(control_probs):.1%}")


def main():
    parser = argparse.ArgumentParser(description='Demo disease predictions')
    parser.add_argument('config', help='Path to YAML config file')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of patients per group (default: 10)')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to CSV file')
    parser.add_argument('--no-table', action='store_true',
                        help='Skip markdown table output')

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config file not found: {args.config}")
        return

    run_demo(
        config_path=args.config,
        count=args.count,
        output_csv=args.output,
        show_table=not args.no_table
    )


if __name__ == "__main__":
    main()
