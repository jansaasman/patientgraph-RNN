#!/usr/bin/env python
"""
Train atrial fibrillation prediction model with MATCHED case-control design.

Design:
1. Cases: Patients with atrial fibrillation
2. Controls: Patients with cardiovascular risk factors (hypertension, obesity, diabetes) but no AF
3. Cases: ALL events up to AF diagnosis (no lookback limit)
4. Controls: 5-year lookback window before last event
5. Matched on demographics (age, gender)

Note: Only 86 AF cases exist, so we use 4:1 control ratio and careful matching.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

from src.agraph_client import PatientGraphClient
from src.query_templates import PREFIXES
from src.sequence_preprocessor import SequencePreprocessor
from src.models import AttentionPatientRNN
from src.config import get_config


def get_afib_patients(client: PatientGraphClient) -> pd.DataFrame:
    """Get all patients with atrial fibrillation and their demographics."""
    query = f"""{PREFIXES}
    SELECT ?patientId
           (MIN(?diagDate) AS ?afibDate)
           (SAMPLE(?bd) AS ?birthDate)
           (SAMPLE(?g) AS ?gender)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?condition .

        OPTIONAL {{ ?patient ns28:birthDate ?bd }}
        OPTIONAL {{ ?patient ns28:gender ?g }}

        ?condition ns28:code ?code ;
                   ns28:startDateTime ?diagDate .

        ?code skos:prefLabel ?label .

        FILTER(CONTAINS(LCASE(?label), "atrial fibrillation"))
    }}
    GROUP BY ?patientId
    ORDER BY ?patientId
    """
    return client.query(query)


def get_cardiovascular_risk_patients(client: PatientGraphClient) -> pd.DataFrame:
    """
    Get patients with cardiovascular risk factors (potential controls).
    Excludes patients who have atrial fibrillation.
    """
    query = f"""{PREFIXES}
    SELECT ?patientId
           (MIN(?diagDate) AS ?riskDate)
           (SAMPLE(?bd) AS ?birthDate)
           (SAMPLE(?g) AS ?gender)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?condition .

        OPTIONAL {{ ?patient ns28:birthDate ?bd }}
        OPTIONAL {{ ?patient ns28:gender ?g }}

        ?condition ns28:code ?code ;
                   ns28:startDateTime ?diagDate .

        ?code skos:prefLabel ?label .

        # Cardiovascular risk factors
        FILTER(
            CONTAINS(LCASE(?label), "hypertension") ||
            CONTAINS(LCASE(?label), "obesity") ||
            CONTAINS(LCASE(?label), "diabetes") ||
            CONTAINS(LCASE(?label), "hyperlipidemia") ||
            CONTAINS(LCASE(?label), "coronary")
        )

        # Exclude AF patients
        FILTER NOT EXISTS {{
            ?patient ns28:patientCondition ?afCond .
            ?afCond ns28:code ?afCode .
            ?afCode skos:prefLabel ?afLabel .
            FILTER(CONTAINS(LCASE(?afLabel), "atrial fibrillation"))
        }}
    }}
    GROUP BY ?patientId
    ORDER BY ?patientId
    """
    return client.query(query)


def get_patient_events_t047(client: PatientGraphClient, patient_ids: list) -> pd.DataFrame:
    """Get clinical events for patients, filtering CONDITIONS to T047 only."""
    patient_values = " ".join([f'"{pid}"' for pid in patient_ids])
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
        VALUES ?patientId {{ {patient_values} }}

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
            OPTIONAL {{ ?event ns28:code ?eventCode }}
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
    ORDER BY ?patientId ?eventDateTime
    """
    return client.query(query)


def match_controls(cases_df: pd.DataFrame, potential_controls_df: pd.DataFrame,
                   ratio: int = 4) -> pd.DataFrame:
    """Match controls to cases based on demographics."""
    ref_date = pd.Timestamp('2024-01-01')

    cases_df = cases_df.copy()
    potential_controls_df = potential_controls_df.copy()

    cases_df['birthDate'] = pd.to_datetime(cases_df['birthDate'], errors='coerce')
    potential_controls_df['birthDate'] = pd.to_datetime(potential_controls_df['birthDate'], errors='coerce')

    cases_df['age'] = (ref_date - cases_df['birthDate']).dt.days / 365.25
    potential_controls_df['age'] = (ref_date - potential_controls_df['birthDate']).dt.days / 365.25

    matched_controls = []
    used_control_ids = set()

    for _, case in cases_df.iterrows():
        case_gender = case.get('gender', '')
        case_age = case.get('age', np.nan)

        candidates = potential_controls_df[
            ~potential_controls_df['patientId'].isin(used_control_ids)
        ].copy()

        if pd.notna(case_gender) and case_gender:
            gender_match = candidates['gender'] == case_gender
            if gender_match.any():
                candidates = candidates[gender_match]

        if pd.notna(case_age):
            candidates['age_diff'] = abs(candidates['age'] - case_age)
            candidates = candidates[candidates['age_diff'] <= 10]  # Wider age range due to fewer cases
            candidates = candidates.sort_values('age_diff')

        selected = candidates.head(ratio)

        for _, ctrl in selected.iterrows():
            matched_controls.append(ctrl)
            used_control_ids.add(ctrl['patientId'])

    if matched_controls:
        return pd.DataFrame(matched_controls)
    else:
        return pd.DataFrame(columns=potential_controls_df.columns)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        sequences, lengths, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            sequences, lengths, labels = [x.to(device) for x in batch]
            outputs = model(sequences, lengths)
            loss = criterion(outputs.squeeze(), labels.float())
            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)

    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
        auprc = average_precision_score(all_labels, all_preds)
    else:
        auc = 0.5
        auprc = sum(all_labels) / len(all_labels)

    return avg_loss, auc, auprc, all_preds, all_labels


def main():
    print("=" * 70)
    print("ATRIAL FIBRILLATION PREDICTION - MATCHED CASE-CONTROL DESIGN")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Parameters
    lookback_years = 5
    lookback_days = lookback_years * 365
    prediction_gap_days = 182  # 6-month prediction horizon
    control_ratio = 4  # 4 controls per case (more due to few cases)
    batch_size = 16  # Smaller batch due to fewer samples
    epochs = 50  # More epochs for small dataset
    learning_rate = 0.0005  # Lower LR for small dataset

    print(f"\nDesign: Matched case-control")
    print(f"  Prediction gap: {prediction_gap_days} days ({prediction_gap_days/30:.0f} months)")
    print(f"  Cases: events up to (AF diagnosis - {prediction_gap_days} days)")
    print(f"  Controls: {lookback_years}-year window up to (last event - {prediction_gap_days} days)")
    print(f"  Control ratio: {control_ratio}:1")
    print(f"  Controls: cardiovascular risk patients without AF")
    print(f"  Matching: age (±10 years), gender")

    # =========================================================================
    # Step 1: Get AF patients (CASES)
    # =========================================================================
    print("\n--- Step 1: Getting atrial fibrillation patients (cases) ---")

    with PatientGraphClient() as client:
        cases_df = get_afib_patients(client)

    cases_df['afibDate'] = pd.to_datetime(cases_df['afibDate'])
    print(f"Total AF patients (cases): {len(cases_df)}")

    # Use ALL events up to (AF diagnosis - prediction gap)
    cases_df['cutoffDate'] = cases_df['afibDate'] - pd.Timedelta(days=prediction_gap_days)
    cases_df['windowStart'] = pd.Timestamp('1900-01-01', tz='UTC')
    cases_df['outcome'] = 1

    # =========================================================================
    # Step 2: Get cardiovascular risk patients (potential controls)
    # =========================================================================
    print("\n--- Step 2: Getting cardiovascular risk patients (potential controls) ---")

    with PatientGraphClient() as client:
        potential_controls = get_cardiovascular_risk_patients(client)

    print(f"Potential controls: {len(potential_controls)}")

    # =========================================================================
    # Step 3: Match controls to cases
    # =========================================================================
    print("\n--- Step 3: Matching controls to cases ---")

    matched_controls = match_controls(cases_df, potential_controls, ratio=control_ratio)
    print(f"Matched controls: {len(matched_controls)}")

    if len(matched_controls) == 0:
        print("ERROR: No controls matched!")
        return

    control_ids = matched_controls['patientId'].tolist()

    # =========================================================================
    # Step 4: Get events for cases
    # =========================================================================
    print("\n--- Step 4: Extracting events for cases ---")

    case_ids_list = cases_df['patientId'].tolist()
    batch_size_extract = 50  # Smaller batches

    case_events_list = []
    with PatientGraphClient() as client:
        for i in range(0, len(case_ids_list), batch_size_extract):
            batch_ids = case_ids_list[i:i+batch_size_extract]
            print(f"  Cases batch {i//batch_size_extract + 1}/{(len(case_ids_list)-1)//batch_size_extract + 1}...")
            try:
                df = get_patient_events_t047(client, batch_ids)
                if len(df) > 0:
                    case_events_list.append(df)
            except Exception as e:
                print(f"    Error: {e}")

    if case_events_list:
        case_events_df = pd.concat(case_events_list, ignore_index=True)
    else:
        print("ERROR: No case events extracted!")
        return

    print(f"Total case events: {len(case_events_df):,}")

    # =========================================================================
    # Step 5: Get events for controls
    # =========================================================================
    print("\n--- Step 5: Extracting events for controls ---")

    control_events_list = []
    with PatientGraphClient() as client:
        for i in range(0, len(control_ids), batch_size_extract):
            batch_ids = control_ids[i:i+batch_size_extract]
            print(f"  Controls batch {i//batch_size_extract + 1}/{(len(control_ids)-1)//batch_size_extract + 1}...")
            try:
                df = get_patient_events_t047(client, batch_ids)
                if len(df) > 0:
                    control_events_list.append(df)
            except Exception as e:
                print(f"    Error: {e}")

    if control_events_list:
        control_events_df = pd.concat(control_events_list, ignore_index=True)
    else:
        print("ERROR: No control events extracted!")
        return

    print(f"Total control events: {len(control_events_df):,}")

    # =========================================================================
    # Step 6: Apply time windows
    # =========================================================================
    print("\n--- Step 6: Applying time windows ---")

    case_events_df['eventDateTime'] = pd.to_datetime(case_events_df['eventDateTime'])
    control_events_df['eventDateTime'] = pd.to_datetime(control_events_df['eventDateTime'])

    # For cases: all events up to AF diagnosis
    case_events_df = case_events_df.merge(
        cases_df[['patientId', 'windowStart', 'cutoffDate']],
        on='patientId'
    )
    case_events_df = case_events_df[
        (case_events_df['eventDateTime'] >= case_events_df['windowStart']) &
        (case_events_df['eventDateTime'] < case_events_df['cutoffDate'])
    ].copy()

    print(f"Case events after time window: {len(case_events_df):,}")

    # For controls: 5-year window before last event
    control_last_dates = control_events_df.groupby('patientId')['eventDateTime'].max().reset_index()
    control_last_dates.columns = ['patientId', 'lastEventDate']

    matched_controls = matched_controls.merge(control_last_dates, on='patientId', how='left')
    matched_controls['cutoffDate'] = matched_controls['lastEventDate'] - pd.Timedelta(days=prediction_gap_days)
    matched_controls['windowStart'] = matched_controls['cutoffDate'] - pd.Timedelta(days=lookback_days)
    matched_controls['outcome'] = 0

    control_events_df = control_events_df.merge(
        matched_controls[['patientId', 'windowStart', 'cutoffDate']],
        on='patientId'
    )
    control_events_df = control_events_df[
        (control_events_df['eventDateTime'] >= control_events_df['windowStart']) &
        (control_events_df['eventDateTime'] < control_events_df['cutoffDate'])
    ].copy()

    print(f"Control events after time window: {len(control_events_df):,}")

    # =========================================================================
    # Step 7: Save event data for inspection
    # =========================================================================
    print("\n--- Step 7: Saving event data for inspection ---")

    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)

    case_events_df['outcome'] = 1
    control_events_df['outcome'] = 0

    all_events_df = pd.concat([case_events_df, control_events_df], ignore_index=True)

    events_output_path = output_dir / "afib_training_events.csv"
    all_events_df.to_csv(events_output_path, index=False)
    print(f"Saved all events to: {events_output_path}")
    print(f"Total events: {len(all_events_df):,}")

    # =========================================================================
    # Step 8: Build vocabulary and sequences
    # =========================================================================
    print("\n--- Step 8: Building vocabulary and sequences ---")

    labels_df = pd.concat([
        cases_df[['patientId', 'outcome']],
        matched_controls[['patientId', 'outcome']]
    ], ignore_index=True)

    patients_with_events = all_events_df['patientId'].unique()
    labels_df = labels_df[labels_df['patientId'].isin(patients_with_events)].copy()

    print(f"Patients for training: {len(labels_df)}")
    print(f"  Cases: {(labels_df['outcome'] == 1).sum()}")
    print(f"  Controls: {(labels_df['outcome'] == 0).sum()}")

    preprocessor = SequencePreprocessor(
        max_length=config.sequence.max_length,
        min_frequency=1  # Lower threshold due to fewer cases
    )

    vocab_df = all_events_df.groupby(['eventType', 'eventCode']).size().reset_index(name='frequency')
    vocab_df.columns = ['eventType', 'code', 'frequency']
    preprocessor.fit(vocab_df)
    print(f"Vocabulary size: {preprocessor.vocab_size}")

    label_map = dict(zip(labels_df['patientId'], labels_df['outcome']))

    sequences = []
    lengths = []
    labels = []
    patient_ids = []

    for patient_id, group in all_events_df.groupby('patientId'):
        if patient_id not in label_map:
            continue

        group = group.sort_values('eventDateTime')

        seq = []
        for _, row in group.iterrows():
            event_type = row['eventType']
            code = str(row['eventCode']) if pd.notna(row['eventCode']) else 'UNK'
            idx = preprocessor.vocab.encode(event_type, code)
            seq.append(idx)

        if len(seq) == 0:
            continue

        if len(seq) > config.sequence.max_length:
            seq = seq[:config.sequence.max_length]

        sequences.append(seq)
        lengths.append(len(seq))
        labels.append(label_map[patient_id])
        patient_ids.append(patient_id)

    print(f"Patients with sequences: {len(sequences)}")
    print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

    all_labels = np.array(labels)
    print(f"Label distribution: {sum(all_labels)} positive, {len(all_labels) - sum(all_labels)} negative")

    # =========================================================================
    # Step 9: Train/val/test split
    # =========================================================================
    print("\n--- Step 9: Splitting data ---")

    indices = list(range(len(sequences)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, stratify=all_labels, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=all_labels[temp_idx], random_state=42
    )

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    print(f"Train positives: {sum(all_labels[train_idx])}")
    print(f"Val positives: {sum(all_labels[val_idx])}")
    print(f"Test positives: {sum(all_labels[test_idx])}")

    max_len = config.sequence.max_length

    def pad_sequences(seqs, max_len):
        padded = np.zeros((len(seqs), max_len), dtype=np.int64)
        for i, seq in enumerate(seqs):
            padded[i, :len(seq)] = seq
        return padded

    X_train = pad_sequences([sequences[i] for i in train_idx], max_len)
    X_val = pad_sequences([sequences[i] for i in val_idx], max_len)
    X_test = pad_sequences([sequences[i] for i in test_idx], max_len)

    len_train = np.array([lengths[i] for i in train_idx])
    len_val = np.array([lengths[i] for i in val_idx])
    len_test = np.array([lengths[i] for i in test_idx])

    y_train = all_labels[train_idx]
    y_val = all_labels[val_idx]
    y_test = all_labels[test_idx]

    train_dataset = TensorDataset(
        torch.LongTensor(X_train),
        torch.LongTensor(len_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.LongTensor(X_val),
        torch.LongTensor(len_val),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.LongTensor(X_test),
        torch.LongTensor(len_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # =========================================================================
    # Step 10: Create model
    # =========================================================================
    print("\n--- Step 10: Creating model ---")

    # Use smaller model for small dataset
    model = AttentionPatientRNN(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=64,  # Smaller embedding
        hidden_size=128,   # Smaller hidden
        num_layers=1,      # Single layer
        dropout=0.3        # More dropout
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    pos_weight = min(pos_weight, 10.0)
    print(f"Positive weight: {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # =========================================================================
    # Step 11: Train
    # =========================================================================
    print("\n--- Step 11: Training ---")
    best_val_auc = 0
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, val_auprc={val_auprc:.4f}{marker}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    # =========================================================================
    # Step 12: Evaluate on test set
    # =========================================================================
    print("\n--- Step 12: Test Evaluation ---")
    model.load_state_dict(best_model_state)

    test_loss, test_auc, test_auprc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUROC: {test_auc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")

    test_preds_binary = (np.array(test_preds) > 0.5).astype(int)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds_binary, labels=[0, 1])
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds_binary,
                                target_names=['No AF', 'Atrial Fibrillation'],
                                zero_division=0))

    # =========================================================================
    # Step 13: Attention analysis
    # =========================================================================
    print("\n--- Step 13: Attention Analysis ---")
    model.eval()

    event_attention_sum = {}
    event_occurrence_count = {}

    with torch.no_grad():
        for batch in test_loader:
            seqs, lens, labs = [x.to(device) for x in batch]
            attention_weights = model.get_attention_weights(seqs, lens)

            for i in range(len(seqs)):
                seq = seqs[i].cpu().numpy()
                attn = attention_weights[i].cpu().numpy()
                length = lens[i].item()

                for j in range(length):
                    token_id = seq[j]
                    if token_id == 0:
                        continue
                    event_name = preprocessor.vocab.decode(token_id)

                    if event_name not in event_attention_sum:
                        event_attention_sum[event_name] = 0.0
                        event_occurrence_count[event_name] = 0

                    event_attention_sum[event_name] += attn[j]
                    event_occurrence_count[event_name] += 1

    avg_attention = {k: event_attention_sum[k] / event_occurrence_count[k]
                     for k in event_attention_sum}

    sorted_by_cumulative = sorted(event_attention_sum.items(), key=lambda x: -x[1])

    print(f"\nTop events by CUMULATIVE attention:")
    for event, score in sorted_by_cumulative[:15]:
        count = event_occurrence_count[event]
        avg = avg_attention[event]
        print(f"  {score:8.2f} (n={count:4d}, avg={avg:.4f}): {event}")

    # =========================================================================
    # Step 14: Save results
    # =========================================================================
    print("\n--- Step 14: Saving results ---")

    torch.save({
        'model_state_dict': best_model_state,
        'vocab': preprocessor.vocab,
        'config': config,
        'test_auc': test_auc,
        'test_auprc': test_auprc
    }, output_dir / "afib_matched_model.pt")

    results = {
        'timestamp': datetime.now().isoformat(),
        'task': 'atrial_fibrillation_prediction',
        'design': 'matched_case_control_alltime_cases',
        'prediction_gap_days': prediction_gap_days,
        'cases_lookback': 'all_time',
        'controls_lookback_years': lookback_years,
        'control_ratio': control_ratio,
        'model_type': 'attention_rnn',
        'n_cases': int((labels_df['outcome'] == 1).sum()),
        'n_controls': int((labels_df['outcome'] == 0).sum()),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'n_test': int(len(test_idx)),
        'vocab_size': int(preprocessor.vocab_size),
        'best_val_auc': float(best_val_auc),
        'test_auroc': float(test_auc),
        'test_auprc': float(test_auprc),
        'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
        'top_attention_events': [(e, float(a)) for e, a in sorted_by_cumulative[:20]]
    }

    with open(output_dir / 'afib_matched_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Model saved to: {output_dir / 'afib_matched_model.pt'}")
    print(f"Results saved to: {output_dir / 'afib_matched_results.json'}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Test AUROC: {test_auc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
