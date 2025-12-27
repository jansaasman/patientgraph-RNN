#!/usr/bin/env python
"""
Train diabetic nephropathy prediction model using PatientGraph data.

This script predicts which diabetic patients will develop kidney disease.

Prediction Task:
- Cohort: Patients with diabetes diagnosis
- Positive outcome: Develops kidney disease AFTER diabetes diagnosis
- Negative outcome: Has diabetes but no kidney disease (yet)
- Prediction gap: 365 days (predict 1 year before kidney diagnosis)

This is a clinically valuable prediction because:
1. 33% of diabetics develop kidney disease - significant population
2. Early intervention (SGLT2 inhibitors, BP control) can slow progression
3. Non-obvious patterns may exist in event sequences before diagnosis
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


def get_diabetic_patients(client: PatientGraphClient) -> pd.DataFrame:
    """
    Get all patients with diabetes/prediabetes and their earliest diagnosis date.

    Returns DataFrame with: patientId, diabetesDate, diabetesType
    """
    query = f"""{PREFIXES}
    SELECT ?patientId
           (MIN(?diagDate) AS ?diabetesDate)
           (SAMPLE(?label) AS ?diabetesType)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?condition .

        ?condition ns28:code ?code ;
                   ns28:startDateTime ?diagDate .

        ?code skos:prefLabel ?label .

        # Match diabetes-related conditions
        FILTER(
            CONTAINS(LCASE(?label), "diabetes") ||
            CONTAINS(LCASE(?label), "diabetic")
        )
    }}
    GROUP BY ?patientId
    ORDER BY ?patientId
    """
    return client.query(query)


def get_kidney_disease_dates(client: PatientGraphClient, patient_ids: list) -> pd.DataFrame:
    """
    Get kidney disease diagnosis dates for patients.

    Returns DataFrame with: patientId, kidneyDate, kidneyType
    """
    patient_values = " ".join([f'"{pid}"' for pid in patient_ids])

    query = f"""{PREFIXES}
    SELECT ?patientId
           (MIN(?diagDate) AS ?kidneyDate)
           (SAMPLE(?label) AS ?kidneyType)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId ;
                 ns28:patientCondition ?condition .

        VALUES ?patientId {{ {patient_values} }}

        ?condition ns28:code ?code ;
                   ns28:startDateTime ?diagDate .

        ?code skos:prefLabel ?label .

        # Match kidney disease conditions
        FILTER(
            CONTAINS(LCASE(?label), "kidney") ||
            CONTAINS(LCASE(?label), "renal") ||
            CONTAINS(LCASE(?label), "nephropathy") ||
            CONTAINS(LCASE(?label), "nephritis") ||
            CONTAINS(LCASE(?label), "albuminuria")
        )
    }}
    GROUP BY ?patientId
    """
    return client.query(query)


def get_patient_events(client: PatientGraphClient, patient_ids: list) -> pd.DataFrame:
    """
    Get all clinical events for patients.
    """
    patient_values = " ".join([f'"{pid}"' for pid in patient_ids])

    query = f"""{PREFIXES}
    SELECT DISTINCT
        ?patientId
        ?eventDateTime
        ?eventType
        ?eventCode
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId .
        VALUES ?patientId {{ {patient_values} }}

        {{
            ?patient ns28:patientCondition ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {{
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
            }}
            BIND("CONDITION" AS ?eventType)
        }}
        UNION
        {{
            ?patient ns28:patientMedication ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {{
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
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
            }}
            BIND("OBSERVATION" AS ?eventType)
        }}
    }}
    ORDER BY ?patientId ?eventDateTime
    """
    return client.query(query)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
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
    """Evaluate model."""
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
    print("DIABETIC NEPHROPATHY PREDICTION TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Parameters
    prediction_gap_days = 365  # Predict kidney disease 1 year ahead
    batch_size = 32
    epochs = 30
    learning_rate = 0.001

    print(f"\nPrediction task: Predict kidney disease {prediction_gap_days} days in advance")
    print("Cohort: Diabetic patients")

    # =========================================================================
    # Step 1: Get diabetic patients
    # =========================================================================
    print("\n--- Step 1: Extracting diabetic patients ---")

    with PatientGraphClient() as client:
        diabetes_df = get_diabetic_patients(client)

    print(f"Total diabetic patients: {len(diabetes_df)}")
    diabetes_df['diabetesDate'] = pd.to_datetime(diabetes_df['diabetesDate'])

    # =========================================================================
    # Step 2: Get kidney disease outcomes
    # =========================================================================
    print("\n--- Step 2: Extracting kidney disease outcomes ---")

    all_patient_ids = diabetes_df['patientId'].tolist()

    # Query in batches
    batch_size_extract = 500
    kidney_dfs = []

    with PatientGraphClient() as client:
        for i in range(0, len(all_patient_ids), batch_size_extract):
            batch_ids = all_patient_ids[i:i+batch_size_extract]
            print(f"  Batch {i//batch_size_extract + 1}/{(len(all_patient_ids)-1)//batch_size_extract + 1}...")

            try:
                df = get_kidney_disease_dates(client, batch_ids)
                if len(df) > 0:
                    kidney_dfs.append(df)
            except Exception as e:
                print(f"    Error: {e}")
                continue

    if kidney_dfs:
        kidney_df = pd.concat(kidney_dfs, ignore_index=True)
        kidney_df['kidneyDate'] = pd.to_datetime(kidney_df['kidneyDate'])
    else:
        kidney_df = pd.DataFrame(columns=['patientId', 'kidneyDate', 'kidneyType'])

    print(f"Patients with kidney disease: {len(kidney_df)}")

    # =========================================================================
    # Step 3: Create labels
    # =========================================================================
    print("\n--- Step 3: Creating outcome labels ---")

    # Merge diabetes and kidney data
    labels_df = diabetes_df.merge(kidney_df, on='patientId', how='left')

    # Positive: kidney disease AFTER diabetes
    labels_df['has_kidney'] = labels_df['kidneyDate'].notna()
    labels_df['kidney_after_diabetes'] = (
        labels_df['has_kidney'] &
        (labels_df['kidneyDate'] >= labels_df['diabetesDate'])
    )

    # Create binary label
    labels_df['outcome'] = labels_df['kidney_after_diabetes'].astype(int)

    positive_count = labels_df['outcome'].sum()
    negative_count = len(labels_df) - positive_count

    print(f"Positive cases (kidney disease after diabetes): {positive_count}")
    print(f"Negative cases (no kidney disease): {negative_count}")
    print(f"Positive rate: {positive_count/len(labels_df)*100:.1f}%")

    # Calculate cutoff date for each patient
    # Positive: kidney date - prediction_gap_days
    # Negative: use last event date (will be determined later)
    labels_df['cutoffDate'] = labels_df.apply(
        lambda row: row['kidneyDate'] - pd.Timedelta(days=prediction_gap_days)
        if row['outcome'] == 1 else pd.NaT,
        axis=1
    )

    # =========================================================================
    # Step 4: Extract events
    # =========================================================================
    print("\n--- Step 4: Extracting patient events ---")

    all_events = []

    with PatientGraphClient() as client:
        for i in range(0, len(all_patient_ids), batch_size_extract):
            batch_ids = all_patient_ids[i:i+batch_size_extract]
            print(f"  Extracting events batch {i//batch_size_extract + 1}/{(len(all_patient_ids)-1)//batch_size_extract + 1}...")

            try:
                events_df = get_patient_events(client, batch_ids)
                all_events.append(events_df)
            except Exception as e:
                print(f"    Error: {e}")
                continue

    if not all_events:
        print("No events extracted!")
        return

    events_df = pd.concat(all_events, ignore_index=True)
    events_df['eventDateTime'] = pd.to_datetime(events_df['eventDateTime'])
    print(f"Total events extracted: {len(events_df):,}")

    # Get last event date for negative cases
    last_dates = events_df.groupby('patientId')['eventDateTime'].max().reset_index()
    last_dates.columns = ['patientId', 'lastEventDate']
    labels_df = labels_df.merge(last_dates, on='patientId', how='left')

    # Set cutoff for negative cases
    labels_df.loc[labels_df['cutoffDate'].isna(), 'cutoffDate'] = (
        labels_df.loc[labels_df['cutoffDate'].isna(), 'lastEventDate']
        - pd.Timedelta(days=prediction_gap_days)
    )

    # =========================================================================
    # Step 5: Filter events by cutoff
    # =========================================================================
    print(f"\n--- Step 5: Applying {prediction_gap_days}-day prediction gap ---")

    # Merge cutoff dates into events
    events_df = events_df.merge(
        labels_df[['patientId', 'cutoffDate', 'outcome']],
        on='patientId'
    )

    # Filter events before cutoff (and after diabetes diagnosis)
    events_df = events_df.merge(
        labels_df[['patientId', 'diabetesDate']],
        on='patientId'
    )

    # Keep only events between diabetes diagnosis and cutoff
    events_df = events_df[
        (events_df['eventDateTime'] >= events_df['diabetesDate']) &
        (events_df['eventDateTime'] <= events_df['cutoffDate'])
    ].copy()

    # Drop helper columns
    events_df = events_df[['patientId', 'eventDateTime', 'eventType', 'eventCode', 'outcome']]

    print(f"Events after filtering: {len(events_df):,}")

    # Filter to patients with events
    patients_with_events = events_df['patientId'].unique()
    print(f"Patients with events: {len(patients_with_events)}")

    labels_df = labels_df[labels_df['patientId'].isin(patients_with_events)].copy()
    print(f"Patients for training: {len(labels_df)}")

    positive_count = labels_df['outcome'].sum()
    print(f"  Positive: {positive_count} ({positive_count/len(labels_df)*100:.1f}%)")
    print(f"  Negative: {len(labels_df) - positive_count}")

    # =========================================================================
    # Step 6: Build vocabulary and sequences
    # =========================================================================
    print("\n--- Step 6: Building vocabulary and sequences ---")

    preprocessor = SequencePreprocessor(
        max_length=config.sequence.max_length,
        min_frequency=2
    )

    # Create vocab from events
    vocab_df = events_df.groupby(['eventType', 'eventCode']).size().reset_index(name='frequency')
    vocab_df.columns = ['eventType', 'code', 'frequency']
    preprocessor.fit(vocab_df)
    print(f"Vocabulary size: {preprocessor.vocab_size}")

    # Create label mapping
    label_map = dict(zip(labels_df['patientId'], labels_df['outcome']))

    # Build sequences
    sequences = []
    lengths = []
    labels = []
    patient_ids = []

    for patient_id, group in events_df.groupby('patientId'):
        if patient_id not in label_map:
            continue

        # Sort by time
        group = group.sort_values('eventDateTime')

        # Encode events
        seq = []
        for _, row in group.iterrows():
            event_type = row['eventType']
            code = str(row['eventCode']) if pd.notna(row['eventCode']) else 'UNK'
            idx = preprocessor.vocab.encode(event_type, code)
            seq.append(idx)

        if len(seq) == 0:
            continue

        # Truncate if needed
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
    # Step 7: Train/val/test split
    # =========================================================================
    print("\n--- Step 7: Splitting data ---")

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

    # Pad sequences
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

    # Create datasets
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
    # Step 8: Create model
    # =========================================================================
    print("\n--- Step 8: Creating model ---")

    model = AttentionPatientRNN(
        vocab_size=preprocessor.vocab_size,
        embedding_dim=config.model.embedding_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Class weight for imbalanced data
    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    pos_weight = min(pos_weight, 5.0)  # Cap at 5 (data is fairly balanced)
    print(f"Positive weight: {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # =========================================================================
    # Step 9: Train
    # =========================================================================
    print("\n--- Step 9: Training ---")
    best_val_auc = 0
    best_model_state = None
    patience = 7
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

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}, val_auprc={val_auprc:.4f}{marker}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    # =========================================================================
    # Step 10: Evaluate on test set
    # =========================================================================
    print("\n--- Step 10: Test Evaluation ---")
    model.load_state_dict(best_model_state)

    test_loss, test_auc, test_auprc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUROC: {test_auc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")

    # Binary predictions at 0.5 threshold
    test_preds_binary = (np.array(test_preds) > 0.5).astype(int)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds_binary, labels=[0, 1])
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds_binary,
                                target_names=['No Kidney Disease', 'Kidney Disease'],
                                zero_division=0))

    # =========================================================================
    # Step 11: Attention analysis (ALL test samples)
    # =========================================================================
    print("\n--- Step 11: Attention Analysis (all test samples) ---")
    model.eval()

    # Collect attention across ALL test batches
    event_attention_sum = {}    # Total attention per event
    event_occurrence_count = {} # How many times each event appeared

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

    # Compute average attention AND cumulative importance
    avg_attention = {k: event_attention_sum[k] / event_occurrence_count[k]
                     for k in event_attention_sum}

    # Sort by cumulative attention (total importance, not average)
    sorted_by_cumulative = sorted(event_attention_sum.items(), key=lambda x: -x[1])
    sorted_by_average = sorted(avg_attention.items(), key=lambda x: -x[1])

    print(f"\nAnalyzed {len(test_loader) * batch_size} test samples")
    print(f"Events in vocabulary: {len(event_attention_sum)}")

    print("\nTop events by CUMULATIVE attention (frequency-weighted importance):")
    for event, score in sorted_by_cumulative[:15]:
        count = event_occurrence_count[event]
        avg = avg_attention[event]
        print(f"  {score:8.2f} (n={count:4d}, avg={avg:.4f}): {event}")

    print("\nTop events by AVERAGE attention (per-occurrence, may be noisy for rare events):")
    for event, score in sorted_by_average[:15]:
        count = event_occurrence_count[event]
        print(f"  {score:.4f} (n={count:4d}): {event}")

    # Use cumulative for the saved results (more robust)
    sorted_events = sorted_by_cumulative

    # =========================================================================
    # Step 12: Save results
    # =========================================================================
    print("\n--- Step 12: Saving results ---")

    model_path = Path("models")
    model_path.mkdir(exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': best_model_state,
        'vocab': preprocessor.vocab,
        'config': config,
        'test_auc': test_auc,
        'test_auprc': test_auprc
    }, model_path / "nephropathy_model.pt")

    # Save results JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'task': 'diabetic_nephropathy_prediction',
        'prediction_gap_days': prediction_gap_days,
        'model_type': 'attention_rnn',
        'n_diabetic_patients': int(len(diabetes_df)),
        'n_with_kidney_disease': int(positive_count),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'n_test': int(len(test_idx)),
        'vocab_size': int(preprocessor.vocab_size),
        'best_val_auc': float(best_val_auc),
        'test_auroc': float(test_auc),
        'test_auprc': float(test_auprc),
        'confusion_matrix': [[int(x) for x in row] for row in cm.tolist()],
        'top_attention_events': [(e, float(a)) for e, a in sorted_events[:20]]
    }

    with open(model_path / 'nephropathy_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Model saved to: {model_path / 'nephropathy_model.pt'}")
    print(f"Results saved to: {model_path / 'nephropathy_results.json'}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Test AUROC: {test_auc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
