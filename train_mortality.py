#!/usr/bin/env python
"""
Train mortality prediction model using PatientGraph data.

This script:
1. Extracts all patients and their mortality status
2. Uses events from first year of patient history for prediction
3. Trains an Attention RNN to predict mortality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.agraph_client import PatientGraphClient
from src.query_templates import PREFIXES
from src.sequence_preprocessor import SequencePreprocessor
from src.models import AttentionPatientRNN
from src.config import get_config


def extract_mortality_labels(client: PatientGraphClient, limit: int = None) -> pd.DataFrame:
    """Extract patients with mortality labels and last event date."""

    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""{PREFIXES}
    SELECT DISTINCT
        ?patientId
        ?birthdate
        ?deathdate
        (IF(BOUND(?deathdate), 1, 0) AS ?deceased)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId .
        OPTIONAL {{ ?patient ns28:birthdate ?birthdate }}
        OPTIONAL {{ ?patient ns28:deathdate ?deathdate }}
    }}
    ORDER BY ?patientId
    {limit_clause}
    """

    return client.query(query)


def get_patient_last_events(client: PatientGraphClient, patient_ids: list) -> pd.DataFrame:
    """Get the last event date for each patient."""

    patient_values = " ".join([f'"{pid}"' for pid in patient_ids])

    query = f"""{PREFIXES}
    SELECT ?patientId (MAX(?eventDate) AS ?lastEventDate)
    WHERE {{
        ?patient a ns28:Patient ;
                 rdfs:label ?patientId .
        VALUES ?patientId {{ {patient_values} }}

        {{
            ?patient ns28:patientCondition ?event .
            ?event ns28:startDateTime ?eventDate .
        }}
        UNION
        {{
            ?patient ns28:patientMedication ?event .
            ?event ns28:startDateTime ?eventDate .
        }}
        UNION
        {{
            ?patient ns28:patientProcedure ?event .
            ?event ns28:startDateTime ?eventDate .
        }}
        UNION
        {{
            ?patient ns28:patientObservation ?event .
            ?event ns28:startDateTime ?eventDate .
        }}
    }}
    GROUP BY ?patientId
    """

    return client.query(query)


def extract_patient_events(client: PatientGraphClient, patient_ids: list) -> pd.DataFrame:
    """
    Extract all events for patients.
    Filtering by observation window is done in Python.
    """

    # Build patient filter
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

    # Calculate AUC
    if len(set(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_preds)
    else:
        auc = 0.5

    return avg_loss, auc, all_preds, all_labels


def main():
    print("=" * 60)
    print("Mortality Prediction Training")
    print("=" * 60)

    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Parameters
    prediction_gap_days = 90  # Predict mortality 90 days ahead
    batch_size = 32
    epochs = 30
    learning_rate = 0.001

    print(f"\nPrediction gap: {prediction_gap_days} days (predict 90-day mortality)")

    # Step 1: Extract mortality labels
    print("\n--- Step 1: Extracting mortality labels ---")
    with PatientGraphClient() as client:
        labels_df = extract_mortality_labels(client)

    print(f"Total patients: {len(labels_df)}")
    deceased_count = labels_df['deceased'].sum()
    print(f"Deceased: {deceased_count} ({deceased_count/len(labels_df)*100:.1f}%)")
    print(f"Alive: {len(labels_df) - deceased_count}")

    # Step 2: Extract events for all patients (in batches)
    print("\n--- Step 2: Extracting patient events ---")
    all_patient_ids = labels_df['patientId'].tolist()

    batch_size_extract = 500
    all_events = []

    with PatientGraphClient() as client:
        for i in range(0, len(all_patient_ids), batch_size_extract):
            batch_ids = all_patient_ids[i:i+batch_size_extract]
            print(f"  Extracting batch {i//batch_size_extract + 1}/{(len(all_patient_ids)-1)//batch_size_extract + 1}...")

            try:
                events_df = extract_patient_events(client, batch_ids)
                all_events.append(events_df)
            except Exception as e:
                print(f"    Error: {e}")
                continue

    if not all_events:
        print("No events extracted!")
        return

    events_df = pd.concat(all_events, ignore_index=True)
    print(f"Total events extracted: {len(events_df):,}")

    # Filter with prediction gap
    print(f"\n--- Applying {prediction_gap_days}-day prediction gap ---")
    events_df['eventDateTime'] = pd.to_datetime(events_df['eventDateTime'])
    labels_df['deathdate'] = pd.to_datetime(labels_df['deathdate'])

    # Create cutoff date for each patient:
    # - Deceased: deathdate - 90 days (predict death 90 days ahead)
    # - Living: last event date - 90 days (same temporal setup)

    # Get last event date per patient
    last_dates = events_df.groupby('patientId')['eventDateTime'].max().reset_index()
    last_dates.columns = ['patientId', 'lastEventDate']

    # Merge with labels
    labels_df = labels_df.merge(last_dates, on='patientId', how='left')

    # Calculate cutoff: for deceased use deathdate, for living use last event
    labels_df['cutoffDate'] = labels_df.apply(
        lambda row: row['deathdate'] - pd.Timedelta(days=prediction_gap_days)
        if pd.notna(row['deathdate'])
        else row['lastEventDate'] - pd.Timedelta(days=prediction_gap_days),
        axis=1
    )

    # Merge cutoff into events
    events_df = events_df.merge(
        labels_df[['patientId', 'cutoffDate']],
        on='patientId'
    )

    # Filter events before cutoff
    events_df = events_df[events_df['eventDateTime'] <= events_df['cutoffDate']].copy()
    events_df = events_df.drop(columns=['cutoffDate'])

    print(f"Events after filtering: {len(events_df):,}")

    # Filter to patients with events
    patients_with_events = events_df['patientId'].unique()
    print(f"Patients with events: {len(patients_with_events)}")

    # Filter labels to patients with events
    labels_df = labels_df[labels_df['patientId'].isin(patients_with_events)].copy()
    print(f"Patients for training: {len(labels_df)}")
    deceased_count = labels_df['deceased'].sum()
    print(f"  Deceased: {deceased_count} ({deceased_count/len(labels_df)*100:.1f}%)")

    # Step 3: Build vocabulary
    print("\n--- Step 3: Building vocabulary ---")
    preprocessor = SequencePreprocessor(
        max_length=config.sequence.max_length,
        min_frequency=2
    )

    # Create vocab from events
    vocab_df = events_df.groupby(['eventType', 'eventCode']).size().reset_index(name='frequency')
    vocab_df.columns = ['eventType', 'code', 'frequency']
    preprocessor.fit(vocab_df)
    print(f"Vocabulary size: {preprocessor.vocab_size}")

    # Step 4: Prepare sequences
    print("\n--- Step 4: Preparing sequences ---")

    # Rename column for preprocessor
    events_df = events_df.rename(columns={'eventCode': 'eventCode'})

    # Create label mapping
    label_map = dict(zip(labels_df['patientId'], labels_df['deceased']))

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

    # Check class balance
    all_labels = np.array(labels)
    print(f"Label distribution: {sum(all_labels)} positive, {len(all_labels) - sum(all_labels)} negative")

    # Step 5: Train/val/test split (stratified)
    print("\n--- Step 5: Splitting data ---")

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

    # Step 6: Create model
    print("\n--- Step 6: Creating model ---")
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
    pos_weight = min(pos_weight, 10.0)  # Cap at 10
    print(f"Positive weight: {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Step 7: Train
    print("\n--- Step 7: Training ---")
    best_val_auc = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_auc={val_auc:.4f}{marker}")

    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    # Step 8: Evaluate on test set
    print("\n--- Step 8: Test Evaluation ---")
    model.load_state_dict(best_model_state)

    test_loss, test_auc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test AUROC: {test_auc:.4f}")

    # Binary predictions
    test_preds_binary = (np.array(test_preds) > 0.5).astype(int)

    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds_binary, labels=[0, 1])
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds_binary,
                                target_names=['Alive', 'Deceased'],
                                zero_division=0))

    # Step 9: Attention analysis
    print("\n--- Step 9: Attention Analysis ---")
    model.eval()

    # Get attention for a few test examples
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        seqs, lens, labs = [x.to(device) for x in sample_batch]
        attention_weights = model.get_attention_weights(seqs, lens)

    # Find high-attention events
    print("\nTop attended events (averaged across samples):")

    # Collect event attention scores
    event_attention = {}
    for i in range(min(len(seqs), 10)):
        seq = seqs[i].cpu().numpy()
        attn = attention_weights[i].cpu().numpy()
        length = lens[i].item()

        for j in range(length):
            token_id = seq[j]
            if token_id == 0:  # Skip padding
                continue
            event_name = preprocessor.vocab.decode(token_id)
            if event_name not in event_attention:
                event_attention[event_name] = []
            event_attention[event_name].append(attn[j])

    # Average and sort
    avg_attention = {k: np.mean(v) for k, v in event_attention.items()}
    sorted_events = sorted(avg_attention.items(), key=lambda x: -x[1])

    for event, score in sorted_events[:15]:
        print(f"  {score:.4f}: {event}")

    # Save model
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    torch.save({
        'model_state_dict': best_model_state,
        'vocab': preprocessor.vocab,
        'config': config,
        'test_auc': test_auc
    }, model_path / "mortality_model.pt")
    print(f"\nModel saved to {model_path / 'mortality_model.pt'}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
