#!/usr/bin/env python
"""
Train hospital readmission prediction model on PatientGraph data.

This script:
1. Extracts patients with inpatient hospitalizations
2. Generates 30-day readmission labels
3. Extracts event sequences for these patients
4. Trains an LSTM model to predict readmission
5. Evaluates performance with AUROC, AUPRC, etc.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.agraph_client import PatientGraphClient
from src.data_extractor import DataExtractor
from src.sequence_preprocessor import SequencePreprocessor, PatientSequenceDataset
from src.models import create_model


def get_readmission_labels(client: PatientGraphClient, days: int = 30) -> pd.DataFrame:
    """
    Generate 30-day readmission labels from PatientGraph.

    For each patient with an inpatient stay, we check if they have
    another inpatient stay within `days` of discharge.

    Returns DataFrame with: patientId, indexDate, readmitted (0/1)
    """
    # Use a simpler approach: find patients with multiple inpatient stays
    # and check if any two are within 30 days of each other
    query = f'''
PREFIX ns28: <http://patientgraph.ai/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?patientId ?dischargeDate ?nextAdmitDate
WHERE {{
    # Get patients with inpatient encounters
    ?patient a ns28:Patient ;
             rdfs:label ?patientId ;
             ns28:patientEncounter ?enc1 .

    ?enc1 ns28:encounterclass "inpatient" ;
          ns28:endDateTime ?dischargeDate ;
          ns28:startDateTime ?admitDate1 .

    # Look for subsequent inpatient admission
    OPTIONAL {{
        ?patient ns28:patientEncounter ?enc2 .
        ?enc2 ns28:encounterclass "inpatient" ;
              ns28:startDateTime ?nextAdmitDate .

        # Must be different encounter and after discharge
        FILTER(?enc1 != ?enc2)
        FILTER(?nextAdmitDate > ?dischargeDate)
    }}
}}
ORDER BY ?patientId ?dischargeDate
'''

    df = client.query(query)

    if len(df) == 0:
        return pd.DataFrame(columns=['patientId', 'dischargeDate', 'readmitted'])

    # Convert dates
    df['dischargeDate'] = pd.to_datetime(df['dischargeDate'], errors='coerce')
    df['nextAdmitDate'] = pd.to_datetime(df['nextAdmitDate'], errors='coerce')

    # Calculate days until next admission
    df['daysToReadmit'] = (df['nextAdmitDate'] - df['dischargeDate']).dt.days

    # Readmitted if next admission within specified days
    df['readmitted'] = ((df['daysToReadmit'] > 0) & (df['daysToReadmit'] <= days)).astype(int)

    # Take the earliest discharge per patient (for a single prediction point)
    df_unique = df.sort_values('dischargeDate').groupby('patientId').first().reset_index()

    return df_unique[['patientId', 'dischargeDate', 'readmitted']]


def get_patient_events_before_discharge(
    client: PatientGraphClient,
    patient_ids: list,
    max_events: int = 200
) -> pd.DataFrame:
    """
    Get clinical events for patients, limited to events before their first discharge.
    """
    from src.query_templates import event_sequence_query

    all_events = []
    batch_size = 100  # Larger batches for speed

    for i in range(0, len(patient_ids), batch_size):
        batch_ids = patient_ids[i:i+batch_size]
        if i % 500 == 0:
            print(f"  Fetching events for patients {i+1}-{min(i+batch_size, len(patient_ids))} of {len(patient_ids)}...")

        query = event_sequence_query(patient_ids=batch_ids, limit=max_events * len(batch_ids))
        df = client.query(query)
        if len(df) > 0:
            all_events.append(df)

    if not all_events:
        return pd.DataFrame()

    events_df = pd.concat(all_events, ignore_index=True)
    events_df['eventDateTime'] = pd.to_datetime(events_df['eventDateTime'], errors='coerce')

    return events_df


def main():
    print("=" * 70)
    print("HOSPITAL READMISSION PREDICTION - TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = get_config()

    # =========================================================================
    # 1. EXTRACT READMISSION LABELS
    # =========================================================================
    print("\n[1/6] Extracting 30-day readmission labels...")

    with PatientGraphClient() as client:
        labels_df = get_readmission_labels(client, days=30)

    total_patients = len(labels_df)
    readmitted = labels_df['readmitted'].sum()
    readmit_rate = readmitted / total_patients * 100

    print(f"       Total patients with inpatient stays: {total_patients}")
    print(f"       30-day readmissions: {readmitted} ({readmit_rate:.1f}%)")

    # =========================================================================
    # 2. USE ALL PATIENTS
    # =========================================================================
    print("\n[2/6] Preparing all patients for training...")

    readmitted_patients = labels_df[labels_df['readmitted'] == 1]['patientId'].tolist()
    non_readmitted_patients = labels_df[labels_df['readmitted'] == 0]['patientId'].tolist()

    print(f"       Total patients: {len(labels_df)}")
    print(f"       - Readmitted (positive): {len(readmitted_patients)}")
    print(f"       - Not readmitted (negative): {len(non_readmitted_patients)}")

    # =========================================================================
    # 3. EXTRACT EVENTS
    # =========================================================================
    print("\n[3/6] Extracting clinical events (this may take a few minutes)...")

    all_patient_ids = labels_df['patientId'].tolist()

    with PatientGraphClient() as client:
        events_df = get_patient_events_before_discharge(
            client,
            all_patient_ids,
            max_events=150  # Limit per patient to manage memory
        )

    print(f"       Total events: {len(events_df):,}")
    print(f"       Event types: {events_df['eventType'].value_counts().to_dict()}")

    # =========================================================================
    # 4. PREPROCESS SEQUENCES
    # =========================================================================
    print("\n[4/6] Preprocessing sequences...")

    preprocessor = SequencePreprocessor(
        config=config,
        max_length=150,
        min_frequency=3
    )
    preprocessor.fit_from_events(events_df)
    print(f"       Vocabulary size: {preprocessor.vocab_size}")

    # Rename column for compatibility
    labels_for_transform = labels_df.rename(columns={'readmitted': 'outcomeOccurred'})

    # Transform
    data = preprocessor.transform(events_df, labels_for_transform)
    print(f"       Patients with sequences: {len(data['patient_ids'])}")
    print(f"       Avg sequence length: {np.mean(data['lengths']):.1f}")

    # Convert to tensors
    tensors = preprocessor.to_tensors(data)

    # =========================================================================
    # 5. CREATE DATASET AND SPLIT
    # =========================================================================
    print("\n[5/6] Creating train/val/test splits...")

    dataset = PatientSequenceDataset.from_preprocessor_output(tensors)

    # Use stratified split to ensure positives in each split
    from sklearn.model_selection import train_test_split

    all_labels = tensors['labels'].numpy()
    all_indices = np.arange(len(dataset))

    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=all_labels
    )

    # Second split: temp into val and test (50-50 of the 30% = 15% each)
    temp_labels = all_labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    n_train, n_val, n_test = len(train_idx), len(val_idx), len(test_idx)

    # Calculate class weights for imbalanced data (cap at 20 to prevent instability)
    train_labels = [dataset[i]['label'].item() for i in train_ds.indices]
    raw_pos_weight = (len(train_labels) - sum(train_labels)) / max(sum(train_labels), 1)
    pos_weight = min(raw_pos_weight, 20.0)  # Cap to prevent training instability

    print(f"       Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"       Train positives: {sum(train_labels):.0f}")
    print(f"       Positive weight: {pos_weight:.2f} (raw: {raw_pos_weight:.2f})")

    # Create dataloaders with larger batch size
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # =========================================================================
    # 6. TRAIN MODEL
    # =========================================================================
    print("\n[6/6] Training Attention LSTM model...")

    model = create_model(
        'attention',  # Use attention model for interpretability
        preprocessor.vocab_size,
        config=config,
        pad_idx=preprocessor.vocab.pad_id
    )

    # Weighted loss for class imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Lower LR for stability
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    n_epochs = 30  # More epochs for larger dataset
    best_val_auc = 0
    patience = 5
    patience_counter = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        train_steps = 0

        for batch in train_loader:
            sequences = batch['sequence']
            lengths = batch['length']
            labels = batch['label']

            optimizer.zero_grad()
            outputs = model(sequences, lengths).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence']
                lengths = batch['length']
                labels = batch['label']

                outputs = model(sequences, lengths).squeeze(-1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_steps += 1

                probs = torch.sigmoid(outputs)
                val_preds.extend(probs.numpy())
                val_labels.extend(labels.numpy())

        avg_val_loss = val_loss / val_steps

        # Calculate AUC
        if len(set(val_labels)) > 1:
            val_auc = roc_auc_score(val_labels, val_preds)
        else:
            val_auc = 0.5

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(f"       Epoch {epoch+1:2d}/{n_epochs}: "
              f"Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, "
              f"Val AUC={val_auc:.4f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'models/best_readmission_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"       Early stopping at epoch {epoch+1}")
                break

    # =========================================================================
    # EVALUATE ON TEST SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)

    # Load best model
    model.load_state_dict(torch.load('models/best_readmission_model.pt'))
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence']
            lengths = batch['length']
            labels = batch['label']

            outputs = model(sequences, lengths).squeeze(-1)
            probs = torch.sigmoid(outputs)

            test_preds.extend(probs.numpy())
            test_labels.extend(labels.numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)

    # Metrics
    if len(set(test_labels)) > 1:
        auroc = roc_auc_score(test_labels, test_preds)
        auprc = average_precision_score(test_labels, test_preds)
    else:
        auroc = 0.5
        auprc = sum(test_labels) / len(test_labels)

    # Binary predictions at 0.5 threshold
    binary_preds = (test_preds > 0.5).astype(int)

    print(f"\nTest Set Results ({len(test_labels)} patients):")
    print(f"  AUROC:  {auroc:.4f}")
    print(f"  AUPRC:  {auprc:.4f}")
    print(f"\nConfusion Matrix (threshold=0.5):")
    cm = confusion_matrix(test_labels, binary_preds, labels=[0, 1])
    print(f"  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")

    print(f"\nClassification Report:")
    print(classification_report(test_labels, binary_preds,
                                labels=[0, 1],
                                target_names=['No Readmit', 'Readmit'],
                                zero_division=0))

    # =========================================================================
    # ATTENTION WEIGHT ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("ATTENTION WEIGHT ANALYSIS")
    print("=" * 70)

    # Get attention weights for test set
    all_attention_weights = []
    all_event_codes = []
    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence']
            lengths = batch['length']
            labels = batch['label']

            # Get predictions and attention weights
            outputs = model(sequences, lengths).squeeze(-1)
            probs = torch.sigmoid(outputs)
            attn_weights = model.get_attention_weights(sequences, lengths)

            for i in range(len(sequences)):
                seq_len = lengths[i].item()
                seq = sequences[i, :seq_len].numpy()
                attn = attn_weights[i, :seq_len].numpy()

                all_attention_weights.append(attn)
                all_event_codes.append(seq)
                all_predictions.append(probs[i].item())
                all_true_labels.append(labels[i].item())

    # Aggregate attention by event type
    print("\nTop attended events (aggregated across all test patients):")

    event_attention = {}
    for seq, attn in zip(all_event_codes, all_attention_weights):
        for code_id, weight in zip(seq, attn):
            token = preprocessor.vocab.decode(code_id)
            if token not in ['<PAD>', '<UNK>', '<START>', '<END>']:
                if token not in event_attention:
                    event_attention[token] = []
                event_attention[token].append(weight)

    # Calculate mean attention per event type
    mean_attention = {k: np.mean(v) for k, v in event_attention.items()}
    top_events = sorted(mean_attention.items(), key=lambda x: -x[1])[:15]

    print(f"\n{'Event Code':<50} {'Mean Attention':>15}")
    print("-" * 67)
    for event, attn in top_events:
        print(f"{event:<50} {attn:>15.4f}")

    # Show example for a readmitted patient
    readmitted_indices = [i for i, l in enumerate(all_true_labels) if l == 1]
    if readmitted_indices:
        print(f"\n\nExample: Attention weights for a readmitted patient:")
        idx = readmitted_indices[0]
        seq = all_event_codes[idx]
        attn = all_attention_weights[idx]

        # Get top 10 attended events for this patient
        event_attn_pairs = [(preprocessor.vocab.decode(c), a) for c, a in zip(seq, attn)]
        event_attn_pairs = [(e, a) for e, a in event_attn_pairs if e not in ['<PAD>', '<UNK>']]
        top_patient_events = sorted(event_attn_pairs, key=lambda x: -x[1])[:10]

        print(f"Prediction probability: {all_predictions[idx]:.3f}")
        print(f"\n{'Event':<50} {'Attention':>10}")
        print("-" * 62)
        for event, attn_val in top_patient_events:
            print(f"{event:<50} {attn_val:>10.4f}")

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'attention',
        'n_patients': len(labels_df),
        'n_train': len(train_ds),
        'n_val': len(val_ds),
        'n_test': len(test_ds),
        'vocab_size': preprocessor.vocab_size,
        'best_val_auc': best_val_auc,
        'test_auroc': float(auroc),
        'test_auprc': float(auprc),
        'confusion_matrix': cm.tolist(),
        'top_attention_events': [(e, float(a)) for e, a in top_events]
    }

    with open('models/readmission_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save preprocessor for later use
    preprocessor.save(Path('models/preprocessor'))

    print("\n" + "=" * 70)
    print(f"Training complete!")
    print(f"  Model saved to: models/best_readmission_model.pt")
    print(f"  Preprocessor saved to: models/preprocessor/")
    print(f"  Results saved to: models/readmission_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
