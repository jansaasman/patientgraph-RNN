"""
Generic disease prediction trainer.

Trains matched case-control disease prediction models using configuration files.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

from .disease_config import DiseaseConfig
from .agraph_client import PatientGraphClient
from .query_templates import PREFIXES
from .sequence_preprocessor import SequencePreprocessor
from .models import AttentionPatientRNN
from .config import get_config


class DiseasePredictor:
    """Generic disease prediction trainer with matched case-control design."""

    def __init__(self, config: DiseaseConfig):
        self.config = config
        self.app_config = get_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Will be set after loading data
        self.n_cases = 0
        self.control_ratio = config.control_ratio
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate

        # Output directory
        self.output_dir = Path("models")
        self.output_dir.mkdir(exist_ok=True)

    def _auto_tune_hyperparameters(self):
        """Auto-adjust hyperparameters based on dataset size."""
        if self.n_cases < 100:
            # Very small dataset
            if self.control_ratio is None:
                self.control_ratio = 4
            if self.batch_size is None:
                self.batch_size = 16
            if self.hidden_size is None:
                self.hidden_size = 128
            if self.num_layers is None:
                self.num_layers = 1
            if self.epochs is None:
                self.epochs = 50
            if self.learning_rate is None:
                self.learning_rate = 0.0005
        elif self.n_cases < 300:
            # Small dataset
            if self.control_ratio is None:
                self.control_ratio = 3
            if self.batch_size is None:
                self.batch_size = 32
            if self.hidden_size is None:
                self.hidden_size = 256
            if self.num_layers is None:
                self.num_layers = 2
            if self.epochs is None:
                self.epochs = 40
            if self.learning_rate is None:
                self.learning_rate = 0.001
        else:
            # Normal/large dataset
            if self.control_ratio is None:
                self.control_ratio = 2
            if self.batch_size is None:
                self.batch_size = 32
            if self.hidden_size is None:
                self.hidden_size = 256
            if self.num_layers is None:
                self.num_layers = 2
            if self.epochs is None:
                self.epochs = 30
            if self.learning_rate is None:
                self.learning_rate = 0.001

    def _build_case_filter(self) -> str:
        """Build SPARQL FILTER clause for case conditions."""
        conditions = [f'CONTAINS(LCASE(?label), "{f.lower()}")'
                      for f in self.config.case_condition_filters]
        return " || ".join(conditions)

    def _build_control_filter(self) -> str:
        """Build SPARQL FILTER clause for control risk factors."""
        conditions = [f'CONTAINS(LCASE(?label), "{f.lower()}")'
                      for f in self.config.control_risk_filters]
        return " || ".join(conditions)

    def get_cases(self, client: PatientGraphClient) -> pd.DataFrame:
        """Get all patients with the target disease."""
        case_filter = self._build_case_filter()

        query = f"""{PREFIXES}
        SELECT ?patientId
               (MIN(?diagDate) AS ?diseaseDate)
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

            FILTER({case_filter})
        }}
        GROUP BY ?patientId
        ORDER BY ?patientId
        """
        return client.query(query)

    def get_potential_controls(self, client: PatientGraphClient) -> pd.DataFrame:
        """Get patients with risk factors but without the target disease."""
        control_filter = self._build_control_filter()
        case_filter = self._build_case_filter()

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

            # Risk factors for control population
            FILTER({control_filter})

            # Exclude patients with target disease
            FILTER NOT EXISTS {{
                ?patient ns28:patientCondition ?diseaseCond .
                ?diseaseCond ns28:code ?diseaseCode .
                ?diseaseCode skos:prefLabel ?diseaseLabel .
                FILTER({case_filter.replace('?label', '?diseaseLabel')})
            }}
        }}
        GROUP BY ?patientId
        ORDER BY ?patientId
        """
        return client.query(query)

    def get_patient_events(self, client: PatientGraphClient, patient_ids: list) -> pd.DataFrame:
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

    def match_controls(self, cases_df: pd.DataFrame,
                       potential_controls_df: pd.DataFrame) -> pd.DataFrame:
        """Match controls to cases based on demographics."""
        ref_date = pd.Timestamp('2024-01-01')

        cases_df = cases_df.copy()
        potential_controls_df = potential_controls_df.copy()

        cases_df['birthDate'] = pd.to_datetime(cases_df['birthDate'], errors='coerce')
        potential_controls_df['birthDate'] = pd.to_datetime(
            potential_controls_df['birthDate'], errors='coerce')

        cases_df['age'] = (ref_date - cases_df['birthDate']).dt.days / 365.25
        potential_controls_df['age'] = (
            ref_date - potential_controls_df['birthDate']).dt.days / 365.25

        # Age matching tolerance based on dataset size
        age_tolerance = 10 if self.n_cases < 300 else 5

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
                candidates = candidates[candidates['age_diff'] <= age_tolerance]
                candidates = candidates.sort_values('age_diff')

            selected = candidates.head(self.control_ratio)

            for _, ctrl in selected.iterrows():
                matched_controls.append(ctrl)
                used_control_ids.add(ctrl['patientId'])

        if matched_controls:
            return pd.DataFrame(matched_controls)
        else:
            return pd.DataFrame(columns=potential_controls_df.columns)

    def train(self) -> dict:
        """Run the full training pipeline."""
        print("=" * 70)
        print(f"{self.config.display_name.upper()} PREDICTION - MATCHED CASE-CONTROL DESIGN")
        print("=" * 70)
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {self.device}")

        # =====================================================================
        # Step 1: Get cases
        # =====================================================================
        print(f"\n--- Step 1: Getting {self.config.display_name.lower()} patients (cases) ---")

        with PatientGraphClient() as client:
            cases_df = self.get_cases(client)

        cases_df['diseaseDate'] = pd.to_datetime(cases_df['diseaseDate'])
        self.n_cases = len(cases_df)
        print(f"Total cases: {self.n_cases}")

        # Auto-tune hyperparameters based on dataset size
        self._auto_tune_hyperparameters()

        print(f"\nDesign: Matched case-control")
        print(f"  Prediction gap: {self.config.prediction_gap_days} days")
        print(f"  Cases: events up to (diagnosis - {self.config.prediction_gap_days} days)")
        print(f"  Controls: {self.config.lookback_years}-year window")
        print(f"  Control ratio: {self.control_ratio}:1")
        print(f"  Model: hidden={self.hidden_size}, layers={self.num_layers}")

        # Apply prediction gap to cases
        cases_df['cutoffDate'] = cases_df['diseaseDate'] - pd.Timedelta(
            days=self.config.prediction_gap_days)
        cases_df['windowStart'] = pd.Timestamp('1900-01-01', tz='UTC')
        cases_df['outcome'] = 1

        # =====================================================================
        # Step 2: Get potential controls
        # =====================================================================
        print(f"\n--- Step 2: Getting potential controls ---")

        with PatientGraphClient() as client:
            potential_controls = self.get_potential_controls(client)

        print(f"Potential controls: {len(potential_controls)}")

        # =====================================================================
        # Step 3: Match controls
        # =====================================================================
        print(f"\n--- Step 3: Matching controls to cases ---")

        matched_controls = self.match_controls(cases_df, potential_controls)
        print(f"Matched controls: {len(matched_controls)}")

        if len(matched_controls) == 0:
            raise ValueError("No controls matched!")

        control_ids = matched_controls['patientId'].tolist()

        # =====================================================================
        # Step 4: Extract case events
        # =====================================================================
        print(f"\n--- Step 4: Extracting events for cases ---")

        case_ids = cases_df['patientId'].tolist()
        batch_size_extract = 100 if self.n_cases > 200 else 50

        case_events_list = []
        with PatientGraphClient() as client:
            for i in range(0, len(case_ids), batch_size_extract):
                batch_ids = case_ids[i:i+batch_size_extract]
                batch_num = i // batch_size_extract + 1
                total_batches = (len(case_ids) - 1) // batch_size_extract + 1
                print(f"  Cases batch {batch_num}/{total_batches}...")
                try:
                    df = self.get_patient_events(client, batch_ids)
                    if len(df) > 0:
                        case_events_list.append(df)
                except Exception as e:
                    print(f"    Error: {e}")

        if not case_events_list:
            raise ValueError("No case events extracted!")

        case_events_df = pd.concat(case_events_list, ignore_index=True)
        print(f"Total case events: {len(case_events_df):,}")

        # =====================================================================
        # Step 5: Extract control events
        # =====================================================================
        print(f"\n--- Step 5: Extracting events for controls ---")

        control_events_list = []
        with PatientGraphClient() as client:
            for i in range(0, len(control_ids), batch_size_extract):
                batch_ids = control_ids[i:i+batch_size_extract]
                batch_num = i // batch_size_extract + 1
                total_batches = (len(control_ids) - 1) // batch_size_extract + 1
                print(f"  Controls batch {batch_num}/{total_batches}...")
                try:
                    df = self.get_patient_events(client, batch_ids)
                    if len(df) > 0:
                        control_events_list.append(df)
                except Exception as e:
                    print(f"    Error: {e}")

        if not control_events_list:
            raise ValueError("No control events extracted!")

        control_events_df = pd.concat(control_events_list, ignore_index=True)
        print(f"Total control events: {len(control_events_df):,}")

        # =====================================================================
        # Step 6: Apply time windows
        # =====================================================================
        print(f"\n--- Step 6: Applying time windows ---")

        case_events_df['eventDateTime'] = pd.to_datetime(case_events_df['eventDateTime'])
        control_events_df['eventDateTime'] = pd.to_datetime(control_events_df['eventDateTime'])

        # Cases: all events up to (diagnosis - prediction gap)
        case_events_df = case_events_df.merge(
            cases_df[['patientId', 'windowStart', 'cutoffDate']],
            on='patientId'
        )
        case_events_df = case_events_df[
            (case_events_df['eventDateTime'] >= case_events_df['windowStart']) &
            (case_events_df['eventDateTime'] < case_events_df['cutoffDate'])
        ].copy()

        print(f"Case events after time window: {len(case_events_df):,}")

        # Controls: lookback window before (last event - prediction gap)
        lookback_days = self.config.lookback_years * 365

        control_last_dates = control_events_df.groupby('patientId')['eventDateTime'].max().reset_index()
        control_last_dates.columns = ['patientId', 'lastEventDate']

        matched_controls = matched_controls.merge(control_last_dates, on='patientId', how='left')
        matched_controls['cutoffDate'] = matched_controls['lastEventDate'] - pd.Timedelta(
            days=self.config.prediction_gap_days)
        matched_controls['windowStart'] = matched_controls['cutoffDate'] - pd.Timedelta(
            days=lookback_days)
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

        # =====================================================================
        # Step 7: Save event data
        # =====================================================================
        print(f"\n--- Step 7: Saving event data for inspection ---")

        case_events_df['outcome'] = 1
        control_events_df['outcome'] = 0

        all_events_df = pd.concat([case_events_df, control_events_df], ignore_index=True)

        prefix = self.config.get_output_prefix()
        events_path = self.output_dir / f"{prefix}_training_events.csv"
        all_events_df.to_csv(events_path, index=False)
        print(f"Saved all events to: {events_path}")
        print(f"Total events: {len(all_events_df):,}")

        # =====================================================================
        # Step 8: Build vocabulary and sequences
        # =====================================================================
        print(f"\n--- Step 8: Building vocabulary and sequences ---")

        labels_df = pd.concat([
            cases_df[['patientId', 'outcome']],
            matched_controls[['patientId', 'outcome']]
        ], ignore_index=True)

        patients_with_events = all_events_df['patientId'].unique()
        labels_df = labels_df[labels_df['patientId'].isin(patients_with_events)].copy()

        n_cases_final = (labels_df['outcome'] == 1).sum()
        n_controls_final = (labels_df['outcome'] == 0).sum()

        print(f"Patients for training: {len(labels_df)}")
        print(f"  Cases: {n_cases_final}")
        print(f"  Controls: {n_controls_final}")

        min_freq = 1 if self.n_cases < 100 else 2
        preprocessor = SequencePreprocessor(
            max_length=self.app_config.sequence.max_length,
            min_frequency=min_freq
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

            if len(seq) > self.app_config.sequence.max_length:
                seq = seq[:self.app_config.sequence.max_length]

            sequences.append(seq)
            lengths.append(len(seq))
            labels.append(label_map[patient_id])
            patient_ids.append(patient_id)

        print(f"Patients with sequences: {len(sequences)}")
        print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")

        all_labels = np.array(labels)
        print(f"Label distribution: {sum(all_labels)} positive, {len(all_labels) - sum(all_labels)} negative")

        # =====================================================================
        # Step 9: Train/val/test split
        # =====================================================================
        print(f"\n--- Step 9: Splitting data ---")

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

        max_len = self.app_config.sequence.max_length

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

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        # =====================================================================
        # Step 10: Create model
        # =====================================================================
        print(f"\n--- Step 10: Creating model ---")

        model = AttentionPatientRNN(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=self.app_config.model.embedding_dim if self.hidden_size >= 256 else 64,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.3 if self.n_cases < 100 else self.app_config.model.dropout
        ).to(self.device)

        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {param_count:,}")

        pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
        pos_weight = min(pos_weight, 10.0)
        print(f"Positive weight: {pos_weight:.2f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                     weight_decay=0.01 if self.n_cases < 100 else 0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10 if self.n_cases < 100 else 5, factor=0.5)

        # =====================================================================
        # Step 11: Train
        # =====================================================================
        print(f"\n--- Step 11: Training ---")

        best_val_auc = 0
        best_model_state = None
        patience = 15 if self.n_cases < 100 else 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            model.train()
            total_loss = 0
            for batch in train_loader:
                seqs, lens, labs = [x.to(self.device) for x in batch]
                optimizer.zero_grad()
                outputs = model(seqs, lens)
                loss = criterion(outputs.squeeze(), labs.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            train_loss = total_loss / len(train_loader)

            # Validate
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch in val_loader:
                    seqs, lens, labs = [x.to(self.device) for x in batch]
                    outputs = model(seqs, lens)
                    loss = criterion(outputs.squeeze(), labs.float())
                    val_loss += loss.item()
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                    val_labels.extend(labs.cpu().numpy())

            val_loss /= len(val_loader)
            val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
            val_auprc = average_precision_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0

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

        # =====================================================================
        # Step 12: Test evaluation
        # =====================================================================
        print(f"\n--- Step 12: Test Evaluation ---")
        model.load_state_dict(best_model_state)
        model.eval()

        test_loss = 0
        test_preds = []
        test_labels = []
        with torch.no_grad():
            for batch in test_loader:
                seqs, lens, labs = [x.to(self.device) for x in batch]
                outputs = model(seqs, lens)
                loss = criterion(outputs.squeeze(), labs.float())
                test_loss += loss.item()
                test_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
                test_labels.extend(labs.cpu().numpy())

        test_loss /= len(test_loader)
        test_auc = roc_auc_score(test_labels, test_preds)
        test_auprc = average_precision_score(test_labels, test_preds)

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
                                    target_names=[f'No {self.config.display_name}',
                                                  self.config.display_name],
                                    zero_division=0))

        # =====================================================================
        # Step 13: Attention analysis
        # =====================================================================
        print(f"\n--- Step 13: Attention Analysis ---")

        event_attention_sum = {}
        event_occurrence_count = {}

        with torch.no_grad():
            for batch in test_loader:
                seqs, lens, labs = [x.to(self.device) for x in batch]
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

        # =====================================================================
        # Step 14: Save results
        # =====================================================================
        print(f"\n--- Step 14: Saving results ---")

        model_path = self.output_dir / f"{prefix}_model.pt"
        torch.save({
            'model_state_dict': best_model_state,
            'vocab': preprocessor.vocab,
            'config': self.app_config,
            'disease_config': {
                'name': self.config.name,
                'display_name': self.config.display_name,
                'case_condition_filters': self.config.case_condition_filters,
                'control_risk_filters': self.config.control_risk_filters,
            },
            'test_auc': test_auc,
            'test_auprc': test_auprc
        }, model_path)

        results = {
            'timestamp': datetime.now().isoformat(),
            'task': f'{self.config.name}_prediction',
            'disease': self.config.display_name,
            'design': 'matched_case_control',
            'prediction_gap_days': self.config.prediction_gap_days,
            'lookback_years': self.config.lookback_years,
            'control_ratio': self.control_ratio,
            'model_type': 'attention_rnn',
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'n_cases': int(n_cases_final),
            'n_controls': int(n_controls_final),
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

        results_path = self.output_dir / f"{prefix}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Model saved to: {model_path}")
        print(f"Results saved to: {results_path}")

        print("\n" + "=" * 70)
        print("Training complete!")
        print(f"Test AUROC: {test_auc:.4f}")
        print(f"Test AUPRC: {test_auprc:.4f}")
        print("=" * 70)

        return results
