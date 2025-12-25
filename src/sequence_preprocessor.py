"""
Sequence Preprocessor for RNN Training.

Converts extracted patient event DataFrames into PyTorch-ready tensors.
Handles vocabulary building, sequence encoding, and padding.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import Config, get_config


class EventVocabulary:
    """
    Vocabulary for encoding clinical events.

    Maps event_type + code combinations to integer IDs.
    """

    def __init__(
        self,
        special_tokens: Optional[Dict[str, str]] = None,
        min_frequency: int = 1
    ):
        """
        Initialize vocabulary.

        Args:
            special_tokens: Dict of special token names to tokens
            min_frequency: Minimum frequency for a code to be included
        """
        self.min_frequency = min_frequency

        # Special tokens
        self.special_tokens = special_tokens or {
            'pad': '<PAD>',
            'unk': '<UNK>',
            'start': '<START>',
            'end': '<END>'
        }

        # Initialize mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_counts: Dict[str, int] = {}

        # Add special tokens
        for name, token in self.special_tokens.items():
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self._fitted = False

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special_tokens['pad']]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.special_tokens['unk']]

    @property
    def start_id(self) -> int:
        return self.token_to_id[self.special_tokens['start']]

    @property
    def end_id(self) -> int:
        return self.token_to_id[self.special_tokens['end']]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def fit(self, vocab_df: pd.DataFrame) -> 'EventVocabulary':
        """
        Build vocabulary from vocabulary DataFrame.

        Args:
            vocab_df: DataFrame with columns: eventType, code, frequency

        Returns:
            self for chaining
        """
        # Count frequencies
        for _, row in vocab_df.iterrows():
            token = self._make_token(row['eventType'], row['code'])
            freq = int(row.get('frequency', 1))
            self.token_counts[token] = freq

        # Add tokens above threshold
        for token, freq in sorted(self.token_counts.items(), key=lambda x: -x[1]):
            if freq >= self.min_frequency and token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        self._fitted = True
        return self

    def fit_from_events(self, events_df: pd.DataFrame) -> 'EventVocabulary':
        """
        Build vocabulary directly from events DataFrame.

        Args:
            events_df: DataFrame with columns: eventType, eventCode

        Returns:
            self for chaining
        """
        # Count frequencies
        for _, row in events_df.iterrows():
            if pd.notna(row.get('eventCode')):
                token = self._make_token(row['eventType'], row['eventCode'])
                self.token_counts[token] = self.token_counts.get(token, 0) + 1

        # Add tokens above threshold
        for token, freq in sorted(self.token_counts.items(), key=lambda x: -x[1]):
            if freq >= self.min_frequency and token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        self._fitted = True
        return self

    def _make_token(self, event_type: str, code: str) -> str:
        """Create a token from event type and code."""
        return f"{event_type}_{code}"

    def encode(self, event_type: str, code: str) -> int:
        """
        Encode a single event to its ID.

        Args:
            event_type: Event type (CONDITION, MEDICATION, etc.)
            code: Event code (SNOMED, RxNorm, LOINC, etc.)

        Returns:
            Integer ID
        """
        token = self._make_token(event_type, code)
        return self.token_to_id.get(token, self.unk_id)

    def decode(self, idx: int) -> str:
        """
        Decode an ID back to token string.

        Args:
            idx: Integer ID

        Returns:
            Token string
        """
        return self.id_to_token.get(idx, self.special_tokens['unk'])

    def save(self, path: Path):
        """Save vocabulary to file."""
        data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'token_counts': self.token_counts,
            'special_tokens': self.special_tokens,
            'min_frequency': self.min_frequency
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'EventVocabulary':
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        vocab = cls(
            special_tokens=data['special_tokens'],
            min_frequency=data['min_frequency']
        )
        vocab.token_to_id = data['token_to_id']
        vocab.id_to_token = {int(k): v for k, v in data['id_to_token'].items()}
        vocab.token_counts = data['token_counts']
        vocab._fitted = True

        return vocab


class SequencePreprocessor:
    """
    Preprocessor that converts event DataFrames to RNN-ready sequences.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        max_length: int = 500,
        min_frequency: int = 5
    ):
        """
        Initialize preprocessor.

        Args:
            config: Configuration object
            max_length: Maximum sequence length (truncate longer)
            min_frequency: Minimum code frequency for vocabulary
        """
        self.config = config or get_config()
        self.max_length = max_length
        self.min_frequency = min_frequency

        # Get special tokens from config
        special_tokens = self.config.special_tokens

        self.vocab = EventVocabulary(
            special_tokens=special_tokens,
            min_frequency=min_frequency
        )

    def fit(self, vocab_df: pd.DataFrame) -> 'SequencePreprocessor':
        """
        Fit vocabulary from vocabulary DataFrame.

        Args:
            vocab_df: DataFrame from vocabulary query

        Returns:
            self for chaining
        """
        self.vocab.fit(vocab_df)
        return self

    def fit_from_events(self, events_df: pd.DataFrame) -> 'SequencePreprocessor':
        """
        Fit vocabulary directly from events.

        Args:
            events_df: Events DataFrame

        Returns:
            self for chaining
        """
        self.vocab.fit_from_events(events_df)
        return self

    def transform(
        self,
        events_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Transform events DataFrame to sequences.

        Args:
            events_df: Events DataFrame with columns:
                       patientId, eventType, eventCode, eventDateTime
            labels_df: Optional labels DataFrame with columns:
                       patientId, outcomeOccurred

        Returns:
            Dictionary with:
                - patient_ids: List of patient IDs
                - sequences: List of encoded sequences (lists of ints)
                - lengths: List of sequence lengths
                - labels: Optional array of labels
        """
        # Group events by patient
        patient_groups = events_df.groupby('patientId')

        patient_ids = []
        sequences = []
        lengths = []

        for patient_id, group in patient_groups:
            # Sort by time
            group = group.sort_values('eventDateTime')

            # Encode events
            seq = []
            for _, row in group.iterrows():
                if pd.notna(row.get('eventCode')):
                    event_id = self.vocab.encode(row['eventType'], row['eventCode'])
                    seq.append(event_id)

            if len(seq) == 0:
                continue

            # Truncate to max length
            if len(seq) > self.max_length:
                seq = seq[-self.max_length:]  # Keep most recent

            patient_ids.append(patient_id)
            sequences.append(seq)
            lengths.append(len(seq))

        result = {
            'patient_ids': patient_ids,
            'sequences': sequences,
            'lengths': lengths
        }

        # Add labels if provided
        if labels_df is not None:
            labels = self._align_labels(patient_ids, labels_df)
            result['labels'] = labels

        return result

    def _align_labels(
        self,
        patient_ids: List[str],
        labels_df: pd.DataFrame
    ) -> np.ndarray:
        """Align labels to patient order."""
        labels_dict = dict(zip(
            labels_df['patientId'],
            labels_df['outcomeOccurred']
        ))

        labels = np.array([
            labels_dict.get(pid, 0) for pid in patient_ids
        ], dtype=np.float32)

        return labels

    def to_tensors(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, 'torch.Tensor']:
        """
        Convert processed data to PyTorch tensors.

        Args:
            data: Output from transform()

        Returns:
            Dictionary with tensors:
                - sequences: Padded sequence tensor [batch, max_len]
                - lengths: Length tensor [batch]
                - labels: Optional label tensor [batch]
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for tensor conversion")

        sequences = data['sequences']
        lengths = data['lengths']

        # Pad sequences
        padded = self._pad_sequences(sequences)
        seq_tensor = torch.LongTensor(padded)
        len_tensor = torch.LongTensor(lengths)

        result = {
            'sequences': seq_tensor,
            'lengths': len_tensor,
            'patient_ids': data['patient_ids']
        }

        if 'labels' in data:
            result['labels'] = torch.FloatTensor(data['labels'])

        return result

    def _pad_sequences(self, sequences: List[List[int]]) -> np.ndarray:
        """Pad sequences to same length."""
        max_len = max(len(s) for s in sequences)
        padded = np.full((len(sequences), max_len), self.vocab.pad_id, dtype=np.int64)

        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq

        return padded

    def save(self, path: Path):
        """Save preprocessor state."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vocab
        self.vocab.save(path / 'vocabulary.json')

        # Save config
        config = {
            'max_length': self.max_length,
            'min_frequency': self.min_frequency
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'SequencePreprocessor':
        """Load preprocessor from saved state."""
        path = Path(path)

        # Load config
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        preprocessor = cls(
            max_length=config['max_length'],
            min_frequency=config['min_frequency']
        )

        # Load vocab
        preprocessor.vocab = EventVocabulary.load(path / 'vocabulary.json')

        return preprocessor

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)


if HAS_TORCH:
    class PatientSequenceDataset(Dataset):
        """
        PyTorch Dataset for patient sequences.
        """

        def __init__(
            self,
            sequences: torch.Tensor,
            lengths: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            patient_ids: Optional[List[str]] = None
        ):
            """
            Initialize dataset.

            Args:
                sequences: Padded sequence tensor [N, max_len]
                lengths: Length tensor [N]
                labels: Optional label tensor [N]
                patient_ids: Optional list of patient IDs
            """
            self.sequences = sequences
            self.lengths = lengths
            self.labels = labels
            self.patient_ids = patient_ids

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
            item = {
                'sequence': self.sequences[idx],
                'length': self.lengths[idx]
            }

            if self.labels is not None:
                item['label'] = self.labels[idx]

            return item

        @classmethod
        def from_preprocessor_output(
            cls,
            tensors: Dict[str, torch.Tensor]
        ) -> 'PatientSequenceDataset':
            """Create dataset from preprocessor output."""
            return cls(
                sequences=tensors['sequences'],
                lengths=tensors['lengths'],
                labels=tensors.get('labels'),
                patient_ids=tensors.get('patient_ids')
            )


    def create_dataloaders(
        dataset: PatientSequenceDataset,
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split dataset and create DataLoaders.

        Args:
            dataset: PatientSequenceDataset
            batch_size: Batch size
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            seed: Random seed

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Calculate split sizes
        n = len(dataset)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        # Random split
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val, n_test]
        )

        # Create loaders
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False
        )

        return train_loader, val_loader, test_loader
