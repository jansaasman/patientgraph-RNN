"""
Data Extractor for PatientGraph RNN Training.

Orchestrates extraction of patient event sequences from AllegroGraph
using the query templates. Handles batching, caching, and DataFrame output.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd

from .config import Config, get_config
from .agraph_client import PatientGraphClient
from . import query_templates as qt


class DataExtractor:
    """
    Extracts patient event sequences from PatientGraph for RNN training.

    Usage:
        extractor = DataExtractor()
        extractor.connect()

        # Get patient base table
        patients_df = extractor.extract_patients(limit=1000)

        # Get event sequences
        events_df = extractor.extract_event_sequences(patient_ids=patient_ids)

        # Get vocabulary
        vocab_df = extractor.extract_vocabulary()

        extractor.close()
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the extractor.

        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.client = PatientGraphClient(self.config)
        self._connected = False

    def connect(self) -> 'DataExtractor':
        """Connect to PatientGraph."""
        self.client.connect()
        self._connected = True
        return self

    def close(self):
        """Close the connection."""
        self.client.close()
        self._connected = False

    def __enter__(self) -> 'DataExtractor':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_connected(self):
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

    def test_connection(self) -> Dict[str, Any]:
        """Test connection and return stats."""
        self._ensure_connected()
        return self.client.test_connection()

    # =========================================================================
    # Core Extraction Methods
    # =========================================================================

    def extract_patients(
        self,
        cohort_filter: Optional[str] = None,
        limit: Optional[int] = None,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract patient base table with demographics.

        Args:
            cohort_filter: Optional SPARQL WHERE clause for cohort selection
            limit: Optional limit on number of patients
            cache: Whether to cache results

        Returns:
            DataFrame with patient demographics and summary stats
        """
        self._ensure_connected()

        # Check cache
        cache_key = self._cache_key("patients", cohort_filter, limit)
        if cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # Execute query
        query = qt.patient_base_query(cohort_filter=cohort_filter, limit=limit)
        df = self.client.query(query)

        # Process DataFrame
        df = self._process_patient_df(df)

        # Cache results
        if cache:
            self._save_cache(cache_key, df)

        return df

    def extract_event_sequences(
        self,
        patient_ids: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        limit: Optional[int] = None,
        batch_size: int = 100,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract chronological event sequences for patients.

        Args:
            patient_ids: List of patient IDs (None = all patients)
            event_types: Event types to include (default: all)
            limit: Optional limit on total events
            batch_size: Number of patients per batch
            cache: Whether to cache results

        Returns:
            DataFrame with events ordered by patient + time
        """
        self._ensure_connected()

        # If no patient IDs provided, fetch all
        if patient_ids is None:
            # Get all patient IDs first
            patients_df = self.extract_patients(limit=limit)
            patient_ids = patients_df['patientId'].tolist()

        # Check cache for full result
        cache_key = self._cache_key("events", str(patient_ids[:5]), event_types, limit)
        if cache and len(patient_ids) <= batch_size:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # Batch extraction for large patient lists
        if len(patient_ids) > batch_size:
            return self._batch_extract_events(
                patient_ids, event_types, limit, batch_size
            )

        # Single query for small lists
        query = qt.event_sequence_query(
            patient_ids=patient_ids,
            event_types=event_types,
            limit=limit
        )
        df = self.client.query(query)

        # Process DataFrame
        df = self._process_events_df(df)

        # Cache
        if cache:
            self._save_cache(cache_key, df)

        return df

    def _batch_extract_events(
        self,
        patient_ids: List[str],
        event_types: Optional[List[str]],
        limit: Optional[int],
        batch_size: int
    ) -> pd.DataFrame:
        """Extract events in batches to avoid query timeouts."""
        all_dfs = []

        for i in range(0, len(patient_ids), batch_size):
            batch_ids = patient_ids[i:i + batch_size]
            print(f"Extracting events for patients {i+1}-{i+len(batch_ids)} of {len(patient_ids)}...")

            query = qt.event_sequence_query(
                patient_ids=batch_ids,
                event_types=event_types,
                limit=None  # No limit per batch
            )
            df = self.client.query(query)
            df = self._process_events_df(df)
            all_dfs.append(df)

        # Combine all batches
        combined = pd.concat(all_dfs, ignore_index=True)

        # Apply overall limit if specified
        if limit and len(combined) > limit:
            combined = combined.head(limit)

        return combined

    def extract_vocabulary(
        self,
        limit: Optional[int] = None,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract event vocabulary with frequencies.

        Args:
            limit: Optional limit on vocabulary size
            cache: Whether to cache results

        Returns:
            DataFrame with event codes and frequencies
        """
        self._ensure_connected()

        cache_key = self._cache_key("vocabulary", limit)
        if cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        query = qt.vocabulary_query(limit=limit)
        df = self.client.query(query)

        # Process
        df = self._process_vocabulary_df(df)

        if cache:
            self._save_cache(cache_key, df)

        return df

    def extract_observations(
        self,
        patient_ids: Optional[List[str]] = None,
        observation_codes: Optional[List[str]] = None,
        limit: Optional[int] = None,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract numeric observation time series.

        Args:
            patient_ids: Optional list of patient IDs
            observation_codes: Optional LOINC codes to filter
            limit: Optional limit

        Returns:
            DataFrame with observation values over time
        """
        self._ensure_connected()

        cache_key = self._cache_key("observations", str(patient_ids), observation_codes, limit)
        if cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        query = qt.observation_timeseries_query(
            patient_ids=patient_ids,
            observation_codes=observation_codes,
            limit=limit
        )
        df = self.client.query(query)
        df = self._process_observations_df(df)

        if cache:
            self._save_cache(cache_key, df)

        return df

    def extract_outcome_labels(
        self,
        outcome_type: str = "hospitalization",
        prediction_horizon_days: int = 30,
        cohort_filter: Optional[str] = None,
        limit: Optional[int] = None,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract outcome labels for prediction tasks.

        Args:
            outcome_type: "hospitalization", "mortality", or "complication"
            prediction_horizon_days: Days ahead to predict
            cohort_filter: Optional cohort filter
            limit: Optional limit

        Returns:
            DataFrame with patient IDs and outcome labels
        """
        self._ensure_connected()

        cache_key = self._cache_key(
            "outcomes", outcome_type, prediction_horizon_days, cohort_filter, limit
        )
        if cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        query = qt.outcome_labels_query(
            outcome_type=outcome_type,
            prediction_horizon_days=prediction_horizon_days,
            cohort_filter=cohort_filter,
            limit=limit
        )
        df = self.client.query(query)
        df = self._process_outcomes_df(df)

        if cache:
            self._save_cache(cache_key, df)

        return df

    def extract_encounters(
        self,
        patient_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract encounter-level sequences.

        Args:
            patient_ids: Optional patient IDs
            limit: Optional limit

        Returns:
            DataFrame with encounters and event counts
        """
        self._ensure_connected()

        cache_key = self._cache_key("encounters", str(patient_ids), limit)
        if cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        query = qt.encounter_sequence_query(patient_ids=patient_ids, limit=limit)
        df = self.client.query(query)

        if cache:
            self._save_cache(cache_key, df)

        return df

    # =========================================================================
    # DataFrame Processing
    # =========================================================================

    def _process_patient_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process patient DataFrame."""
        # Convert date columns
        date_cols = ['birthdate', 'deathdate', 'firstEncounterDate', 'lastEncounterDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Convert numeric columns
        numeric_cols = ['age', 'died', 'totalEncounters', 'uniqueConditions',
                       'uniqueMedications', 'uniqueProcedures']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def _process_events_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process events DataFrame."""
        if 'eventDateTime' in df.columns:
            df['eventDateTime'] = pd.to_datetime(df['eventDateTime'], errors='coerce')

        # Sort by patient and time
        df = df.sort_values(['patientId', 'eventDateTime']).reset_index(drop=True)

        return df

    def _process_vocabulary_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process vocabulary DataFrame."""
        if 'frequency' in df.columns:
            df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')

        # Sort by frequency descending
        df = df.sort_values('frequency', ascending=False).reset_index(drop=True)

        return df

    def _process_observations_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process observations DataFrame."""
        if 'observationDateTime' in df.columns:
            df['observationDateTime'] = pd.to_datetime(
                df['observationDateTime'], errors='coerce'
            )

        if 'numericValue' in df.columns:
            df['numericValue'] = pd.to_numeric(df['numericValue'], errors='coerce')

        df = df.sort_values(['patientId', 'observationDateTime']).reset_index(drop=True)

        return df

    def _process_outcomes_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process outcomes DataFrame."""
        date_cols = ['indexDate', 'outcomeDate']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        if 'outcomeOccurred' in df.columns:
            df['outcomeOccurred'] = pd.to_numeric(df['outcomeOccurred'], errors='coerce')

        return df

    # =========================================================================
    # Caching
    # =========================================================================

    def _cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = "_".join(str(a) for a in args if a is not None)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        """Get cache file path."""
        cache_dir = self.config.data_raw_path
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"cache_{key}.parquet"

    def _load_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Load DataFrame from cache if exists."""
        path = self._cache_path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, df: pd.DataFrame):
        """Save DataFrame to cache."""
        path = self._cache_path(key)
        try:
            df.to_parquet(path, index=False)
        except Exception as e:
            print(f"Warning: Could not cache data: {e}")

    def clear_cache(self):
        """Clear all cached data."""
        cache_dir = self.config.data_raw_path
        if cache_dir.exists():
            for f in cache_dir.glob("cache_*.parquet"):
                f.unlink()


# =========================================================================
# Convenience Functions
# =========================================================================

def extract_training_data(
    cohort_filter: Optional[str] = None,
    outcome_type: str = "hospitalization",
    prediction_horizon_days: int = 30,
    max_patients: Optional[int] = None,
    event_types: Optional[List[str]] = None,
    config: Optional[Config] = None
) -> Dict[str, pd.DataFrame]:
    """
    Extract complete training dataset in one call.

    Args:
        cohort_filter: Optional SPARQL filter for cohort
        outcome_type: Type of outcome to predict
        prediction_horizon_days: Prediction horizon
        max_patients: Optional limit on patients
        event_types: Event types to include
        config: Configuration object

    Returns:
        Dictionary with DataFrames: patients, events, vocabulary, labels
    """
    with DataExtractor(config) as extractor:
        print("Testing connection...")
        info = extractor.test_connection()
        print(f"Connected to {info['repository']} ({info['triple_count']:,} triples)")

        print("\nExtracting patients...")
        patients_df = extractor.extract_patients(
            cohort_filter=cohort_filter,
            limit=max_patients
        )
        print(f"  Found {len(patients_df)} patients")

        patient_ids = patients_df['patientId'].tolist()

        print("\nExtracting event sequences...")
        events_df = extractor.extract_event_sequences(
            patient_ids=patient_ids,
            event_types=event_types
        )
        print(f"  Found {len(events_df)} events")

        print("\nExtracting vocabulary...")
        vocab_df = extractor.extract_vocabulary()
        print(f"  Found {len(vocab_df)} unique codes")

        print("\nExtracting outcome labels...")
        labels_df = extractor.extract_outcome_labels(
            outcome_type=outcome_type,
            prediction_horizon_days=prediction_horizon_days,
            cohort_filter=cohort_filter,
            limit=max_patients
        )
        print(f"  Found {labels_df['outcomeOccurred'].sum()} positive outcomes")

        return {
            'patients': patients_df,
            'events': events_df,
            'vocabulary': vocab_df,
            'labels': labels_df
        }
