"""
Configuration management for PG-RNN.

Loads configuration from config.yaml and provides easy access to settings.
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_project_root() / "config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


@dataclass
class AGConfig:
    """AllegroGraph connection configuration."""
    host: str = "localhost"
    port: int = 10035
    repository: str = "PatientGraph"
    user: str = "test"
    password: str = "xyzzy"
    catalog: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> 'AGConfig':
        return cls(**d)


@dataclass
class SequenceConfig:
    """Sequence processing configuration."""
    max_length: int = 500
    min_frequency: int = 5
    event_types: List[str] = field(default_factory=lambda: [
        "CONDITION", "MEDICATION", "OBSERVATION",
        "PROCEDURE", "IMMUNIZATION", "ENCOUNTER"
    ])

    @classmethod
    def from_dict(cls, d: dict) -> 'SequenceConfig':
        return cls(**d)


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    type: str = "lstm"
    embedding_dim: int = 128
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> 'ModelConfig':
        return cls(**d)


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 50
    early_stopping_patience: int = 5
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        return cls(**d)


class Config:
    """
    Main configuration class that loads and provides access to all settings.

    Usage:
        config = Config()
        conn = ag_connect(config.agraph.repository, ...)
    """

    def __init__(self, config_path: Optional[str] = None):
        self._raw = load_config(config_path)
        self._project_root = get_project_root()

        # Parse sub-configurations
        self.agraph = AGConfig.from_dict(self._raw.get('agraph', {}))
        self.namespaces = self._raw.get('namespaces', {})
        self.sequence = SequenceConfig.from_dict(self._raw.get('sequence', {}))
        self.model = ModelConfig.from_dict(self._raw.get('model', {}))
        self.training = TrainingConfig.from_dict(self._raw.get('training', {}))
        self.special_tokens = self._raw.get('special_tokens', {
            'pad': '<PAD>', 'unk': '<UNK>', 'start': '<START>', 'end': '<END>'
        })
        self.tasks = self._raw.get('tasks', {})
        self.extraction = self._raw.get('extraction', {})

    def get_path(self, key: str) -> Path:
        """Get a path from the paths configuration."""
        paths = self._raw.get('paths', {})
        relative_path = paths.get(key, key)
        return self._project_root / relative_path

    @property
    def data_raw_path(self) -> Path:
        return self.get_path('data_raw')

    @property
    def data_processed_path(self) -> Path:
        return self.get_path('data_processed')

    @property
    def models_path(self) -> Path:
        return self.get_path('models')

    @property
    def logs_path(self) -> Path:
        return self.get_path('logs')

    @property
    def queries_path(self) -> Path:
        return self.get_path('queries')


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to config file (only used on first call)

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
