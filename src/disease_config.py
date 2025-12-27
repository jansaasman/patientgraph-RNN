"""
Disease configuration for generic disease prediction.

Defines the schema for disease prediction tasks using YAML configuration files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class DiseaseConfig:
    """Configuration for a disease prediction task."""

    # Required fields
    name: str                           # e.g., "heart_failure"
    display_name: str                   # e.g., "Heart Failure"
    case_condition_filters: list[str]   # e.g., ["heart failure", "congestive heart"]
    control_risk_filters: list[str]     # e.g., ["hypertension", "diabetes"]

    # Optional overrides (auto-tuned if None)
    control_ratio: Optional[int] = None
    batch_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    epochs: Optional[int] = None
    learning_rate: Optional[float] = None

    # Defaults that are usually fixed
    prediction_gap_days: int = 182      # 6-month prediction horizon
    lookback_years: int = 5             # 5-year lookback for controls

    @classmethod
    def from_yaml(cls, path: str) -> "DiseaseConfig":
        """Load configuration from a YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        # Validate required fields
        required = ['name', 'display_name', 'case_condition_filters', 'control_risk_filters']
        for field_name in required:
            if field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")

        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        data = {
            'name': self.name,
            'display_name': self.display_name,
            'case_condition_filters': self.case_condition_filters,
            'control_risk_filters': self.control_risk_filters,
        }

        # Only include optional fields if they're set
        if self.control_ratio is not None:
            data['control_ratio'] = self.control_ratio
        if self.batch_size is not None:
            data['batch_size'] = self.batch_size
        if self.hidden_size is not None:
            data['hidden_size'] = self.hidden_size
        if self.num_layers is not None:
            data['num_layers'] = self.num_layers
        if self.epochs is not None:
            data['epochs'] = self.epochs
        if self.learning_rate is not None:
            data['learning_rate'] = self.learning_rate

        # Include non-default values
        if self.prediction_gap_days != 182:
            data['prediction_gap_days'] = self.prediction_gap_days
        if self.lookback_years != 5:
            data['lookback_years'] = self.lookback_years

        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_output_prefix(self) -> str:
        """Get the prefix for output files (model, results, events)."""
        return self.name.replace(' ', '_').lower()
