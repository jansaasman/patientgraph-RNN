# Disease Prediction Tutorial

This tutorial explains how to use the PG-RNN disease prediction system to train models and make predictions for individual patients.

## Overview

The system uses patient event sequences (conditions, medications, procedures, observations) from PatientGraph to predict disease risk. It employs:
- **Matched case-control design**: Cases matched to controls by age and gender
- **6-month prediction horizon**: Predicts disease 6 months before diagnosis
- **Attention-based LSTM**: Interpretable model showing which events matter most

## Table of Contents

1. [Creating a Disease Configuration](#1-creating-a-disease-configuration)
2. [Training a Model](#2-training-a-model)
3. [Making Predictions](#3-making-predictions)
4. [Understanding Results](#4-understanding-results)

---

## 1. Creating a Disease Configuration

Create a YAML file in the `configs/` directory.

### Required Fields

```yaml
# configs/my_disease.yaml

name: my_disease                    # Used for output filenames (no spaces)
display_name: My Disease            # Human-readable name for reports

case_condition_filters:             # Conditions that define the disease
  - heart failure                   # Searches for CONTAINS(LCASE(label), "heart failure")
  - congestive heart                # Multiple terms are OR'd together

control_risk_filters:               # At-risk population (without the disease)
  - hypertension                    # These patients form the control group
  - diabetes
  - coronary
```

### Optional Fields

```yaml
# Hyperparameters (auto-tuned if omitted)
control_ratio: 3                    # Controls per case (default: auto)
prediction_gap_days: 182            # Prediction horizon in days (default: 182 = 6 months)
lookback_years: 5                   # Event history window for controls (default: 5)

# Model architecture (auto-tuned based on dataset size)
hidden_size: 256                    # LSTM hidden size
num_layers: 2                       # Number of LSTM layers
batch_size: 32                      # Training batch size
epochs: 40                          # Maximum training epochs
learning_rate: 0.001                # Initial learning rate
```

### Auto-Tuning Rules

If you omit hyperparameters, they auto-tune based on the number of cases:

| Cases | Control Ratio | Hidden Size | Layers | Batch Size |
|-------|---------------|-------------|--------|------------|
| < 100 | 4:1 | 128 | 1 | 16 |
| 100-300 | 3:1 | 256 | 2 | 32 |
| > 300 | 2:1 | 256 | 2 | 32 |

### Example Configurations

**COPD Prediction:**
```yaml
name: copd
display_name: COPD

case_condition_filters:
  - chronic obstructive pulmonary
  - copd
  - emphysema

control_risk_filters:
  - smoking
  - asthma
  - bronchitis
  - respiratory infection
```

**Stroke Prediction:**
```yaml
name: stroke
display_name: Stroke

case_condition_filters:
  - stroke
  - cerebrovascular accident
  - cerebral infarction

control_risk_filters:
  - hypertension
  - atrial fibrillation
  - diabetes
  - hyperlipidemia
  - coronary artery disease
```

---

## 2. Training a Model

### Prerequisites

```bash
# Activate the conda environment
conda activate pgrnn

# Ensure PatientGraph (AllegroGraph) is running
```

### Running Training

```bash
# Train using a config file
python train_disease.py configs/heart_failure.yaml

# Train other diseases
python train_disease.py configs/nephropathy.yaml
python train_disease.py configs/atrial_fibrillation.yaml
```

### Training Output

The training process:
1. Queries PatientGraph for cases and controls
2. Matches controls to cases by demographics
3. Extracts event sequences
4. Trains an attention-based LSTM
5. Evaluates on held-out test set
6. Saves model and results

### Output Files

After training, you'll find in `models/`:

| File | Description |
|------|-------------|
| `{name}_model.pt` | Trained PyTorch model |
| `{name}_results.json` | Metrics, confusion matrix, top attention events |
| `{name}_training_events.csv` | All events used for training (for inspection) |

---

## 3. Making Predictions

### Using predict_patient.py

```bash
# Predict disease risk for a specific patient
python predict_patient.py <patient_id> <model_path>

# Examples:
python predict_patient.py "patient123" models/heart_failure_model.pt
python predict_patient.py "abc-def-ghi" models/nephropathy_model.pt
```

### Output

```
Patient: patient123
Risk Score: 0.7823 (78.23%)
Prediction: HIGH RISK for Heart Failure

Top Contributing Events:
  0.156  MEDICATION_314076 (lisinopril 10 MG)
  0.134  CONDITION_271737000 (hypertension)
  0.098  MEDICATION_310798 (hydrochlorothiazide 25 MG)
  ...
```

### Programmatic Usage

```python
import torch
from src.agraph_client import PatientGraphClient
from src.generic_trainer import DiseasePredictor

# Load trained model
checkpoint = torch.load('models/heart_failure_model.pt')
vocab = checkpoint['vocab']
model_state = checkpoint['model_state_dict']

# Create model and load weights
from src.models import AttentionPatientRNN
model = AttentionPatientRNN(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_size=256,
    num_layers=2
)
model.load_state_dict(model_state)
model.eval()

# Get patient events and predict
# ... (see predict_patient.py for full example)
```

### Important: Interpreting Predictions

The model is designed for **prospective prediction** - identifying patients who will develop the disease in the next 6 months. This has important implications:

| Patient Type | Expected Prediction | Why |
|--------------|---------------------|-----|
| **At-risk, no disease** | HIGH (70-99%) | Correct! Model identifies risk factors |
| **Already has disease** | LOW (1-10%) | Expected! Post-diagnosis treatment looks different |
| **Low-risk, healthy** | LOW (1-30%) | Correct! No risk factors |

**Example (Heart Failure):**
```
# At-risk patient (hypertension, no HF yet) → HIGH RISK
python predict_patient.py "Abe Brown" --model models/heart_failure_model.pt
# Heart Failure Risk Probability: 95.0%

# Patient already on HF treatment → LOW RISK
python predict_patient.py "Aaron Flatley" --model models/heart_failure_model.pt
# Heart Failure Risk Probability: 1.2%
```

The low score for existing patients makes sense because:
1. They're on disease-specific medications the model never saw during training
2. Training stopped 6 months before diagnosis
3. Post-diagnosis care patterns differ from pre-diagnosis

**Use case**: Screen at-risk patients (e.g., hypertensive, diabetic) to identify who needs closer monitoring.

---

## 4. Understanding Results

### Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| AUROC | Area under ROC curve | > 0.85 |
| AUPRC | Area under precision-recall curve | > 0.80 |
| Precision | Of predicted positives, how many are correct | > 0.80 |
| Recall | Of actual positives, how many were found | > 0.80 |

### Confusion Matrix

```
          Predicted
          No    Yes
Actual No  TN    FP    <- False alarms
      Yes  FN    TP    <- Missed cases
```

### Attention Analysis

The model outputs attention weights showing which events contributed most to predictions:

```
Top events by CUMULATIVE attention:
  20.33 (n=1489, avg=0.0137): MEDICATION_314076  <- lisinopril
  14.81 (n=1300, avg=0.0114): MEDICATION_308136  <- amlodipine
  13.26 (n= 953, avg=0.0139): MEDICATION_310798  <- hydrochlorothiazide
```

- **Cumulative**: Total attention across all patients (frequency × importance)
- **Average**: Per-occurrence importance (rare but highly predictive events)

### Looking Up Event Codes

To translate medication/condition codes to names:

```python
from src.agraph_client import PatientGraphClient
from src.query_templates import PREFIXES

code = "314076"
with PatientGraphClient() as client:
    query = f'''{PREFIXES}
    SELECT ?label WHERE {{
        ?med skos:notation "{code}" .
        ?med skos:prefLabel ?label .
    }} LIMIT 1
    '''
    result = client.query(query)
    print(result.iloc[0]['label'])
# Output: lisinopril 10 MG Oral Tablet
```

---

## Quick Reference

```bash
# Train a model
python train_disease.py configs/heart_failure.yaml

# Predict for a patient
python predict_patient.py "patient-id" models/heart_failure_model.pt

# Analyze attention weights
python analyze_attention.py models/heart_failure_model.pt
```

## Troubleshooting

**No cases found:**
- Check that `case_condition_filters` match condition labels in PatientGraph
- Try broader search terms

**Poor performance:**
- Increase `control_ratio` to get more training data
- Check that `control_risk_filters` define a meaningful at-risk population
- Reduce `prediction_gap_days` if 6 months is too aggressive

**Out of memory:**
- Reduce `batch_size`
- Reduce `hidden_size`
