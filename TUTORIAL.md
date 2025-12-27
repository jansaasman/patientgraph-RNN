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
4. [Running a Demo](#4-running-a-demo)
5. [Understanding Results](#5-understanding-results)

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

### Basic Usage

```bash
python predict_patient.py <patient_id> --model <model_path>
```

The prediction system automatically:
1. **Checks if patient is already diagnosed** → returns 100%
2. **If not diagnosed** → runs the model to predict risk

### Command Options

| Option | Description |
|--------|-------------|
| `--model PATH` | Path to trained model file |
| `--date YYYY-MM-DD` | Point-in-time prediction (use events up to this date) |
| `--top-k N` | Number of top attention events to show (default: 10) |
| `--no-attention` | Skip attention analysis for faster output |

### Example 1: Patient Already Diagnosed

```bash
python predict_patient.py "Aaron Flatley" --model models/heart_failure_model.pt
```

Output:
```
============================================================
PREDICTION RESULT
============================================================
Patient ID: Aaron Flatley
Heart Failure Status: DIAGNOSED
Diagnosis Date: 1999-04-30
Condition: Chronic congestive heart failure
Probability: 100.0%
============================================================
```

### Example 2: At-Risk Patient (Not Diagnosed)

```bash
python predict_patient.py "Abe Brown" --model models/heart_failure_model.pt
```

Output:
```
============================================================
PREDICTION RESULT
============================================================
Patient ID: Abe Brown
Heart Failure Risk Probability: 95.0%
Risk Level: HIGH
============================================================

Top 10 events by attention (what the model focused on):
------------------------------------------------------------
   1. [0.041] OBSERVATION  | 2008-09-22 | Body Weight
   2. [0.041] OBSERVATION  | 2008-09-22 | Pain severity
   3. [0.039] OBSERVATION  | 2008-09-22 | Total Cholesterol
   ...
```

### Example 3: Point-in-Time Prediction

Ask "what was the risk on a specific date?" - useful for validating the model on known cases.

```bash
# What was Aaron's risk in 1998 (before his 1999 diagnosis)?
python predict_patient.py "Aaron Flatley" --model models/heart_failure_model.pt --date 1998-01-01
```

Output:
```
============================================================
PREDICTION RESULT (as of 1998-01-01)
============================================================
Patient ID: Aaron Flatley
Heart Failure Risk Probability: 99.9%
Risk Level: HIGH
Note: Patient was later diagnosed on 1999-04-30
============================================================
```

The model correctly predicted 99.9% risk ~16 months before actual diagnosis!

### Prediction Logic Summary

| Scenario | What Happens |
|----------|--------------|
| Patient has disease | Returns 100% (DIAGNOSED) |
| Patient doesn't have disease | Runs model prediction |
| `--date` before diagnosis | Uses only events up to that date, runs model |
| `--date` after diagnosis | Returns 100% (already diagnosed by that date) |

### Programmatic Usage

```python
from predict_patient import predict_single_patient
from datetime import datetime

# Current prediction
result = predict_single_patient(
    patient_id="Abe Brown",
    model_path="models/heart_failure_model.pt"
)
print(f"Risk: {result['probability']:.1%}")

# Point-in-time prediction
result = predict_single_patient(
    patient_id="Aaron Flatley",
    model_path="models/heart_failure_model.pt",
    prediction_date=datetime(1998, 1, 1)
)
print(f"Risk in 1998: {result['probability']:.1%}")
```

---

## 4. Running a Demo

The `demo_predictions.py` script selects random cases and controls, runs predictions, and displays results in a formatted table.

### Basic Usage

```bash
# Run demo with default 10 patients per group
python demo_predictions.py configs/heart_failure.yaml

# Specify number of patients
python demo_predictions.py configs/atrial_fibrillation.yaml --count 20

# Save results to CSV
python demo_predictions.py configs/heart_failure.yaml --output results.csv

# CSV only (skip table output)
python demo_predictions.py configs/heart_failure.yaml --output results.csv --no-table
```

### Command Options

| Option | Description |
|--------|-------------|
| `config` | Path to YAML config file (required) |
| `--count N` | Number of patients per group (default: 10) |
| `--output FILE` | Save results to CSV file |
| `--no-table` | Skip markdown table output |

### How It Works

1. **Cases**: Selects random patients with the disease
   - Prediction date = 6 months before diagnosis
   - Shows what the model predicted before they were diagnosed

2. **Controls**: Selects random at-risk patients without the disease
   - Prediction date = their last event date
   - Shows current risk for patients who haven't developed the disease

### Example Output

```
## Atrial Fibrillation Prediction Demo

### Cases (patients with Atrial Fibrillation)
Prediction made 6 months before diagnosis

| Patient | Pred. Date | Risk % | Outcome | Top Predictors |
|---------|------------|--------|---------|----------------|
| Ernesto Koelpin | 2015-12-29 | 99.5% | Diagnosed 2016-06-28 | aspirin 81 MG... |
| Ben Reilly | 2009-07-31 | 99.4% | Diagnosed 2010-01-29 | aspirin 81 MG... |
| Perry Effertz | 2005-09-21 | 97.7% | Diagnosed 2006-03-22 | simvastatin... |

### Controls (at-risk, no Atrial Fibrillation)
Prediction made at last event date

| Patient | Pred. Date | Risk % | Outcome | Top Predictors |
|---------|------------|--------|---------|----------------|
| Claudio Rodríquez | 2016-06-11 | 95.8% | None | Anemia... |
| Peter Deckow | 2012-05-16 | 57.0% | None | Chronic sinusitis |
| Mario Figueroa | 2009-06-15 | 2.1% | None | lisinopril... |

### Summary
Cases: mean=72.3%, min=10.3%, max=99.5%
Controls: mean=30.1%, min=2.1%, max=95.8%
```

### Interpreting Demo Results

- **Good model**: Cases have higher mean risk than controls
- **High-risk controls**: May be future cases worth monitoring
- **Low-risk cases**: Model may have missed some patterns, or disease onset was sudden

---

## 5. Understanding Results

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

# Predict for a patient (checks if diagnosed, then runs model)
python predict_patient.py "patient-id" --model models/heart_failure_model.pt

# Point-in-time prediction (what was the risk on a specific date?)
python predict_patient.py "patient-id" --model models/heart_failure_model.pt --date 2018-01-01

# Quick prediction without attention analysis
python predict_patient.py "patient-id" --model models/heart_failure_model.pt --no-attention

# Run demo on random cases and controls
python demo_predictions.py configs/heart_failure.yaml

# Demo with more patients and CSV output
python demo_predictions.py configs/atrial_fibrillation.yaml --count 20 --output results.csv
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
