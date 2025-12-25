# PG-RNN Project Session Summary
**Date:** December 24, 2025
**Status:** Active Development

## Project Overview

Building RNN-based predictive models using PatientGraph - a knowledge graph in AllegroGraph containing:
- **Synthea** synthetic patient data (11,584 patients, 474K encounters)
- **FDA clinical trials** from LinkedCT
- **PubMed COVID literature**
- **VAERS** vaccine adverse events
- All connected via **UMLS taxonomies**

Database: **424 million triples** in AllegroGraph on `localhost:10035`, repository `PatientGraph`

## Infrastructure Built

### Directory Structure
```
PG-RNN/
├── config.yaml              # AllegroGraph connection, model hyperparameters
├── requirements.txt         # Python dependencies
├── test_infrastructure.py   # Validates all components work
├── train_mortality.py       # 90-day mortality prediction training
├── train_readmission.py     # 30-day readmission prediction (poor results)
├── models/
│   └── mortality_model.pt   # Saved model checkpoint
└── src/
    ├── config.py            # Configuration dataclasses
    ├── agraph_client.py     # AllegroGraph wrapper using agraph-python
    ├── query_templates.py   # SPARQL query templates for RNN data
    ├── data_extractor.py    # Orchestrates data extraction with batching
    ├── sequence_preprocessor.py  # Vocabulary & tensor conversion
    └── models.py            # PyTorch RNN architectures
```

### Python Environment
```bash
# Conda environment
conda activate pgrnn  # Python 3.11

# Key packages
# - franz (agraph-python) for AllegroGraph
# - torch (CPU) for PyTorch
# - pandas, numpy, scikit-learn
```

### Configuration (config.yaml)
```yaml
agraph:
  host: localhost
  port: 10035
  repository: PatientGraph
  user: test
  password: xyzzy

model:
  type: "lstm"
  embedding_dim: 128
  hidden_size: 256
  num_layers: 2
  dropout: 0.3

sequence:
  max_length: 500
```

## SPARQL Query Templates (src/query_templates.py)

Six query templates for RNN data extraction:

1. **patient_base_query()** - Demographics (gender, race, birthdate, deathdate)
2. **event_sequence_query()** - Chronological events (CONDITIONS, MEDICATIONS, OBSERVATIONS, PROCEDURES, IMMUNIZATIONS)
3. **vocabulary_query()** - Event code frequencies for embedding lookup
4. **observation_timeseries_query()** - Numeric lab/vital values
5. **outcome_labels_query()** - Flexible prediction targets
6. **encounter_sequence_query()** - Encounter-level sequences

### Key Schema Properties
- `ns28:Patient` - Patient class
- `ns28:patientCondition`, `ns28:patientMedication`, etc. - Patient events
- `ns28:startDateTime`, `ns28:endDateTime` - Event timing
- `ns28:code` -> `skos:notation` - Medical codes (SNOMED, RxNorm, LOINC)
- `skos:prefLabel` - Human-readable descriptions

## Models Implemented (src/models.py)

1. **PatientLSTM** - Standard LSTM
2. **PatientGRU** - GRU variant
3. **PatientBiLSTM** - Bidirectional LSTM
4. **AttentionPatientRNN** - LSTM with attention mechanism (best performer)

All models:
- Embedding layer for medical codes
- Packed sequences for variable lengths
- Binary classification output
- ~1.1M parameters

## Training Results

### Task 1: Hospital Readmission (30-day) - POOR RESULTS
- **Problem:** Only 1% positive rate (35 readmissions in 3,447 inpatient patients)
- **Result:** Test AUROC = 0.50 (random chance)
- **Conclusion:** Insufficient positive samples for learning

### Task 2: Mortality Prediction - EXCELLENT RESULTS

#### First Attempt (First Year Only)
```
Observation window: 365 days (first year of patient history)
Events: 64,788
Patients: 11,534
Test AUROC: 0.72
```
Problem: Only used early patient data, prediction target unclear.

#### Improved Approach (All Events, 90-day Gap)
```
Prediction task: 90-day mortality
- Deceased: use events up to 90 days before death
- Living: use events up to 90 days before last encounter

Events: 1,893,564 (30x more)
Patients: 11,060
Vocabulary: 967 codes
Mean sequence length: 111.8 events

Training Results (stopped at epoch 15):
- Epoch 1:  val_auc = 0.9081
- Epoch 5:  val_auc = 0.9508 (best)
- Epoch 10: val_auc = 0.9414
- Epoch 15: val_auc = 0.9417

Best Validation AUROC: 0.95
```

### Comparison Table
| Metric | Readmission | Mortality (1yr) | Mortality (90-day gap) |
|--------|-------------|-----------------|------------------------|
| Test/Val AUROC | 0.50 | 0.72 | **0.95** |
| Events | 517K | 65K | 1.9M |
| Positive rate | 1% | 14% | 14% |
| Mean seq len | ~200 | 5.6 | 111.8 |

## Key Findings

### Top Attended Events (Mortality Predictors)
From attention analysis on first-year model:
| Code | Description | Attention |
|------|-------------|-----------|
| 80583007 | Severe anxiety (panic) | 1.00 |
| 243670 | Aspirin 81mg (low-dose) | 1.00 |
| 698754002 | Chronic paralysis (spinal cord) | 1.00 |
| 73595000 | Stress | 0.78 |
| 993770 | Acetaminophen/codeine | 0.31 |

### Technical Issues Encountered & Fixed

1. **SPARQL 500 errors** - Complex aggregations caused server errors; simplified queries
2. **Wrong property name** - Used `stopDateTime` instead of `endDateTime`
3. **Attention mask mismatch** - `pad_packed_sequence` returns dynamic size; fixed mask creation
4. **Training instability** - Capped pos_weight at reasonable values (6-10)
5. **Date arithmetic in SPARQL** - Duration filters failed; moved to Python filtering

## How to Run

### Test Infrastructure
```bash
conda activate pgrnn
python test_infrastructure.py
# Expected: 8 passed, 0 failed
```

### Train Mortality Model
```bash
conda activate pgrnn
PYTHONUNBUFFERED=1 python train_mortality.py
```

### Quick Query Test
```python
from src.agraph_client import PatientGraphClient
from src.query_templates import patient_base_query

with PatientGraphClient() as client:
    df = client.query(patient_base_query(limit=10))
    print(df)
```

## Files to Review

| File | Purpose |
|------|---------|
| `train_mortality.py` | Main training script with 90-day gap logic |
| `src/models.py` | AttentionPatientRNN with fixed mask handling |
| `src/query_templates.py` | SPARQL templates for data extraction |
| `src/agraph_client.py` | AllegroGraph connection wrapper |
| `config.yaml` | All configuration parameters |

## Next Steps (Not Yet Implemented)

1. **Add demographics** - Include age, gender as features
2. **Complete training** - Run full 30 epochs, evaluate on test set
3. **Attention analysis** - Identify top predictive events with full model
4. **Try other tasks** - ED visits, specific disease complications
5. **Cross-validation** - More robust evaluation
6. **Feature engineering** - Time gaps, event frequencies, observation values

## GraphTalker Integration

The project was designed to work with GraphTalker (eval server on port 9000) for interactive SPARQL exploration. GraphTalker recommended the 6-query framework used here.

Start GraphTalker:
```bash
cd /path/to/graphtalker
# Start eval server on port 9000
```

## Repository Info
- **Location:** `/mnt/c/Dropbox/lisp/PG-RNN`
- **Not a git repo** (consider initializing)
- **Conda env:** `pgrnn` (Python 3.11)
