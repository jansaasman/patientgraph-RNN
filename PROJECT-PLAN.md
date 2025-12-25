# PatientGraph RNN Project Plan

**Created:** 2025-12-24
**Status:** Exploratory Phase Complete, Ready for Implementation

---

## 1. Project Goal

Build a **Recurrent Neural Network (RNN) toolkit** for PatientGraph that:
- Extracts chronological patient event sequences from AllegroGraph
- Trains LSTM/GRU models to predict clinical outcomes
- Is **generic and reusable** for any disease cohort (diabetes, heart failure, COPD, etc.)
- Integrates with the existing GraphTalker system for query generation

---

## 2. PatientGraph Database Summary

**Connection Details:**
```lisp
(initialize-agraph-connection "http" "localhost" 10035 "" "PatientGraph" "test" "xyzzy")
```

**Scale:**
- **424 million triples**
- **11,584 patients**
- **474,005 encounters**
- **333,487 conditions**
- **843,220 medications**
- **1,208,911 observations**

**Five Integrated Data Domains:**
1. **Patient Care Data** - Synthea-based synthetic EHR (ns28: namespace)
2. **UMLS Ontology** - 114 semantic types with hierarchical relationships (ns33:)
3. **Clinical Trials** - LinkedCT trial data (ns27:)
4. **PubMed Literature** - Articles with disease/drug/gene mentions (ns29:)
5. **VAERS** - Vaccine adverse event reports (ns31:, ns32:)

**Key Entity Relationships:**
```
Patient --patientEncounter--> Encounter
   |                              |
   +--patientCondition--> Condition <--encounterCondition--+
   +--patientMedication--> Medication <--encounterMedication--+
   +--patientObservation--> Observation <--encounterObservation--+
   +--patientProcedure--> Procedure <--encounterProcedure--+
```

**Code Mapping Chain:**
```
Clinical Event --> ns28:Code --> skos:exactMatch --> UMLS Concept (ns33:T047, etc.)
```

---

## 3. Data Distribution (From Exploratory Queries)

### Encounter Class Distribution
| Type | Count | Percentage |
|------|-------|------------|
| Wellness | 185,317 | 39.1% |
| Outpatient | 96,659 | 20.4% |
| Ambulatory | 83,135 | 17.5% |
| Urgent Care | 68,515 | 14.5% |
| Emergency | 22,194 | 4.7% |
| **Inpatient** | **5,671** | **1.2%** |
| Hospice | 1,072 | 0.2% |
| Virtual | 432 | 0.1% |
| Home | 366 | 0.1% |
| SNF | 182 | 0.04% |

### Events Per Patient (Averages)
| Event Type | Avg | Min | Max |
|------------|-----|-----|-----|
| Observations | 104 | 1 | 2,857 |
| Medications | 73 | 0 | 5,174 |
| Encounters | 40 | 0 | 1,658 |
| Conditions | 29 | 0 | 887 |
| Procedures | 16 | 0 | 235 |
| Immunizations | 1.5 | 0 | 20 |

**Implication for RNN:** Average sequence length ~260 events per patient (sum of all event types). Max sequence length up to ~10,000 for complex patients. Need truncation/windowing strategy.

---

## 4. Existing Diabetes Queries in Query Library

GraphTalker already has 10 sophisticated diabetes queries we can learn from:

1. **NQF0059 HbA1c Control** - Quality metric with nested subselects
2. **UMLS Hierarchy Cohort** - Uses `franz:narrower-inclusive-of` for diabetes subtypes
3. **Co-morbidities by Gender** - Disease clustering analysis
4. **Social Determinants** - SDOH findings (T033)
5. **Treatment Analysis** - Evidence-based prescribing using UMLS `may_be_treated_by`
6. **Billy Miller Disease History** - Individual patient timeline
7. **Billy Miller Medications** - Medication history with dates
8. **Medication Adherence Gaps** - Found 72.2% of diabetics without diabetes meds
9. **Multi-Specialty Care** - Cardiology/Endocrinology/Nephrology patterns
10. **SDOH Healthcare Costs** - Cost impact by social determinants

**Key Finding:** 3,655 patients (72.2%) have diabetes diagnosis but no diabetes medications - potential prediction target for medication adherence.

---

## 5. Generic Query Framework for RNN Training

GraphTalker recommended a **6-query framework** for extracting RNN training data:

### Query 1: Patient Base Table
**Purpose:** Static patient features and outcomes
**Output:** One row per patient with demographics, observation window, outcome labels

Key fields:
- patientId, gender, age, race, ethnicity, birthdate, deathdate
- died (binary), survivalDays
- totalEncounters, firstEncounterDate, lastEncounterDate, observationWindowDays
- totalHealthcareCost, uniqueConditions, uniqueMedications, uniqueProcedures
- chadvascScore, hasbledScore (existing risk scores)

### Query 2: Complete Event Sequence (Core RNN Input)
**Purpose:** ALL clinical events in chronological order
**Output:** One row per event, ordered by patient + timestamp

Key fields:
- patientId, eventDateTime
- daysSinceBirth, daysSinceFirstEvent (temporal features)
- eventType: CONDITION | MEDICATION | OBSERVATION | PROCEDURE | IMMUNIZATION | ENCOUNTER
- eventCode, eventDescription
- eventValue, eventUnits (for observations)
- eventCost, encounterClass, encounterCost

**Design:** Uses UNION to combine all 6 event types into single stream.

### Query 3: Event Vocabulary
**Purpose:** Build embedding lookup tables
**Output:** eventType, code, description, frequency

Used to create:
- `event_to_id` dictionary for encoding
- Special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`
- Frequency-based pruning of rare codes

### Query 4: Time-Windowed Aggregates
**Purpose:** Rolling window features (30-day, 90-day)
**Output:** Per patient per encounter: counts and costs in recent windows

Features:
- conditions_last30d, medications_last30d, procedures_last30d, etc.
- conditions_last90d, medications_last90d, encounters_last90d, etc.
- totalCost_last30d, totalCost_last90d

### Query 5: Observation Time Series
**Purpose:** Numeric lab/vital values over time
**Output:** Continuous time series for physiological modeling

Key fields:
- patientId, observationDateTime, daysSinceBirth, daysSinceFirstEvent
- obsCode, obsDescription (e.g., HbA1c, blood pressure, creatinine)
- numericValue, units, category

### Query 6: Outcome Labels (Flexible Template)
**Purpose:** Generate labels for any prediction task
**Output:** patientId, indexDate, outcomeOccurred (0/1), daysToOutcome

Customizable for:
- Hospital readmission (30 days) - use encounterclass = "inpatient"
- Diabetes complications - retinopathy, nephropathy, neuropathy
- Heart failure onset
- Mortality prediction
- Atrial fibrillation
- Medication adherence gaps

---

## 6. Proposed Python Architecture

### Directory Structure
```
PG-RNN/
├── PROJECT-PLAN.md              # This document
├── README.md                    # Original RNN guide (from Claude)
├── config.yaml                  # Configuration (AG connection, model params)
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Load config.yaml
│   ├── agraph_client.py         # AllegroGraph REST/SPARQL client
│   ├── query_templates.py       # The 6 query templates (parameterized)
│   ├── data_extractor.py        # Execute queries, return DataFrames
│   ├── sequence_preprocessor.py # Convert to tensors, build vocab
│   ├── models.py                # LSTM, GRU, BiLSTM architectures
│   ├── train.py                 # Training loop with early stopping
│   ├── evaluate.py              # Metrics (AUROC, AUPRC, confusion matrix)
│   └── predict.py               # Inference on new patients
│
├── notebooks/
│   ├── 0_exploration.ipynb      # Explore PatientGraph data
│   ├── 1_cohort_selection.ipynb # Build patient cohorts
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_training.ipynb
│   └── 4_evaluation.ipynb
│
├── queries/                     # Saved SPARQL templates
│   ├── patient_base.sparql
│   ├── event_sequence.sparql
│   ├── vocabulary.sparql
│   ├── time_windows.sparql
│   ├── observations.sparql
│   └── outcome_labels.sparql
│
├── data/                        # Cached data (gitignore)
│   ├── raw/                     # Raw SPARQL results (CSV/JSON)
│   ├── processed/               # Preprocessed tensors
│   └── embeddings/              # Code embeddings (optional pretrained)
│
├── models/                      # Saved model checkpoints
│   └── best_model.pt
│
└── logs/                        # Training logs, TensorBoard
```

### Key Python Classes

**1. AllegroGraphClient**
```python
class AllegroGraphClient:
    def __init__(self, url, repository, catalog="/"):
        self.endpoint = f"{url}/repositories/{repository}"

    def query(self, sparql: str, timeout=300) -> pd.DataFrame:
        # Execute SPARQL, return DataFrame

    def test_connection(self) -> bool:
        # Verify connection
```

**2. QueryTemplates** (Parameterized)
```python
class QueryTemplates:
    @staticmethod
    def patient_base(cohort_filter: str = None) -> str:
        # Return Query 1 with optional cohort WHERE clause

    @staticmethod
    def event_sequence(patient_ids: List[str] = None) -> str:
        # Return Query 2, optionally filtered to specific patients

    @staticmethod
    def outcome_labels(
        outcome_pattern: str,  # REGEX for outcome condition
        prediction_horizon_days: int = 30
    ) -> str:
        # Return Query 6 customized for prediction task
```

**3. SequencePreprocessor**
```python
class SequencePreprocessor:
    def __init__(self, max_seq_length=500, min_freq=5):
        self.vocab = {}
        self.max_seq_length = max_seq_length

    def fit(self, vocab_df: pd.DataFrame):
        # Build event_to_id mapping from vocabulary query

    def transform(self, events_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        # Convert events to padded sequences
        # Returns: {sequences, lengths, patient_ids}

    def encode_event(self, event_type: str, code: str) -> int:
        # Map event to ID
```

**4. PatientLSTM**
```python
class PatientLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256,
                 num_layers=2, dropout=0.3, num_classes=2):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(...)
        self.fc = nn.Linear(...)

    def forward(self, sequences, lengths):
        # Pack sequences, run LSTM, return predictions
```

---

## 7. Potential Prediction Tasks

### Task 1: Hospital Readmission (30-day)
- **Target:** Inpatient encounter within 30 days of discharge
- **Cohort:** All patients with at least one inpatient stay
- **Class balance:** ~5,671 inpatient stays, need to check readmission rate

### Task 2: Diabetes Complications
- **Target:** Onset of retinopathy, nephropathy, or neuropathy
- **Cohort:** Type 2 diabetes patients (codes 44054006, 127013003)
- **Features:** HbA1c trajectory, medication adherence, comorbidities

### Task 3: Medication Adherence
- **Target:** Gap in expected medications (e.g., diabetic without metformin)
- **Cohort:** Patients with chronic conditions
- **Novel:** 72.2% gap already identified in diabetes population

### Task 4: Emergency Department Utilization
- **Target:** Emergency encounter within 90 days
- **Cohort:** All patients with prior healthcare utilization
- **Features:** Prior ED visits, chronic conditions, social determinants

### Task 5: Mortality Prediction
- **Target:** Death within 1 year
- **Cohort:** All patients
- **Features:** Full clinical history, demographics, SDOH

---

## 8. Integration with GraphTalker

### Option A: Direct AllegroGraph REST API
- Python connects directly to AllegroGraph via SPARQLWrapper
- Simpler, no dependency on Lisp server
- Queries stored as .sparql files

### Option B: Via GraphTalker Eval Server
- Python sends queries through eval server at port 9000
- Can leverage Claude for query debugging/iteration
- More complex but enables natural language query refinement

**Recommendation:** Start with Option A for batch data extraction, use Option B for interactive exploration and query development.

### Eval Server Access Pattern
```python
import requests

def eval_lisp(expression: str, port=9000):
    response = requests.post(
        f"http://localhost:{port}/eval",
        json={"expression": expression}
    )
    return response.json()

# Example: Ask Claude to generate a query
result = eval_lisp('(claude-query "Find all diabetic patients with HbA1c > 9")')
```

---

## 9. Implementation Phases

### Phase 1: Data Infrastructure (Current)
- [x] Explore PatientGraph structure via GraphTalker
- [x] Identify encounter/event distributions
- [x] Review existing diabetes queries
- [x] Design generic query framework (6 queries)
- [x] Document project plan (this file)
- [ ] Create query templates in Python
- [ ] Build AllegroGraphClient
- [ ] Test data extraction on small cohort

### Phase 2: Sequence Processing
- [ ] Build vocabulary from Query 3
- [ ] Implement SequencePreprocessor
- [ ] Handle variable-length sequences (padding, packing)
- [ ] Add temporal features (time deltas, age)
- [ ] Cache processed tensors

### Phase 3: Model Development
- [ ] Implement PatientLSTM and PatientGRU
- [ ] Add embedding layer for medical codes
- [ ] Implement training loop with early stopping
- [ ] Add evaluation metrics (AUROC, AUPRC, etc.)

### Phase 4: First Prediction Task
- [ ] Choose task (recommend: hospital readmission or diabetes complications)
- [ ] Generate outcome labels with Query 6
- [ ] Train baseline model
- [ ] Evaluate and iterate

### Phase 5: Advanced Features
- [ ] Bidirectional LSTM
- [ ] Attention mechanisms
- [ ] Pre-trained UMLS embeddings (using hierarchy)
- [ ] Multi-task learning
- [ ] Incorporate SDOH features

---

## 10. Technical Decisions

### Sequence Length
- **Average:** ~260 events/patient
- **Max:** ~10,000 events for complex patients
- **Strategy:** Truncate to most recent 500 events, or use attention for longer sequences

### Code Vocabulary
- Combine event type + code: `CONDITION_44054006`, `MEDICATION_RxNorm123`
- Prune codes with frequency < 5
- Add special tokens: `<PAD>`, `<UNK>`, `<START>`, `<END>`
- Expected vocab size: ~5,000-10,000 unique codes

### Temporal Encoding
- `daysSinceBirth`: Absolute patient age at event
- `daysSinceFirstEvent`: Relative position in observation window
- `timeSinceLastEvent`: Gap between consecutive events (computed in Python)

### Class Imbalance
- Inpatient encounters: 1.2% of all encounters
- Use weighted loss or oversampling
- Report both AUROC and AUPRC

---

## 11. agraph-python Patterns (From Notebooks & Library)

The project will use the official `agraph-python` library rather than a custom REST client.

### Connection Pattern
```python
from franz.openrdf.connect import ag_connect
from franz.openrdf.query.query import QueryLanguage

# Connect to PatientGraph
conn = ag_connect('PatientGraph',
                  host='localhost',
                  port='10035',
                  user='test',
                  password='xyzzy')

print(f"Connected: {conn.size()} triples")
```

### Query Execution with Pandas
```python
# Method 1: Direct to pandas (preferred for RNN data extraction)
def rsp(query):
    df = conn.executeTupleQuery(query).toPandas()
    return df

# Method 2: With context manager for iteration
with conn.executeTupleQuery(query) as result:
    for row in result:
        print(row['patientId'], row['eventType'])

# Method 3: Prepared query with inference
query = conn.prepareTupleQuery(QueryLanguage.SPARQL, sparql_string)
query.setIncludeInferred(True)
df = query.evaluate().toPandas()
```

### Namespace Handling
```python
# Register namespaces for cleaner queries
conn.setNamespace('ns28', 'http://patientgraph.ai/')
conn.setNamespace('skos', 'http://www.w3.org/2004/02/skos/core#')
conn.setNamespace('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
conn.setNamespace('xsd', 'http://www.w3.org/2001/XMLSchema#')

# Get all namespaces
namespaces = dict(conn.getNamespaces())
```

### URI Shortening for Display
```python
def shorten_uri(val, namespaces):
    """Convert full URIs to prefixed form."""
    s = str(val)
    if s.startswith('<') and s.endswith('>'):
        uri = s[1:-1]
        for prefix, ns in namespaces.items():
            if uri.startswith(ns):
                return prefix + ':' + uri[len(ns):]
    return s

def shorten_df(df, namespaces):
    """Apply URI shortening to DataFrame."""
    for col in df.columns:
        df[col] = df[col].apply(lambda val: shorten_uri(val, namespaces))
    return df
```

### Key Classes from agraph-python
- `ag_connect()` - Convenience function for connection
- `RepositoryConnection` - Main connection class
- `TupleQuery.evaluate()` - Returns `TupleQueryResult`
- `TupleQueryResult.toPandas()` - Converts to DataFrame
- `QueryLanguage.SPARQL` / `QueryLanguage.PROLOG`

### Example: Extract Patient Events
```python
from franz.openrdf.connect import ag_connect
import pandas as pd

conn = ag_connect('PatientGraph', host='localhost', port='10035',
                  user='test', password='xyzzy')

# Set up namespaces
conn.setNamespace('ns28', 'http://patientgraph.ai/')
conn.setNamespace('skos', 'http://www.w3.org/2004/02/skos/core#')

query = """
PREFIX ns28: <http://patientgraph.ai/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT ?patientId ?eventType ?eventCode ?eventDateTime
WHERE {
    ?patient a ns28:Patient ;
             rdfs:label ?patientId .
    ?patient ns28:patientCondition ?condition .
    ?condition ns28:startDateTime ?eventDateTime ;
               ns28:code ?codeUri .
    ?codeUri skos:notation ?eventCode .
    BIND("CONDITION" AS ?eventType)
}
ORDER BY ?patientId ?eventDateTime
LIMIT 1000
"""

df = conn.executeTupleQuery(query).toPandas()
print(f"Extracted {len(df)} events")
```

---

## 12. Dependencies

```
# requirements.txt
# AllegroGraph Python client
agraph-python>=103.0.0

# ML/Deep Learning
torch>=2.0
numpy
pandas

# ML utilities
scikit-learn
tqdm

# Visualization
matplotlib
seaborn

# Notebooks
jupyter

# Logging
tensorboard

# Config
pyyaml
```

---

## 13. Next Steps

1. **Create Python infrastructure** - AllegroGraphClient, QueryTemplates
2. **Test Query 2** (event sequence) on 100 patients
3. **Build vocabulary** from Query 3
4. **Implement SequencePreprocessor**
5. **Train first LSTM** on hospital readmission task

---

## 14. Key Resources

### GraphTalker
- Location: `./graphtalker/` (softlink)
- Eval server: `~/dropboxlisp/claude-mcp-in-lisp/start-server.sh 9000`
- Docs: `graphtalker/CLAUDE.md`, `graphtalker/docs/AGENT-GUIDE.md`

### PatientGraph
- Host: localhost:10035
- Repository: PatientGraph
- Credentials: test/xyzzy

### Existing Queries
- Query library has 10+ diabetes queries
- Access via: `(claude-query "Search the query library for diabetes")`

---

*This plan was created through interactive exploration of PatientGraph using GraphTalker. The 6-query framework and Python architecture were designed to be generic and reusable across any disease cohort.*
