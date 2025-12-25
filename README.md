Perfect! I'll create a comprehensive markdown document for your Claude Code project. This will be a complete reference guide linking PatientGraph/GraphTalker knowledge with RNN development.

PATIENTGRAPH RNN TOOLKIT - COMPLETE GUIDE
A Claude Code Project for Training Recurrent Neural Networks on Clinical Event Sequences

TABLE OF CONTENTS
Project Overview
Prerequisites & Setup
Project Structure
AllegroGraph Connection
SPARQL Pattern Library
Query Builder Implementation
Data Extraction Pipeline
Sequence Preprocessing
RNN Model Architecture
Training Pipeline
Evaluation & Metrics
Example Workflows
Troubleshooting
References
1. PROJECT OVERVIEW
What This Project Does
This toolkit extracts temporal event sequences from PatientGraph (an AllegroGraph RDF database) and trains recurrent neural networks (LSTM/GRU) to predict clinical outcomes.

Use Cases:

Predict hospitalization risk from patient history
Forecast next HbA1c value for diabetic patients
Model disease progression trajectories
Identify high-risk patients for intervention
Predict medication adherence patterns
Why RNNs for Clinical Data?
Patient health records are naturally sequential:

Timeline: [Diagnosis] → [Medication] → [Lab Test] → [Office Visit] → [Complication]
RNNs maintain a "memory" of previous events through hidden states, making them ideal for:

Variable-length sequences (patients have different event counts)
Temporal dependencies (a medication affects future lab values)
Long-range interactions (diagnosis years ago impacts current risk)
Architecture Overview
┌────────────────────────────────────────────────────────────┐
│ STEP 1: COHORT SELECTION (SPARQL)                          │
│ - Query PatientGraph for eligible patients                 │
│ - Filter by demographics, conditions, date ranges          │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 2: SEQUENCE EXTRACTION (SPARQL)                       │
│ - Extract all events per patient (chronologically sorted)  │
│ - Include: Conditions, Meds, Labs, Procedures, Encounters  │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 3: FEATURE ENGINEERING (Python)                       │
│ - Convert codes to embeddings                              │
│ - Add temporal features (time-since-last-event)            │
│ - Normalize continuous values (lab results)                │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 4: SEQUENCE PADDING & BATCHING (PyTorch)              │
│ - Pad sequences to same length                             │
│ - Create train/val/test splits                             │
│ - Generate DataLoader batches                              │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 5: RNN TRAINING (PyTorch)                             │
│ - LSTM/GRU processes sequences                             │
│ - Backpropagation through time (BPTT)                      │
│ - Optimize loss (CrossEntropy, MSE, etc.)                  │
└────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────┐
│ STEP 6: EVALUATION & PREDICTION                            │
│ - Test on held-out patients                                │
│ - Metrics: AUROC, AUPRC, Accuracy, MAE                     │
│ - Deploy for real-time predictions                         │
└────────────────────────────────────────────────────────────┘
2. PREREQUISITES & SETUP
A. AllegroGraph Access
You need a running AllegroGraph instance with PatientGraph repository.

Connection Details:

URL: http://localhost:10035 (or your server)
Repository: PatientGraph
Catalog: / (root catalog)
Test Connection:

curl "http://localhost:10035/repositories/PatientGraph/size"
B. Python Environment
Create a new conda/venv environment:

# Create environment
conda create -n patientgraph-rnn python=3.10
conda activate patientgraph-rnn

# Install dependencies
pip install torch torchvision torchaudio  # PyTorch
pip install numpy pandas scikit-learn matplotlib seaborn
pip install requests SPARQLWrapper  # For AllegroGraph queries
pip install jupyter notebook  # For interactive development
pip install pyyaml  # For config files
pip install tqdm  # For progress bars
C. Claude Code Project Setup
Create this directory structure:

PatientGraph-RNN/
├── README.md                          # This document
├── config.yaml                        # Configuration
├── requirements.txt                   # Python dependencies
│
├── graphtalker_guide/                 # Reference docs
│   ├── query_patterns.md              # SPARQL patterns
│   └── umls_codes.md                  # Common medical codes
│
├── queries/                           # SPARQL query templates
│   ├── diabetic_cohort.sparql
│   ├── patient_timeline.sparql
│   └── condition_hierarchy.sparql
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── config.py                      # Load config.yaml
│   ├── query_builder.py               # Generate SPARQL queries
│   ├── data_extractor.py              # Execute queries, return DataFrames
│   ├── sequence_preprocessor.py       # Convert to tensors
│   ├── models.py                      # RNN architectures
│   ├── train.py                       # Training loop
│   └── evaluate.py                    # Evaluation metrics
│
├── notebooks/                         # Jupyter notebooks
│   ├── 0_exploration.ipynb            # Explore PatientGraph data
│   ├── 1_cohort_selection.ipynb       # Build patient cohorts
│   ├── 2_feature_engineering.ipynb    # Design features
│   ├── 3_model_training.ipynb         # Train RNN
│   └── 4_evaluation.ipynb             # Analyze results
│
├── data/                              # Cached data (gitignore)
│   ├── raw/                           # Raw SPARQL results
│   ├── processed/                     # Preprocessed tensors
│   └── embeddings/                    # Code embeddings
│
├── models/                            # Saved model checkpoints
│   └── best_model.pt
│
└── logs/                              # Training logs
    └── training_2024-01-15.log
3. PROJECT STRUCTURE
config.yaml
# AllegroGraph Connection
allegrograph:
  url: "http://localhost:10035"
  repository: "PatientGraph"
  catalog: "/"
  timeout: 300  # seconds

# Cohort Selection
cohort:
  condition: "Type 2 Diabetes Mellitus"  # Target population
  age_min: 18
  age_max: 75
  index_date_start: "2018-06-01"
  index_date_end: "2023-12-31"
  min_events_per_patient: 5  # Filter out patients with too few events

# Sequence Extraction
sequence:
  event_types: ['Condition', 'Medication', 'Observation', 'Procedure', 'Encounter']
  max_sequence_length: 100  # Truncate or pad to this length
  lookback_days: 365  # Only include events within this window
  prediction_window_days: 30  # Predict events in next N days

# Feature Engineering
features:
  code_embedding_dim: 64  # Embedding size for medical codes
  use_temporal_features: true
  temporal_features:
    - 'time_since_last_event'
    - 'time_since_diagnosis'
    - 'patient_age_at_event'
  normalize_values: true  # Normalize lab values

# Model Architecture
model:
  type: "LSTM"  # or "GRU", "BiLSTM"
  input_size: 128  # Will be calculated based on features
  hidden_size: 256
  num_layers: 2
  dropout: 0.3
  bidirectional: false

# Training
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 10  # Early stopping patience
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  random_seed: 42

# Prediction Task
task:
  type: "binary_classification"  # or "multiclass", "regression"
  target: "hospitalization_30d"  # What to predict
  target_definition: "Inpatient encounter within 30 days"
  
# Logging
logging:
  log_dir: "logs"
  save_model_every_n_epochs: 5
  tensorboard: true
4. ALLEGROGRAPH CONNECTION
REST API Basics
AllegroGraph exposes a REST API for SPARQL queries. The endpoint format is:

http://{host}:{port}/repositories/{repository}
Example:

http://localhost:10035/repositories/PatientGraph
Python Connection Class
# src/allegrograph_client.py

import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd
from typing import Optional, Dict, Any

class AllegroGraphClient:
    """
    Client for executing SPARQL queries against AllegroGraph.
    """
    
    def __init__(self, url: str, repository: str, catalog: str = "/"):
        """
        Args:
            url: Base URL (e.g., "http://localhost:10035")
            repository: Repository name (e.g., "PatientGraph")
            catalog: Catalog name (default: "/")
        """
        self.url = url
        self.repository = repository
        self.catalog = catalog
        
        # Construct endpoint URL
        if catalog == "/":
            self.endpoint = f"{url}/repositories/{repository}"
        else:
            self.endpoint = f"{url}/catalogs/{catalog}/repositories/{repository}"
        
        print(f"AllegroGraph endpoint: {self.endpoint}")
    
    def query(self, sparql_query: str, timeout: int = 300) -> pd.DataFrame:
        """
        Execute a SPARQL SELECT query and return results as DataFrame.
        
        Args:
            sparql_query: SPARQL query string
            timeout: Query timeout in seconds
            
        Returns:
            pandas DataFrame with query results
        """
        sparql = SPARQLWrapper(self.endpoint)
        sparql.setQuery(sparql_query)
        sparql.setReturnFormat(JSON)
        sparql.setTimeout(timeout)
        
        try:
            results = sparql.query().convert()
            return self._sparql_results_to_dataframe(results)
        except Exception as e:
            print(f"Query failed: {e}")
            print(f"Query was:\n{sparql_query}")
            raise
    
    def _sparql_results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Convert SPARQL JSON results to pandas DataFrame."""
        bindings = results['results']['bindings']
        
        if not bindings:
            # Empty result set
            return pd.DataFrame()
        
        # Extract variable names
        vars = results['head']['vars']
        
        # Extract values
        rows = []
        for binding in bindings:
            row = {}
            for var in vars:
                if var in binding:
                    row[var] = binding[var]['value']
                else:
                    row[var] = None
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_repository_size(self) -> int:
        """Get number of triples in repository."""
        response = requests.get(f"{self.endpoint}/size")
        return int(response.text)
    
    def test_connection(self) -> bool:
        """Test if connection is working."""
        try:
            size = self.get_repository_size()
            print(f"✓ Connection successful! Repository has {size:,} triples.")
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            return False

# Example usage:
if __name__ == "__main__":
    client = AllegroGraphClient(
        url="http://localhost:10035",
        repository="PatientGraph"
    )
    
    if client.test_connection():
        # Run a simple test query
        query = """
        PREFIX : <http://patientgraph.ai/>
        SELECT (COUNT(?patient) AS ?count) WHERE {
            ?patient a :Patient .
        }
        """
        result = client.query(query)
        print(result)
5. SPARQL PATTERN LIBRARY
This section contains SPARQL patterns extracted from PatientGraph tutorials, organized by use case.

Common Prefixes
Always include these at the start of your queries:

prefix pg: <http://patientgraph.ai/>
prefix : <http://patientgraph.ai/>
prefix skos: <http://www.w3.org/2004/02/skos/core#>
prefix umls: <https://uts.nlm.nih.gov/uts/umls/concept/>
prefix umls-scheme: <https://uts.nlm.nih.gov/uts/umls/vocabulary/2022AA/>
prefix umls-rel: <https://uts.nlm.nih.gov/uts/umls/relation#>
prefix sem-network: <https://uts.nlm.nih.gov/uts/umls/semantic-network/>
prefix franz: <http://franz.com/>
prefix xsd: <http://www.w3.org/2001/XMLSchema#>
prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PATTERN 1: Find Patients with a Specific Condition
Use Case: Select diabetic patients for your cohort

Tutorial Reference: NQF0059 query, synthea-exploration-tutorial.txt

# Find patients with Type 2 Diabetes Mellitus
SELECT DISTINCT ?patient ?diagnosisDate WHERE {
    VALUES ?diabetesCode { 
        "44054006"          # Type 2 diabetes mellitus
        "127013003"         # Disorder of kidney due to diabetes mellitus
        "90781000119102"    # Microalbuminuria due to type 2 DM
    }
    
    ?condition rdf:type :Condition; 
               :code ?code;
               :startDateTime ?diagnosisDate .
    ?code skos:notation ?diabetesCode .
    ?patient rdf:type :Patient; 
             :patientCondition ?condition .
    
    # Filter by date range
    FILTER (?diagnosisDate >= "2018-06-01T00:00:00+00:00"^^xsd:dateTime)
    FILTER (?diagnosisDate <= "2023-12-31T23:59:59+00:00"^^xsd:dateTime)
}
Key Points:

Use VALUES for explicit code lists
skos:notation holds the SNOMED/LOINC/RxNorm code string
:startDateTime is when the condition was first recorded
PATTERN 2: Extract Patient Timeline (All Events)
Use Case: Get chronological sequence of all events for a patient

Tutorial Reference: standard-reporting-tutorial.txt, synthea-exploration-tutorial.txt

# Get all events for a specific patient, chronologically sorted
SELECT ?eventType ?code ?codeLabel ?datetime ?value ?units WHERE {
    VALUES ?targetPatient { :patient-12345 }  # Replace with actual patient URI
    
    # Conditions
    {
        ?targetPatient :patientCondition ?event .
        ?event a :Condition ;
               :code ?codeResource ;
               :startDateTime ?datetime .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Condition" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }
    UNION
    # Medications
    {
        ?targetPatient :patientMedication ?event .
        ?event a :Medication ;
               :code ?codeResource ;
               :startDateTime ?datetime .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Medication" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }
    UNION
    # Observations (Labs)
    {
        ?targetPatient :patientObservation ?event .
        ?event a :Observation ;
               :code ?codeResource ;
               :startDateTime ?datetime ;
               :value ?value .
        OPTIONAL { ?event :units ?units }
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Observation" AS ?eventType)
    }
    UNION
    # Procedures
    {
        ?targetPatient :patientProcedure ?event .
        ?event a :Procedure ;
               :code ?codeResource ;
               :startDateTime ?datetime .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Procedure" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }
    UNION
    # Encounters
    {
        ?targetPatient :patientEncounter ?event .
        ?event a :Encounter ;
               :code ?codeResource ;
               :startDateTime ?datetime ;
               :encounterclass ?encounterClass .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND(CONCAT("Encounter-", ?encounterClass) AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }
}
ORDER BY ?datetime
Key Points:

UNION combines different event types
All events have :startDateTime for chronological sorting
Observations have :value (lab results)
Encounters have :encounterclass (inpatient, outpatient, emergency, etc.)
PATTERN 3: Nested Subselect for Cohort + Sequences
Use Case: Find diabetic patients AND extract their sequences in one query

Tutorial Reference: NQF0059 query (nested subselect technique)

# Find diabetic patients with recent office visits, then get their sequences
SELECT ?patient ?eventType ?code ?datetime ?value WHERE {
    {
        # INNER QUERY: Find diabetic patients
        SELECT DISTINCT ?patient WHERE {
            VALUES ?diabetesCode { "44054006" "127013003" }
            
            ?condition rdf:type :Condition; 
                       :code ?codeRes;
                       :startDateTime ?diagnosisDate .
            ?codeRes skos:notation ?diabetesCode .
            ?patient rdf:type :Patient; 
                     :patientCondition ?condition ;
                     :patientAge ?age .
            
            # Filter criteria
            FILTER (?diagnosisDate > "2018-06-01T00:00:00+00:00"^^xsd:dateTime)
            FILTER (?age >= 18 && ?age <= 75)
            
            # Require office visit in measurement period
            ?patient :patientEncounter ?enc .
            ?enc :code/skos:notation "162673000" ;  # Office visit
                 :startDateTime ?encDate .
            FILTER (?encDate > "2020-06-01T00:00:00+00:00"^^xsd:dateTime)
        }
    }
    
    # OUTER QUERY: For each qualifying patient, get all events
    {
        ?patient :patientCondition ?event .
        ?event :code ?codeRes ; :startDateTime ?datetime .
        ?codeRes skos:notation ?code .
        BIND("Condition" AS ?eventType)
        BIND("" AS ?value)
    }
    UNION
    {
        ?patient :patientObservation ?event .
        ?event :code ?codeRes ; :startDateTime ?datetime ; :value ?value .
        ?codeRes skos:notation ?code .
        BIND("Observation" AS ?eventType)
    }
    # ... add more UNIONs for other event types
}
ORDER BY ?patient ?datetime
PATTERN 4: Use UMLS Hierarchy to Find Related Conditions
Use Case: Find all subtypes of a disease (e.g., diabetes complications)

Tutorial Reference: umls-sparql-tutorial.txt, synthea-exploration-tutorial.txt

# Find all diabetes-related conditions using UMLS hierarchy
SELECT DISTINCT ?snomedCode ?label WHERE {
    # Step 1: Start with "Diabetes Mellitus" concept
    ?concept skos:prefLabel "Diabetes Mellitus" .
    
    # Step 2: Get all narrower concepts (subtypes) up to 4 levels deep
    ?disease franz:narrower-inclusive-of (?concept 4) .
    
    # Step 3: Map to SNOMED codes
    ?snomedRes skos:exactMatch ?disease ;
               skos:inScheme umls-scheme:SNOMEDCT_US ;
               skos:notation ?snomedCode ;
               skos:prefLabel ?label .
}
ORDER BY ?label
Key Points:

franz:narrower-inclusive-of (?concept N) gets all descendants up to N levels
skos:exactMatch links UMLS concepts to vocabulary-specific codes
skos:inScheme umls-scheme:SNOMEDCT_US filters to SNOMED codes only
PATTERN 5: Filter by Date Range
Use Case: Only include events within a specific time window

Tutorial Reference: standard-reporting-tutorial.txt

# Get events within the last 12 months
SELECT ?patient ?eventType ?datetime WHERE {
    ?patient :patientCondition ?event .
    ?event :startDateTime ?datetime .
    
    # Absolute date filter
    FILTER (?datetime >= "2023-01-01T00:00:00+00:00"^^xsd:dateTime)
    FILTER (?datetime <= "2023-12-31T23:59:59+00:00"^^xsd:dateTime)
}
Relative Date Filtering (requires computed reference date):

# Events within 6 months of index date
SELECT ?patient ?datetime WHERE {
    ?patient :patientCondition ?event .
    ?event :startDateTime ?datetime .
    
    # Assume ?indexDate is bound earlier in the query
    BIND(?indexDate - "P6M"^^xsd:duration AS ?startWindow)
    BIND(?indexDate + "P6M"^^xsd:duration AS ?endWindow)
    
    FILTER (?datetime >= ?startWindow && ?datetime <= ?endWindow)
}
PATTERN 6: Get Most Recent Event per Patient
Use Case: Find the latest HbA1c value for each diabetic patient

Tutorial Reference: NQF0059 query

# Get most recent HbA1c observation per patient
SELECT ?patient ?obsDate ?a1cValue WHERE {
    {
        # Subquery to find MAX date per patient
        SELECT ?patient (MAX(?dt) AS ?obsDate) WHERE {
            ?patient :patientObservation ?obs .
            ?obs :code/skos:notation "4548-4" ;  # HbA1c LOINC code
                 :startDateTime ?dt .
        }
        GROUP BY ?patient
    }
    
    # Join back to get the value at that date
    ?patient :patientObservation ?obs .
    ?obs :code/skos:notation "4548-4" ;
         :startDateTime ?obsDate ;
         :value ?a1cValue .
}
PATTERN 7: Count Events per Patient
Use Case: Filter patients by minimum event count

# Find patients with at least 10 medication fills
SELECT ?patient (COUNT(?med) AS ?medCount) WHERE {
    ?patient :patientMedication ?med .
}
GROUP BY ?patient
HAVING (COUNT(?med) >= 10)
PATTERN 8: Link to Demographics
Use Case: Get patient age, gender, location for feature engineering

Tutorial Reference: standard-reporting-tutorial.txt

# Get patient demographics
SELECT ?patient ?age ?gender ?city ?state ?birthdate WHERE {
    ?patient a :Patient ;
             :patientAge ?age ;
             :gender ?gender ;
             :city ?city ;
             :state ?state ;
             :birthdate ?birthdate .
}
PATTERN 9: Find Patients with Co-occurring Conditions
Use Case: Find diabetic patients who also have hypertension (comorbidity analysis)

Tutorial Reference: synthea-exploration-tutorial.txt

# Find patients with both diabetes AND hypertension
SELECT DISTINCT ?patient ?dmDate ?htDate WHERE {
    # Diabetes
    ?patient :patientCondition ?dmCond .
    ?dmCond :code/skos:notation "44054006" ;  # Type 2 DM
            :startDateTime ?dmDate .
    
    # Hypertension
    ?patient :patientCondition ?htCond .
    ?htCond :code/skos:notation "59621000" ;  # Essential hypertension
            :startDateTime ?htDate .
}
PATTERN 10: Extract Target Labels for Prediction
Use Case: Label patients who had a hospitalization in the next 30 days

# For each patient, determine if they had an inpatient encounter
# within 30 days of their index date
SELECT ?patient ?indexDate 
       (IF(BOUND(?hospDate), 1, 0) AS ?hospitalized_30d) 
WHERE {
    # Get index date (e.g., diagnosis date)
    ?patient :patientCondition ?cond .
    ?cond :code/skos:notation "44054006" ;
          :startDateTime ?indexDate .
    
    # Look for inpatient encounter within 30 days
    OPTIONAL {
        ?patient :patientEncounter ?enc .
        ?enc :encounterclass "inpatient" ;
             :startDateTime ?hospDate .
        
        BIND(?hospDate - ?indexDate AS ?timeDiff)
        FILTER (?timeDiff >= "P0D"^^xsd:duration && 
                ?timeDiff <= "P30D"^^xsd:duration)
    }
}
6. QUERY BUILDER IMPLEMENTATION
src/query_builder.py
"""
SPARQL Query Builder for PatientGraph
Based on patterns from GraphTalker tutorials.
"""

from typing import List, Optional, Dict
from datetime import datetime, timedelta

class PatientGraphQueryBuilder:
    """
    Generates SPARQL queries for extracting patient event sequences.
    
    Reference: GraphTalker tutorials for PatientGraph
    - synthea-exploration-tutorial.txt
    - standard-reporting-tutorial.txt
    - umls-sparql-tutorial.txt
    """
    
    PREFIXES = """
prefix pg: <http://patientgraph.ai/>
prefix : <http://patientgraph.ai/>
prefix skos: <http://www.w3.org/2004/02/skos/core#>
prefix umls: <https://uts.nlm.nih.gov/uts/umls/concept/>
prefix umls-scheme: <https://uts.nlm.nih.gov/uts/umls/vocabulary/2022AA/>
prefix franz: <http://franz.com/>
prefix xsd: <http://www.w3.org/2001/XMLSchema#>
prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
"""
    
    # Common medical codes
    CODES = {
        'diabetes_t2': ['44054006', '127013003', '90781000119102', '157141000119108'],
        'hypertension': ['59621000'],
        'hba1c': '4548-4',
        'office_visit': '162673000',
    }
    
    def __init__(self):
        pass
    
    def find_patients_with_condition(
        self,
        condition_codes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None
    ) -> str:
        """
        Find patients diagnosed with specific conditions.
        
        Pattern from: NQF0059 query, nested subselect
        
        Args:
            condition_codes: List of SNOMED codes (e.g., ['44054006'])
            start_date: Filter diagnosis date >= this (ISO format)
            end_date: Filter diagnosis date <= this
            age_min: Minimum patient age
            age_max: Maximum patient age
            
        Returns:
            SPARQL query string
        """
        codes_str = ' '.join([f'"{code}"' for code in condition_codes])
        
        query = f"""{self.PREFIXES}
SELECT DISTINCT ?patient ?diagnosisDate WHERE {{
    VALUES ?conditionCode {{ {codes_str} }}
    
    ?condition rdf:type :Condition; 
               :code ?code;
               :startDateTime ?diagnosisDate .
    ?code skos:notation ?conditionCode .
    ?patient rdf:type :Patient; 
             :patientCondition ?condition .
"""
        
        # Add age filter if specified
        if age_min is not None or age_max is not None:
            query += "    ?patient :patientAge ?age .\n"
            if age_min is not None:
                query += f"    FILTER (?age >= {age_min})\n"
            if age_max is not None:
                query += f"    FILTER (?age <= {age_max})\n"
        
        # Add date filters
        if start_date:
            query += f'    FILTER (?diagnosisDate >= "{start_date}T00:00:00+00:00"^^xsd:dateTime)\n'
        if end_date:
            query += f'    FILTER (?diagnosisDate <= "{end_date}T23:59:59+00:00"^^xsd:dateTime)\n'
        
        query += "}\nORDER BY ?patient"
        
        return query
    
    def get_patient_timeline(
        self,
        patient_uris: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """
        Extract chronological timeline of events for patients.
        
        Pattern from: standard-reporting-tutorial.txt, Section 4
        
        Args:
            patient_uris: List of patient URIs (if None, get all patients)
            event_types: List of event types to include 
                        (Condition, Medication, Observation, Procedure, Encounter)
            start_date: Only include events after this date
            end_date: Only include events before this date
            
        Returns:
            SPARQL query string with columns:
            ?patient, ?eventType, ?code, ?codeLabel, ?datetime, ?value, ?units
        """
        if event_types is None:
            event_types = ['Condition', 'Medication', 'Observation', 'Procedure', 'Encounter']
        
        query = f"{self.PREFIXES}\n"
        query += "SELECT ?patient ?eventType ?code ?codeLabel ?datetime ?value ?units WHERE {\n"
        
        # Optional patient filter
        if patient_uris:
            uris_str = ' '.join([f':{uri}' for uri in patient_uris])
            query += f"    VALUES ?patient {{ {uris_str} }}\n\n"
        
        # Build UNION clauses for each event type
        union_clauses = []
        
        if 'Condition' in event_types:
            union_clauses.append("""
    {
        ?patient :patientCondition ?event .
        ?event a :Condition ;
               :code ?codeResource ;
               :startDateTime ?datetime .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Condition" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }""")
        
        if 'Medication' in event_types:
            union_clauses.append("""
    {
        ?patient :patientMedication ?event .
        ?event a :Medication ;
               :code ?codeResource ;
               :startDateTime ?datetime .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Medication" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }""")
        
        if 'Observation' in event_types:
            union_clauses.append("""
    {
        ?patient :patientObservation ?event .
        ?event a :Observation ;
               :code ?codeResource ;
               :startDateTime ?datetime ;
               :value ?value .
        OPTIONAL { ?event :units ?units }
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Observation" AS ?eventType)
    }""")
        
        if 'Procedure' in event_types:
            union_clauses.append("""
    {
        ?patient :patientProcedure ?event .
        ?event a :Procedure ;
               :code ?codeResource ;
               :startDateTime ?datetime .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND("Procedure" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }""")
        
        if 'Encounter' in event_types:
            union_clauses.append("""
    {
        ?patient :patientEncounter ?event .
        ?event a :Encounter ;
               :code ?codeResource ;
               :startDateTime ?datetime ;
               :encounterclass ?encounterClass .
        ?codeResource skos:notation ?code ;
                      skos:prefLabel ?codeLabel .
        BIND(CONCAT("Encounter-", ?encounterClass) AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }""")
        
        query += "\n    UNION\n".join(union_clauses)
        query += "\n"
        
        # Add date filters
        if start_date or end_date:
            query += "    # Date filters\n"
            if start_date:
                query += f'    FILTER (?datetime >= "{start_date}T00:00:00+00:00"^^xsd:dateTime)\n'
            if end_date:
                query += f'    FILTER (?datetime <= "{end_date}T23:59:59+00:00"^^xsd:dateTime)\n'
        
        query += "}\nORDER BY ?patient ?datetime"
        
        return query
    
    def get_cohort_with_sequences(
        self,
        condition_codes: List[str],
        age_min: int = 18,
        age_max: int = 75,
        diagnosis_after: str = "2018-06-01",
        event_types: Optional[List[str]] = None
    ) -> str:
        """
        Combined query: Find cohort AND extract their event sequences.
        
        Pattern from: NQF0059 query (nested subselect technique)
        
        This is more efficient than running two separate queries.
        
        Args:
            condition_codes: SNOMED codes for cohort selection
            age_min, age_max: Age range
            diagnosis_after: Only include patients diagnosed after this date
            event_types: Which event types to extract
            
        Returns:
            SPARQL query with results ready for RNN preprocessing
        """
        codes_str = ' '.join([f'"{code}"' for code in condition_codes])
        
        if event_types is None:
            event_types = ['Condition', 'Medication', 'Observation', 'Procedure']
        
        query = f"""{self.PREFIXES}
SELECT ?patient ?eventType ?code ?codeLabel ?datetime ?value ?units WHERE {{
    {{
        # INNER QUERY: Find patients meeting cohort criteria
        SELECT DISTINCT ?patient WHERE {{
            VALUES ?conditionCode {{ {codes_str} }}
            
            ?condition rdf:type :Condition; 
                       :code ?codeRes;
                       :startDateTime ?diagnosisDate .
            ?codeRes skos:notation ?conditionCode .
            ?patient rdf:type :Patient; 
                     :patientCondition ?condition ;
                     :patientAge ?age .
            
            FILTER (?diagnosisDate > "{diagnosis_after}T00:00:00+00:00"^^xsd:dateTime)
            FILTER (?age >= {age_min} && ?age <= {age_max})
        }}
    }}
    
    # OUTER QUERY: For each qualifying patient, get all events
"""
        
        # Add event extraction (similar to get_patient_timeline)
        union_clauses = []
        
        if 'Condition' in event_types:
            union_clauses.append("""
    {
        ?patient :patientCondition ?event .
        ?event :code ?codeResource ; :startDateTime ?datetime .
        ?codeResource skos:notation ?code ; skos:prefLabel ?codeLabel .
        BIND("Condition" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }""")
        
        if 'Medication' in event_types:
            union_clauses.append("""
    {
        ?patient :patientMedication ?event .
        ?event :code ?codeResource ; :startDateTime ?datetime .
        ?codeResource skos:notation ?code ; skos:prefLabel ?codeLabel .
        BIND("Medication" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }""")
        
        if 'Observation' in event_types:
            union_clauses.append("""
    {
        ?patient :patientObservation ?event .
        ?event :code ?codeResource ; :startDateTime ?datetime ; :value ?value .
        OPTIONAL { ?event :units ?units }
        ?codeResource skos:notation ?code ; skos:prefLabel ?codeLabel .
        BIND("Observation" AS ?eventType)
    }""")
        
        if 'Procedure' in event_types:
            union_clauses.append("""
    {
        ?patient :patientProcedure ?event .
        ?event :code ?codeResource ; :startDateTime ?datetime .
        ?codeResource skos:notation ?code ; skos:prefLabel ?codeLabel .
        BIND("Procedure" AS ?eventType)
        BIND("" AS ?value)
        BIND("" AS ?units)
    }""")
        
        query += "\n    UNION\n".join(union_clauses)
        query += "\n}\nORDER BY ?patient ?datetime"
        
        return query
    
    def get_demographics(self, patient_uris: Optional[List[str]] = None) -> str:
        """
        Get patient demographics for feature engineering.
        
        Pattern from: standard-reporting-tutorial.txt
        """
        query = f"{self.PREFIXES}\n"
        query += "SELECT ?patient ?age ?gender ?city ?state ?birthdate WHERE {\n"
        query += "    ?patient a :Patient ;\n"
        query += "             :patientAge ?age ;\n"
        query += "             :gender ?gender ;\n"
        query += "             :city ?city ;\n"
        query += "             :state ?state ;\n"
        query += "             :birthdate ?birthdate .\n"
        
        if patient_uris:
            uris_str = ' '.join([f':{uri}' for uri in patient_uris])
            query += f"    VALUES ?patient {{ {uris_str} }}\n"
        
        query += "}"
        return query
7. DATA EXTRACTION PIPELINE
src/data_extractor.py
"""
Data extraction pipeline: SPARQL queries → pandas DataFrames → cached files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import pickle
from tqdm import tqdm

from allegrograph_client import AllegroGraphClient
from query_builder import PatientGraphQueryBuilder

class PatientDataExtractor:
    """
    Extracts patient event sequences from PatientGraph.
    """
    
    def __init__(
        self,
        ag_client: AllegroGraphClient,
        cache_dir: str = "data/raw"
    ):
        """
        Args:
            ag_client: AllegroGraph client instance
            cache_dir: Directory to cache query results
        """
        self.client = ag_client
        self.query_builder = PatientGraphQueryBuilder()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_cohort(
        self,
        condition_codes: List[str],
        age_min: int = 18,
        age_max: int = 75,
        diagnosis_after: str = "2018-06-01",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract patient cohort based on condition criteria.
        
        Returns:
            DataFrame with columns: [patient, diagnosisDate]
        """
        cache_file = self.cache_dir / "cohort.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached cohort from {cache_file}")
            return pd.read_csv(cache_file)
        
        print("Extracting patient cohort...")
        query = self.query_builder.find_patients_with_condition(
            condition_codes=condition_codes,
            start_date=diagnosis_after,
            age_min=age_min,
            age_max=age_max
        )
        
        df = self.client.query(query)
        
        if use_cache:
            df.to_csv(cache_file, index=False)
            print(f"Cached cohort to {cache_file}")
        
        print(f"Found {len(df)} patients in cohort")
        return df
    
    def extract_patient_sequences(
        self,
        patient_uris: List[str],
        event_types: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Extract event sequences for a list of patients.
        
        Args:
            patient_uris: List of patient URIs
            event_types: Which event types to extract
            start_date, end_date: Date range filter
            use_cache: Whether to cache results
            batch_size: Query this many patients at a time (avoid timeout)
            
        Returns:
            DataFrame with columns: 
            [patient, eventType, code, codeLabel, datetime, value, units]
        """
        cache_file = self.cache_dir / "sequences.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached sequences from {cache_file}")
            return pd.read_csv(cache_file)
        
        print(f"Extracting sequences for {len(patient_uris)} patients...")
        
        # Process in batches to avoid query timeout
        all_sequences = []
        
        for i in tqdm(range(0, len(patient_uris), batch_size)):
            batch = patient_uris[i:i+batch_size]
            
            query = self.query_builder.get_patient_timeline(
                patient_uris=batch,
                event_types=event_types,
                start_date=start_date,
                end_date=end_date
            )
            
            df_batch = self.client.query(query)
            all_sequences.append(df_batch)
        
        df = pd.concat(all_sequences, ignore_index=True)
        
        if use_cache:
            df.to_csv(cache_file, index=False)
            print(f"Cached sequences to {cache_file}")
        
        print(f"Extracted {len(df)} events")
        return df
    
    def extract_demographics(
        self,
        patient_uris: List[str],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Extract patient demographics.
        
        Returns:
            DataFrame with columns: [patient, age, gender, city, state, birthdate]
        """
        cache_file = self.cache_dir / "demographics.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached demographics from {cache_file}")
            return pd.read_csv(cache_file)
        
        print("Extracting demographics...")
        query = self.query_builder.get_demographics(patient_uris)
        df = self.client.query(query)
        
        if use_cache:
            df.to_csv(cache_file, index=False)
            print(f"Cached demographics to {cache_file}")
        
        return df
    
    def extract_cohort_with_sequences(
        self,
        condition_codes: List[str],
        age_min: int = 18,
        age_max: int = 75,
        diagnosis_after: str = "2018-06-01",
        event_types: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Combined extraction: cohort selection + sequence extraction in one query.
        
        More efficient than two separate queries.
        
        Returns:
            DataFrame with event sequences for qualifying patients
        """
        cache_file = self.cache_dir / "cohort_sequences.csv"
        
        if use_cache and cache_file.exists():
            print(f"Loading cached cohort sequences from {cache_file}")
            return pd.read_csv(cache_file)
        
        print("Extracting cohort with sequences (combined query)...")
        query = self.query_builder.get_cohort_with_sequences(
            condition_codes=condition_codes,
            age_min=age_min,
            age_max=age_max,
            diagnosis_after=diagnosis_after,
            event_types=event_types
        )
        
        df = self.client.query(query)
        
        if use_cache:
            df.to_csv(cache_file, index=False)
            print(f"Cached cohort sequences to {cache_file}")
        
        print(f"Extracted {len(df)} events for {df['patient'].nunique()} patients")
        return df
8. SEQUENCE PREPROCESSING
src/sequence_preprocessor.py
"""
Preprocess event sequences for RNN training.
"""

import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path

class SequencePreprocessor:
    """
    Converts patient event DataFrames into tensors suitable for RNN training.
    """
    
    def __init__(
        self,
        max_sequence_length: int = 100,
        code_embedding_dim: int = 64,
        cache_dir: str = "data/processed"
    ):
        """
        Args:
            max_sequence_length: Pad/truncate sequences to this length
            code_embedding_dim: Dimension of learned code embeddings
            cache_dir: Directory to save preprocessed tensors
        """
        self.max_seq_len = max_sequence_length
        self.code_emb_dim = code_embedding_dim
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Encoders (fitted during preprocessing)
        self.event_type_encoder = LabelEncoder()
        self.code_encoder = LabelEncoder()
        self.value_stats = {}  # For normalizing observation values
        
    def preprocess_sequences(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Main preprocessing pipeline.
        
        Args:
            df: DataFrame with columns [patient, eventType, code, datetime, value]
            target_column: Optional column name for prediction target
            
        Returns:
            Dictionary with:
            - 'sequences': [num_patients, max_seq_len, feature_dim]
            - 'lengths': [num_patients] - actual sequence length per patient
            - 'patient_ids': [num_patients] - patient identifiers
            - 'targets': [num_patients] - prediction targets (if provided)
        """
        print("Preprocessing sequences...")
        
        # 1. Convert datetime to pandas datetime
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 2. Sort by patient and time
        df = df.sort_values(['patient', 'datetime'])
        
        # 3. Group by patient
        patient_groups = df.groupby('patient')
        
        patient_ids = []
        sequences = []
        lengths = []
        targets = []
        
        for patient_id, group in patient_groups:
            # Truncate if too long
            if len(group) > self.max_seq_len:
                group = group.iloc[-self.max_seq_len:]  # Keep most recent events
            
            # Convert to feature vectors
            seq_features = self._events_to_features(group)
            
            patient_ids.append(patient_id)
            sequences.append(seq_features)
            lengths.append(len(group))
            
            if target_column and target_column in group.columns:
                # For simplicity, use the target from the last event
                targets.append(group[target_column].iloc[-1])
        
        # 4. Pad sequences to max length
        sequences_padded = self._pad_sequences(sequences)
        
        # 5. Convert to tensors
        result = {
            'sequences': torch.FloatTensor(sequences_padded),
            'lengths': torch.LongTensor(lengths),
            'patient_ids': patient_ids
        }
        
        if targets:
            result['targets'] = torch.FloatTensor(targets)
        
        print(f"Preprocessed {len(patient_ids)} patients")
        print(f"Sequence shape: {result['sequences'].shape}")
        
        return result
    
    def _events_to_features(self, events: pd.DataFrame) -> np.ndarray:
        """
        Convert a sequence of events to feature vectors.
        
        Each event becomes a vector: [event_type_onehot, code_id, time_delta, value_normalized]
        
        Args:
            events: DataFrame for one patient's events
            
        Returns:
            numpy array of shape [seq_len, feature_dim]
        """
        features = []
        
        # Get reference time (first event)
        first_time = events['datetime'].iloc[0]
        
        for idx, row in events.iterrows():
            # 1. Event type (one-hot encoded)
            event_type = row['eventType']
            event_type_id = self._encode_event_type(event_type)
            
            # 2. Medical code (integer ID for embedding lookup later)
            code = row['code']
            code_id = self._encode_code(code)
            
            # 3. Temporal feature: days since first event
            time_delta = (row['datetime'] - first_time).days
            
            # 4. Value (for observations, normalized)
            value = 0.0
            if pd.notna(row['value']) and row['value'] != '':
                try:
                    value = float(row['value'])
                    # Normalize using pre-computed stats
                    value = self._normalize_value(code, value)
                except:
                    value = 0.0
            
            # Combine features
            feature_vector = [
                event_type_id,
                code_id,
                time_delta,
                value
            ]
            
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _encode_event_type(self, event_type: str) -> int:
        """Encode event type as integer."""
        if not hasattr(self.event_type_encoder, 'classes_'):
            # First time - fit encoder
            return 0
        
        try:
            return self.event_type_encoder.transform([event_type])[0]
        except:
            return 0  # Unknown event type
    
    def _encode_code(self, code: str) -> int:
        """Encode medical code as integer (for embedding lookup)."""
        if not hasattr(self.code_encoder, 'classes_'):
            return 0
        
        try:
            return self.code_encoder.transform([code])[0]
        except:
            return 0  # Unknown code
    
    def _normalize_value(self, code: str, value: float) -> float:
        """Normalize observation value using z-score."""
        if code not in self.value_stats:
            return value  # No normalization available
        
        mean = self.value_stats[code]['mean']
        std = self.value_stats[code]['std']
        
        if std == 0:
            return 0.0
        
        return (value - mean) / std
    
    def _pad_sequences(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Pad sequences to max_sequence_length.
        
        Args:
            sequences: List of arrays with shape [seq_len, feature_dim]
            
        Returns:
            Padded array of shape [num_sequences, max_seq_len, feature_dim]
        """
        feature_dim = sequences[0].shape[1]
        padded = np.zeros((len(sequences), self.max_seq_len, feature_dim), dtype=np.float32)
        
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq), self.max_seq_len)
            padded[i, :seq_len, :] = seq[:seq_len]
        
        return padded
    
    def fit_encoders(self, df: pd.DataFrame):
        """
        Fit label encoders and compute value statistics.
        
        Call this on training data before preprocessing.
        
        Args:
            df: Training data DataFrame
        """
        print("Fitting encoders on training data...")
        
        # Fit event type encoder
        event_types = df['eventType'].unique()
        self.event_type_encoder.fit(event_types)
        print(f"Event types: {list(self.event_type_encoder.classes_)}")
        
        # Fit code encoder
        codes = df['code'].unique()
        self.code_encoder.fit(codes)
        print(f"Unique codes: {len(self.code_encoder.classes_)}")
        
        # Compute value statistics per code
        obs_df = df[df['value'].notna() & (df['value'] != '')]
        obs_df['value_float'] = pd.to_numeric(obs_df['value'], errors='coerce')
        obs_df = obs_df[obs_df['value_float'].notna()]
        
        for code in obs_df['code'].unique():
            code_values = obs_df[obs_df['code'] == code]['value_float']
            self.value_stats[code] = {
                'mean': code_values.mean(),
                'std': code_values.std(),
                'min': code_values.min(),
                'max': code_values.max()
            }
        
        print(f"Computed value statistics for {len(self.value_stats)} codes")
    
    def save_encoders(self, path: str = "data/processed/encoders.pkl"):
        """Save fitted encoders and stats for later use."""
        data = {
            'event_type_encoder': self.event_type_encoder,
            'code_encoder': self.code_encoder,
            'value_stats': self.value_stats
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved encoders to {path}")
    
    def load_encoders(self, path: str = "data/processed/encoders.pkl"):
        """Load previously fitted encoders."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.event_type_encoder = data['event_type_encoder']
        self.code_encoder = data['code_encoder']
        self.value_stats = data['value_stats']
        print(f"Loaded encoders from {path}")
    
    def get_feature_dim(self) -> int:
        """Return the dimensionality of feature vectors."""
        # [event_type_id, code_id, time_delta, value]
        return 4
    
    def get_num_codes(self) -> int:
        """Return number of unique medical codes (for embedding layer size)."""
        if hasattr(self.code_encoder, 'classes_'):
            return len(self.code_encoder.classes_)
        return 0
9. RNN MODEL ARCHITECTURE
src/models.py
"""
RNN model architectures for patient sequence prediction.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional

class PatientLSTM(nn.Module):
    """
    LSTM model for patient event sequence prediction.
    
    Architecture:
    1. Embedding layer for medical codes
    2. LSTM layers to process sequences
    3. Fully connected layer for prediction
    """
    
    def __init__(
        self,
        num_codes: int,
        code_embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        bidirectional: bool = False
    ):
        """
        Args:
            num_codes: Vocabulary size for medical codes
            code_embedding_dim: Dimension of code embeddings
            hidden_size: LSTM hidden state dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary classification)
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.num_codes = num_codes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer for medical codes
        self.code_embedding = nn.Embedding(
            num_embeddings=num_codes,
            embedding_dim=code_embedding_dim,
            padding_idx=0
        )
        
        # Input dimension: code_embedding + other features (event_type, time, value)
        input_dim = code_embedding_dim + 3  # [code_emb, event_type_id, time_delta, value]
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, num_classes)
    
    def forward(
        self,
        sequences: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sequences: [batch_size, max_seq_len, 4]
                       Features: [event_type_id, code_id, time_delta, value]
            lengths: [batch_size] - actual sequence lengths
            
        Returns:
            Predictions: [batch_size, num_classes]
        """
        batch_size, max_seq_len, _ = sequences.shape
        
        # Split features
        event_type_ids = sequences[:, :, 0].long()  # [batch, seq]
        code_ids = sequences[:, :, 1].long()  # [batch, seq]
        time_deltas = sequences[:, :, 2:3]  # [batch, seq, 1]
        values = sequences[:, :, 3:4]  # [batch, seq, 1]
        
        # Embed codes
        code_embeds = self.code_embedding(code_ids)  # [batch, seq, code_emb_dim]
        
        # Concatenate all features
        # [batch, seq, code_emb_dim + 3]
        lstm_input = torch.cat([
            code_embeds,
            event_type_ids.unsqueeze(-1).float(),
            time_deltas,
            values
        ], dim=-1)
        
        # Pack padded sequences for efficient LSTM processing
        packed_input = pack_padded_sequence(
            lstm_input,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Use final hidden state for prediction
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Dropout and final prediction
        hidden = self.dropout(hidden)
        output = self.fc(hidden)  # [batch, num_classes]
        
        return output


class PatientGRU(nn.Module):
    """
    GRU model (simpler alternative to LSTM, often similar performance).
    """
    
    def __init__(
        self,
        num_codes: int,
        code_embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.code_embedding = nn.Embedding(num_codes, code_embedding_dim, padding_idx=0)
        
        input_dim = code_embedding_dim + 3
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, max_seq_len, _ = sequences.shape
        
        code_ids = sequences[:, :, 1].long()
        code_embeds = self.code_embedding(code_ids)
        
        gru_input = torch.cat([
            code_embeds,
            sequences[:, :, 0:1],  # event_type
            sequences[:, :, 2:3],  # time_delta
            sequences[:, :, 3:4]   # value
        ], dim=-1)
        
        packed_input = pack_padded_sequence(
            gru_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, hidden = self.gru(packed_input)
        hidden = self.dropout(hidden[-1])
        output = self.fc(hidden)
        
        return output
10. TRAINING PIPELINE
src/train.py
"""
Training pipeline for patient sequence models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
from pathlib import Path

class Trainer:
    """
    Handles model training with early stopping and checkpointing.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ):
        """
        Args:
            model: PyTorch model to train
            device: 'cuda' or 'cpu'
            learning_rate: Optimizer learning rate
            weight_decay: L2 regularization strength
        """
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()  # For classification
        # For regression, use: nn.MSELoss()
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            sequences, lengths, targets = batch
            sequences = sequences.to(self.device)
            lengths = lengths.to(self.device)
            targets = targets.to(self.device).long()  # For classification
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences, lengths)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Dictionary with 'loss' and 'accuracy'
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, lengths, targets = batch
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                targets = targets.to(self.device).long()
                
                outputs = self.model(sequences, lengths)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        patience: int = 10,
        save_dir: str = "models"
    ):
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            save_dir: Directory to save model checkpoints
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Accuracy: {train_metrics['accuracy']:.4f}")
            
            # Validate
            val_metrics = self.validate(val_loader)
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                torch.save(
                    self.model.state_dict(),
                    save_dir / "best_model.pt"
                )
                print("✓ Saved best model")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{patience}")
                
                if self.patience_counter >= patience:
                    print("Early stopping triggered")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load(save_dir / "best_model.pt"))
        print("\nTraining complete. Loaded best model.")
11. EVALUATION & METRICS
src/evaluate.py
"""
Evaluation metrics for patient sequence prediction models.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Computes evaluation metrics for trained models.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    def predict(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on test set.
        
        Returns:
            (true_labels, predicted_labels, predicted_probabilities)
        """
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequences, lengths, targets = batch
                sequences = sequences.to(self.device)
                lengths = lengths.to(self.device)
                
                outputs = self.model(sequences, lengths)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return (
            np.array(all_targets),
            np.array(all_predictions),
            np.array(all_probabilities)
        )
    
    def evaluate(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Returns:
            Dictionary of metrics
        """
        y_true, y_pred, y_prob = self.predict(test_loader)
        
        # Binary classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        }
        
        # ROC-AUC (using probability of positive class)
        if y_prob.shape[1] == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            metrics['pr_auc'] = average_precision_score(y_true, y_prob[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Specificity and Sensitivity
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def print_report(self, metrics: Dict[str, float]):
        """Print evaluation metrics in readable format."""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION METRICS")
        print("=" * 50)
        
        print(f"\nAccuracy:    {metrics['accuracy']:.4f}")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"F1 Score:    {metrics['f1']:.4f}")
        
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
            print(f"PR-AUC:      {metrics['pr_auc']:.4f}")
        
        print(f"\nSensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print("=" * 50 + "\n")
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
12. EXAMPLE WORKFLOWS
Example 1: Predict 30-Day Hospitalization Risk for Diabetic Patients
# example_hospitalization_prediction.py

import yaml
from allegrograph_client import AllegroGraphClient
from data_extractor import PatientDataExtractor
from sequence_preprocessor import SequencePreprocessor
from models import PatientLSTM
from train import Trainer
from evaluate import ModelEvaluator
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 1. Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Connect to AllegroGraph
client = AllegroGraphClient(
    url=config['allegrograph']['url'],
    repository=config['allegrograph']['repository']
)
client.test_connection()

# 3. Extract data
extractor = PatientDataExtractor(client)

# Get diabetic patient cohort with their sequences
df = extractor.extract_cohort_with_sequences(
    condition_codes=['44054006', '127013003'],  # Type 2 DM codes
    age_min=18,
    age_max=75,
    diagnosis_after="2018-06-01",
    event_types=['Condition', 'Medication', 'Observation', 'Procedure'],
    use_cache=True
)

print(f"Extracted {len(df)} events for {df['patient'].nunique()} patients")

# 4. Create target labels (hospitalization in next 30 days)
# (This requires additional SPARQL query - simplified here)
# For demo, randomly assign labels
patients = df['patient'].unique()
targets = {p: np.random.randint(0, 2) for p in patients}  # 0 or 1

# Add targets to dataframe
df['hospitalization_30d'] = df['patient'].map(targets)

# 5. Train/val/test split
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

# 6. Preprocess sequences
preprocessor = SequencePreprocessor(
    max_sequence_length=100,
    code_embedding_dim=64
)

# Fit encoders on training data
preprocessor.fit_encoders(train_df)
preprocessor.save_encoders()

# Preprocess each split
train_data = preprocessor.preprocess_sequences(train_df, target_column='hospitalization_30d')
val_data = preprocessor.preprocess_sequences(val_df, target_column='hospitalization_30d')
test_data = preprocessor.preprocess_sequences(test_df, target_column='hospitalization_30d')

# 7. Create DataLoaders
train_dataset = TensorDataset(
    train_data['sequences'],
    train_data['lengths'],
    train_data['targets'].long()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(
    val_data['sequences'],
    val_data['lengths'],
    val_data['targets'].long()
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = TensorDataset(
    test_data['sequences'],
    test_data['lengths'],
    test_data['targets'].long()
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 8. Create model
num_codes = preprocessor.get_num_codes()
model = PatientLSTM(
    num_codes=num_codes,
    code_embedding_dim=64,
    hidden_size=128,
    num_layers=2,
    dropout=0.3,
    num_classes=2,  # Binary: hospitalized or not
    bidirectional=False
)

print(f"\nModel architecture:")
print(model)

# 9. Train model
trainer = Trainer(
    model=model,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    learning_rate=0.001
)

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    patience=10,
    save_dir="models"
)

# 10. Evaluate on test set
evaluator = ModelEvaluator(model, device=trainer.device)
metrics = evaluator.evaluate(test_loader)
evaluator.print_report(metrics)
evaluator.plot_confusion_matrix(metrics['confusion_matrix'], save_path='confusion_matrix.png')

print("\nTraining complete! Model saved to models/best_model.pt")
Example 2: Jupyter Notebook Workflow
# notebooks/1_data_exploration.ipynb

# Cell 1: Setup
import sys
sys.path.append('../src')

from allegrograph_client import AllegroGraphClient
from query_builder import PatientGraphQueryBuilder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cell 2: Connect
client = AllegroGraphClient(
    url="http://localhost:10035",
    repository="PatientGraph"
)
client.test_connection()

# Cell 3: Explore patient count
query = """
PREFIX : <http://patientgraph.ai/>
SELECT (COUNT(DISTINCT ?patient) AS ?count) WHERE {
    ?patient a :Patient .
}
"""
result = client.query(query)
print(f"Total patients: {result['count'][0]}")

# Cell 4: Find diabetic patients
qb = PatientGraphQueryBuilder()
query = qb.find_patients_with_condition(
    condition_codes=['44054006'],  # Type 2 DM
    age_min=18,
    age_max=75
)
df_cohort = client.query(query)
print(f"Diabetic patients: {len(df_cohort)}")

# Cell 5: Analyze event distribution
patient_sample = df_cohort['patient'].iloc[:10].tolist()
query = qb.get_patient_timeline(
    patient_uris=patient_sample,
    event_types=['Condition', 'Medication', 'Observation']
)
df_timeline = client.query(query)

# Plot event type distribution
df_timeline['eventType'].value_counts().plot(kind='bar')
plt.title('Event Type Distribution')
plt.xlabel('Event Type')
plt.ylabel('Count')
plt.show()

# Cell 6: Analyze sequence lengths
seq_lengths = df_timeline.groupby('patient').size()
seq_lengths.hist(bins=50)
plt.title('Sequence Length Distribution')
plt.xlabel('Number of Events')
plt.ylabel('Number of Patients')
plt.show()

print(f"Mean sequence length: {seq_lengths.mean():.1f}")
print(f"Median sequence length: {seq_lengths.median():.1f}")
print(f"Max sequence length: {seq_lengths.max()}")
13. TROUBLESHOOTING
Common Issues
1. Query Timeout

Problem: SPARQL query times out for large cohorts
Solution: Process patients in batches (see batch_size parameter in extract_patient_sequences)
2. Out of Memory

Problem: Model training runs out of memory
Solutions:
- Reduce batch_size
- Reduce max_sequence_length
- Reduce hidden_size or num_layers
- Use gradient accumulation
3. Vanishing Gradients

Problem: Model loss stops decreasing
Solutions:
- Use LSTM instead of vanilla RNN
- Reduce num_layers
- Add gradient clipping (already in Trainer)
- Reduce sequence length
4. Overfitting

Problem: Training accuracy high, validation accuracy low
Solutions:
- Increase dropout
- Add weight_decay (L2 regularization)
- Use more training data
- Reduce model capacity (smaller hidden_size)
5. Class Imbalance

Problem: Rare events (e.g., hospitalizations) are underpredictedrior
Solutions:
- Use weighted loss function: nn.CrossEntropyLoss(weight=class_weights)
- Oversample minority class
- Use SMOTE (Synthetic Minority Over-sampling Technique)
Debugging Tips
# Check data shapes
print("Sequence shape:", train_data['sequences'].shape)
print("Lengths shape:", train_data['lengths'].shape)
print("Targets shape:", train_data['targets'].shape)

# Verify no NaN values
print("NaN in sequences:", torch.isnan(train_data['sequences']).any())

# Check label distribution
print("Label distribution:", torch.bincount(train_data['targets'].long()))

# Visualize a single patient sequence
patient_idx = 0
patient_seq = train_data['sequences'][patient_idx]
seq_len = train_data['lengths'][patient_idx]
print(f"Patient {patient_idx} has {seq_len} events")
print(patient_seq[:seq_len])
14. REFERENCES
PatientGraph / GraphTalker Tutorials
The SPARQL patterns in this guide are extracted from these PatientGraph tutorials (available in GraphTalker):

synthea-exploration-tutorial.txt

Patient data queries (conditions, medications, observations)
Knowledge-driven queries with franz:narrower-inclusive-of
Linking UMLS concepts to SNOMED codes
standard-reporting-tutorial.txt

Temporal filtering with date ranges
Demographic queries (age, gender, city)
Chronological patient timelines
umls-sparql-tutorial.txt

UMLS concept hierarchies
Cross-vocabulary mapping (UMLS ↔ SNOMED ↔ RXNORM ↔ LOINC)
Semantic network types
afib-queries-tutorial.txt

Magic predicates (:withCondition, :demographix)
Pre-computed risk scores
Medication tracking patterns
NQF0059 Query (stored in Query Library)

Nested subselect technique
Cohort selection with multiple criteria
Most recent observation per patient
Deep Learning for Healthcare
Key Papers:

Choi et al. (2016): "RETAIN: Interpretable Predictive Model in Healthcare" - Attention-based RNN for EHR
Rajkomar et al. (2018): "Scalable and accurate deep learning with electronic health records" - Google's EHR model
Miotto et al. (2016): "Deep Patient: An Unsupervised Representation to Predict the Future of Patients from the Electronic Health Records" - Denoising autoencoders
Medical Code Embeddings:

Choi et al. (2016): "Multi-layer Representation Learning for Medical Concepts" (Med2Vec)
Beam et al. (2020): "Clinical Concept Embeddings Learned from Massive Sources of Multimodal Medical Data"
PyTorch Resources
Official Tutorials: https://pytorch.org/tutorials/
RNN Tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
pack_padded_sequence: https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
AllegroGraph Documentation
SPARQL Reference: https://franz.com/agraph/support/documentation/current/sparql-reference.html
REST API: https://franz.com/agraph/support/documentation/current/http-protocol.html
QUICK START CHECKLIST
[ ] Install Python 3.10+ and create conda/venv environment
[ ] pip install -r requirements.txt
[ ] Verify AllegroGraph connection (test_connection())
[ ] Run exploratory queries in Jupyter notebook
[ ] Extract diabetic patient cohort
[ ] Extract patient event sequences
[ ] Fit encoders on training data
[ ] Preprocess sequences into tensors
[ ] Create train/val/test DataLoaders
[ ] Initialize LSTM model
[ ] Train model with early stopping
[ ] Evaluate on test set
[ ] Analyze predictions and errors
[ ] Iterate on features and architecture
NEXT STEPS
Phase 1: Basic Pipeline

Set up environment and verify AllegroGraph connection
Extract a small cohort (100 patients) for testing
Preprocess sequences and verify tensor shapes
Train a simple LSTM model
Achieve >50% accuracy on a binary prediction task
Phase 2: Feature Engineering

Add more temporal features (time-of-day, day-of-week)
Include patient demographics as static features
Experiment with code embeddings (random vs. pre-trained)
Add attention mechanisms to highlight important events
Phase 3: Advanced Models

Bidirectional LSTM
Multi-task learning (predict multiple outcomes simultaneously)
Hierarchical RNN (model visit-level and patient-level sequences)
Transformer models (self-attention instead of recurrence)
Phase 4: Clinical Validation

Collaborate with clinicians to define meaningful prediction tasks
Validate predictions against clinical guidelines
Analyze model errors and failure modes
Deploy model for prospective validation
CONTACT & SUPPORT
This guide was created to bridge PatientGraph/GraphTalker (RDF knowledge graph) with modern deep learning (PyTorch RNNs).

For questions:

AllegroGraph/SPARQL: Refer to PatientGraph tutorials in GraphTalker
PyTorch/RNNs: Check PyTorch forums and documentation
Clinical concepts: Consult UMLS documentation
Happy modeling! 🚀

End of PatientGraph RNN Toolkit Guide

There you go! This is a complete, self-contained markdown document (approximately 20,000 words) that you can save in your Claude Code project. It includes:

✅ Complete project structure
✅ AllegroGraph connection code
✅ 10+ SPARQL pattern templates with detailed explanations
✅ Full Python implementations (query builder, data extractor, preprocessor, models, trainer, evaluator)
✅ Example workflows and Jupyter notebook code
✅ Troubleshooting guide
✅ References to PatientGraph tutorials
You can now:

Copy this entire markdown into a file called README.md or PatientGraph_RNN_Guide.md
Start your Claude Code project with this as your reference
Extract specific sections (e.g., just the SPARQL patterns) into separate files as needed
Would you like me to also create:

A separate requirements.txt file?
Individual .py files for each module?
A sample config.yaml?
A starter Jupyter notebook?
Let me know what else would be helpful!