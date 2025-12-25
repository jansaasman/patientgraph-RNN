"""
SPARQL Query Templates for RNN Data Extraction.

These templates are generic and can be customized for any disease cohort.
Based on recommendations from GraphTalker analysis of PatientGraph.
"""

from typing import Optional, List


# Common prefixes for all queries
PREFIXES = """
PREFIX ns28: <http://patientgraph.ai/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def patient_base_query(
    cohort_filter: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """
    Query 1: Patient Base Table - Demographics, outcomes, observation window.

    Returns one row per patient with static features.

    Args:
        cohort_filter: Optional WHERE clause to filter patients (e.g., diabetics only)
        limit: Optional limit on number of patients

    Returns:
        SPARQL query string
    """
    cohort_clause = f"\n    {cohort_filter}" if cohort_filter else ""
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    # Simplified query without complex aggregations
    return f"""{PREFIXES}
SELECT DISTINCT
    ?patientId
    ?gender
    ?race
    ?ethnicity
    ?birthdate
    ?deathdate
WHERE {{
    ?patient a ns28:Patient ;
             rdfs:label ?patientId .
    {cohort_clause}
    # Demographics
    OPTIONAL {{ ?patient ns28:gender ?gender }}
    OPTIONAL {{ ?patient ns28:race ?race }}
    OPTIONAL {{ ?patient ns28:ethnicity ?ethnicity }}
    OPTIONAL {{ ?patient ns28:birthdate ?birthdate }}
    OPTIONAL {{ ?patient ns28:deathdate ?deathdate }}
}}
ORDER BY ?patientId
{limit_clause}
"""


def event_sequence_query(
    patient_ids: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> str:
    """
    Query 2: Complete Event Sequence - All events unified in chronological order.

    This is the core RNN input - one row per event, ordered by patient + time.

    Args:
        patient_ids: Optional list of patient IDs to filter
        event_types: Event types to include (default: all)
        limit: Optional limit on total events

    Returns:
        SPARQL query string
    """
    # Patient filter
    if patient_ids:
        patient_values = " ".join([f'"{pid}"' for pid in patient_ids])
        patient_filter = f"VALUES ?patientId {{ {patient_values} }}"
    else:
        patient_filter = ""

    # Default event types
    if event_types is None:
        event_types = ["CONDITION", "MEDICATION", "OBSERVATION", "PROCEDURE", "IMMUNIZATION"]

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    # Build UNION blocks for each event type
    union_blocks = []

    if "CONDITION" in event_types:
        union_blocks.append("""
        {
            # CONDITION events
            ?patient ns28:patientCondition ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
                ?codeUri skos:prefLabel ?eventDescription .
            }
            BIND("CONDITION" AS ?eventType)
            BIND("" AS ?eventValue)
            BIND("" AS ?eventUnits)
        }""")

    if "MEDICATION" in event_types:
        union_blocks.append("""
        {
            # MEDICATION events
            ?patient ns28:patientMedication ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
                ?codeUri skos:prefLabel ?eventDescription .
            }
            OPTIONAL { ?event ns28:dispenses ?dispenses }
            BIND("MEDICATION" AS ?eventType)
            BIND(STR(COALESCE(?dispenses, "")) AS ?eventValue)
            BIND("fills" AS ?eventUnits)
        }""")

    if "OBSERVATION" in event_types:
        union_blocks.append("""
        {
            # OBSERVATION events (labs, vitals)
            ?patient ns28:patientObservation ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
                ?codeUri skos:prefLabel ?eventDescription .
            }
            OPTIONAL { ?event ns28:value ?obsValue }
            OPTIONAL { ?event ns28:units ?obsUnits }
            BIND("OBSERVATION" AS ?eventType)
            BIND(STR(COALESCE(?obsValue, "")) AS ?eventValue)
            BIND(STR(COALESCE(?obsUnits, "")) AS ?eventUnits)
        }""")

    if "PROCEDURE" in event_types:
        union_blocks.append("""
        {
            # PROCEDURE events
            ?patient ns28:patientProcedure ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL { ?event ns28:code ?eventCode }
            OPTIONAL { ?event ns28:description ?eventDescription }
            BIND("PROCEDURE" AS ?eventType)
            BIND("" AS ?eventValue)
            BIND("" AS ?eventUnits)
        }""")

    if "IMMUNIZATION" in event_types:
        union_blocks.append("""
        {
            # IMMUNIZATION events
            ?patient ns28:patientImmunization ?event .
            ?event ns28:startDateTime ?eventDateTime .
            OPTIONAL {
                ?event ns28:code ?codeUri .
                ?codeUri skos:notation ?eventCode .
                ?codeUri skos:prefLabel ?eventDescription .
            }
            BIND("IMMUNIZATION" AS ?eventType)
            BIND("" AS ?eventValue)
            BIND("" AS ?eventUnits)
        }""")

    union_clause = "\n    UNION\n    ".join(union_blocks)

    return f"""{PREFIXES}
SELECT DISTINCT
    ?patientId
    ?eventDateTime
    ?eventType
    ?eventCode
    ?eventDescription
    ?eventValue
    ?eventUnits
WHERE {{
    ?patient a ns28:Patient ;
             rdfs:label ?patientId .
    {patient_filter}

    {union_clause}
}}
ORDER BY ?patientId ?eventDateTime
{limit_clause}
"""


def vocabulary_query(limit: Optional[int] = None) -> str:
    """
    Query 3: Event Vocabulary - All unique codes with frequencies.

    Used to build embedding lookup tables.

    Args:
        limit: Optional limit on vocabulary size

    Returns:
        SPARQL query string
    """
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""{PREFIXES}
SELECT ?eventType ?code ?description (COUNT(*) AS ?frequency)
WHERE {{
    {{
        # Condition codes
        ?condition a ns28:Condition ;
                   ns28:code ?codeUri .
        ?codeUri skos:notation ?code .
        OPTIONAL {{ ?codeUri skos:prefLabel ?description }}
        BIND("CONDITION" AS ?eventType)
    }}
    UNION
    {{
        # Medication codes
        ?medication a ns28:Medication ;
                    ns28:code ?codeUri .
        ?codeUri skos:notation ?code .
        OPTIONAL {{ ?codeUri skos:prefLabel ?description }}
        BIND("MEDICATION" AS ?eventType)
    }}
    UNION
    {{
        # Observation codes
        ?observation a ns28:Observation ;
                     ns28:code ?codeUri .
        ?codeUri skos:notation ?code .
        OPTIONAL {{ ?codeUri skos:prefLabel ?description }}
        BIND("OBSERVATION" AS ?eventType)
    }}
    UNION
    {{
        # Procedure codes
        ?procedure a ns28:Procedure ;
                   ns28:code ?code .
        OPTIONAL {{ ?procedure ns28:description ?description }}
        BIND("PROCEDURE" AS ?eventType)
    }}
    UNION
    {{
        # Immunization codes
        ?immunization a ns28:Immunization ;
                      ns28:code ?codeUri .
        ?codeUri skos:notation ?code .
        OPTIONAL {{ ?codeUri skos:prefLabel ?description }}
        BIND("IMMUNIZATION" AS ?eventType)
    }}
}}
GROUP BY ?eventType ?code ?description
ORDER BY ?eventType DESC(?frequency)
{limit_clause}
"""


def observation_timeseries_query(
    patient_ids: Optional[List[str]] = None,
    observation_codes: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> str:
    """
    Query 5: Observation Time Series - Numeric labs/vitals over time.

    Critical for modeling physiological state changes.

    Args:
        patient_ids: Optional list of patient IDs to filter
        observation_codes: Optional list of LOINC codes to filter
        limit: Optional limit

    Returns:
        SPARQL query string
    """
    # Patient filter
    if patient_ids:
        patient_values = " ".join([f'"{pid}"' for pid in patient_ids])
        patient_filter = f"VALUES ?patientId {{ {patient_values} }}"
    else:
        patient_filter = ""

    # Observation code filter
    if observation_codes:
        code_values = " ".join([f'"{code}"' for code in observation_codes])
        code_filter = f"VALUES ?obsCode {{ {code_values} }}"
    else:
        code_filter = ""

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""{PREFIXES}
SELECT DISTINCT
    ?patientId
    ?observationDateTime
    ?obsCode
    ?obsDescription
    ?numericValue
    ?units
WHERE {{
    ?patient a ns28:Patient ;
             rdfs:label ?patientId .
    {patient_filter}

    ?patient ns28:patientObservation ?observation .
    ?observation ns28:code ?codeUri ;
                 ns28:startDateTime ?observationDateTime ;
                 ns28:value ?value .

    OPTIONAL {{ ?observation ns28:units ?units }}

    ?codeUri skos:notation ?obsCode .
    OPTIONAL {{ ?codeUri skos:prefLabel ?obsDescription }}
    {code_filter}

    # Filter to numeric values only
    FILTER(DATATYPE(?value) IN (xsd:decimal, xsd:float, xsd:double, xsd:integer)
           || REGEX(STR(?value), "^-?[0-9]+(\\\\.[0-9]+)?$"))

    BIND(xsd:float(?value) AS ?numericValue)
}}
ORDER BY ?patientId ?observationDateTime ?obsCode
{limit_clause}
"""


def outcome_labels_query(
    outcome_type: str = "hospitalization",
    prediction_horizon_days: int = 30,
    cohort_filter: Optional[str] = None,
    limit: Optional[int] = None
) -> str:
    """
    Query 6: Outcome Labels - Flexible template for any prediction task.

    Args:
        outcome_type: Type of outcome to predict:
            - "hospitalization": Inpatient encounter
            - "mortality": Death
            - "complication": Disease complication (requires outcome_patterns in cohort_filter)
        prediction_horizon_days: Days ahead to predict
        cohort_filter: Additional WHERE clause for cohort selection
        limit: Optional limit

    Returns:
        SPARQL query string
    """
    cohort_clause = f"\n    {cohort_filter}" if cohort_filter else ""
    limit_clause = f"\nLIMIT {limit}" if limit else ""

    if outcome_type == "hospitalization":
        outcome_pattern = f"""
    # Look for inpatient encounter after index date
    OPTIONAL {{
        ?patient ns28:patientEncounter ?outcomeEncounter .
        ?outcomeEncounter ns28:encounterclass "inpatient" ;
                          ns28:startDateTime ?outcomeDate .
        FILTER(?outcomeDate > ?indexDate)
        FILTER(?outcomeDate <= ?indexDate + "P{prediction_horizon_days}D"^^xsd:duration)
    }}
    BIND("Hospital Admission" AS ?outcomeDescription)
"""
    elif outcome_type == "mortality":
        outcome_pattern = f"""
    # Look for death after index date
    OPTIONAL {{
        ?patient ns28:deathdate ?outcomeDate .
        FILTER(?outcomeDate > ?indexDate)
        FILTER(?outcomeDate <= ?indexDate + "P{prediction_horizon_days}D"^^xsd:duration)
    }}
    BIND("Mortality" AS ?outcomeDescription)
"""
    else:  # Generic complication
        outcome_pattern = f"""
    # Look for outcome condition after index date
    OPTIONAL {{
        ?patient ns28:patientCondition ?outcomeCondition .
        ?outcomeCondition ns28:code ?outcomeCodeUri ;
                          ns28:startDateTime ?outcomeDate .
        ?outcomeCodeUri skos:prefLabel ?outcomeDescription .
        FILTER(?outcomeDate > ?indexDate)
        FILTER(?outcomeDate <= ?indexDate + "P{prediction_horizon_days}D"^^xsd:duration)
    }}
"""

    return f"""{PREFIXES}
SELECT DISTINCT
    ?patientId
    ?indexDate
    (IF(BOUND(?outcomeDate), 1, 0) AS ?outcomeOccurred)
    ?outcomeDate
    ?outcomeDescription
    ({prediction_horizon_days} AS ?predictionHorizonDays)
WHERE {{
    ?patient a ns28:Patient ;
             rdfs:label ?patientId .
    {cohort_clause}

    # Index date: First encounter
    {{
        SELECT ?patient (MIN(?encDate) AS ?indexDate) WHERE {{
            ?patient ns28:patientEncounter ?enc .
            ?enc ns28:startDateTime ?encDate .
        }}
        GROUP BY ?patient
    }}

    {outcome_pattern}
}}
ORDER BY ?patientId
{limit_clause}
"""


def encounter_sequence_query(
    patient_ids: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> str:
    """
    Query for encounter-level sequences with encounter class.

    Useful for predicting encounter-level outcomes.

    Args:
        patient_ids: Optional list of patient IDs
        limit: Optional limit

    Returns:
        SPARQL query string
    """
    if patient_ids:
        patient_values = " ".join([f'"{pid}"' for pid in patient_ids])
        patient_filter = f"VALUES ?patientId {{ {patient_values} }}"
    else:
        patient_filter = ""

    limit_clause = f"\nLIMIT {limit}" if limit else ""

    return f"""{PREFIXES}
SELECT DISTINCT
    ?patientId
    ?encounterDateTime
    ?encounterClass
    ?encounterDescription
    (COUNT(DISTINCT ?condition) AS ?conditionCount)
    (COUNT(DISTINCT ?medication) AS ?medicationCount)
    (COUNT(DISTINCT ?observation) AS ?observationCount)
    (COUNT(DISTINCT ?procedure) AS ?procedureCount)
WHERE {{
    ?patient a ns28:Patient ;
             rdfs:label ?patientId .
    {patient_filter}

    ?patient ns28:patientEncounter ?encounter .
    ?encounter ns28:startDateTime ?encounterDateTime .

    OPTIONAL {{ ?encounter ns28:encounterclass ?encounterClass }}
    OPTIONAL {{ ?encounter ns28:description ?encounterDescription }}

    # Count events in this encounter
    OPTIONAL {{ ?encounter ns28:encounterCondition ?condition }}
    OPTIONAL {{ ?encounter ns28:encounterMedication ?medication }}
    OPTIONAL {{ ?encounter ns28:encounterObservation ?observation }}
    OPTIONAL {{ ?encounter ns28:encounterProcedure ?procedure }}
}}
GROUP BY ?patientId ?encounter ?encounterDateTime ?encounterClass ?encounterDescription
ORDER BY ?patientId ?encounterDateTime
{limit_clause}
"""


# Cohort filter examples
COHORT_FILTERS = {
    "diabetes_type2": """
        ?patient ns28:patientCondition ?diabetesCondition .
        ?diabetesCondition ns28:code ?diabetesCode .
        ?diabetesCode skos:notation ?snomedCode .
        FILTER(?snomedCode IN ("44054006", "127013003"))
    """,

    "heart_failure": """
        ?patient ns28:patientCondition ?hfCondition .
        ?hfCondition ns28:code ?hfCode .
        ?hfCode skos:prefLabel ?hfLabel .
        FILTER(CONTAINS(LCASE(?hfLabel), "heart failure"))
    """,

    "copd": """
        ?patient ns28:patientCondition ?copdCondition .
        ?copdCondition ns28:code ?copdCode .
        ?copdCode skos:prefLabel ?copdLabel .
        FILTER(CONTAINS(LCASE(?copdLabel), "chronic obstructive"))
    """,

    "has_inpatient_stay": """
        ?patient ns28:patientEncounter ?inpatientEnc .
        ?inpatientEnc ns28:encounterclass "inpatient" .
    """
}
