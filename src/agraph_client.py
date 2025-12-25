"""
AllegroGraph client wrapper for PatientGraph.

Provides a simplified interface for connecting to AllegroGraph and
executing SPARQL queries with pandas DataFrame output.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from contextlib import contextmanager

from franz.openrdf.connect import ag_connect
from franz.openrdf.query.query import QueryLanguage
from franz.openrdf.repository.repositoryconnection import RepositoryConnection

from .config import Config, get_config


class PatientGraphClient:
    """
    Client for connecting to and querying PatientGraph.

    Usage:
        client = PatientGraphClient()
        client.connect()

        df = client.query("SELECT * WHERE { ?s ?p ?o } LIMIT 10")
        print(df)

        client.close()

    Or using context manager:
        with PatientGraphClient() as client:
            df = client.query("SELECT ...")
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the client.

        Args:
            config: Configuration object. If None, loads from default config.yaml
        """
        self.config = config or get_config()
        self.conn: Optional[RepositoryConnection] = None
        self._namespaces_set = False

    def connect(self) -> 'PatientGraphClient':
        """
        Connect to AllegroGraph.

        Returns:
            self for method chaining
        """
        ag = self.config.agraph

        self.conn = ag_connect(
            ag.repository,
            host=ag.host,
            port=ag.port,
            user=ag.user,
            password=ag.password,
            catalog=ag.catalog if ag.catalog else None
        )

        # Set up namespaces
        self._setup_namespaces()

        return self

    def _setup_namespaces(self):
        """Register namespaces from config for cleaner queries."""
        if self._namespaces_set:
            return

        for prefix, uri in self.config.namespaces.items():
            self.conn.setNamespace(prefix, uri)

        self._namespaces_set = True

    def close(self):
        """Close the connection."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            self._namespaces_set = False

    def __enter__(self) -> 'PatientGraphClient':
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection and return basic stats.

        Returns:
            Dictionary with connection info and triple count
        """
        if self.conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        triple_count = self.conn.size()

        return {
            'connected': True,
            'repository': self.config.agraph.repository,
            'host': self.config.agraph.host,
            'port': self.config.agraph.port,
            'triple_count': triple_count
        }

    def query(self, sparql: str, include_inferred: bool = False) -> pd.DataFrame:
        """
        Execute a SPARQL SELECT query and return results as DataFrame.

        Args:
            sparql: SPARQL query string
            include_inferred: Whether to include inferred triples

        Returns:
            pandas DataFrame with query results
        """
        if self.conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        if include_inferred:
            q = self.conn.prepareTupleQuery(QueryLanguage.SPARQL, sparql)
            q.setIncludeInferred(True)
            result = q.evaluate()
        else:
            result = self.conn.executeTupleQuery(sparql)

        return result.toPandas()

    def query_with_prefixes(self, sparql: str, include_inferred: bool = False) -> pd.DataFrame:
        """
        Execute query with standard prefixes prepended.

        Args:
            sparql: SPARQL query (without PREFIX declarations)
            include_inferred: Whether to include inferred triples

        Returns:
            pandas DataFrame with query results
        """
        prefixes = self._get_prefix_declarations()
        full_query = prefixes + "\n" + sparql
        return self.query(full_query, include_inferred)

    def _get_prefix_declarations(self) -> str:
        """Generate PREFIX declarations from config namespaces."""
        lines = []
        for prefix, uri in self.config.namespaces.items():
            lines.append(f"PREFIX {prefix}: <{uri}>")
        return "\n".join(lines)

    def get_namespaces(self) -> Dict[str, str]:
        """
        Get all registered namespaces.

        Returns:
            Dictionary mapping prefix to URI
        """
        if self.conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        return dict(self.conn.getNamespaces())

    def shorten_uri(self, uri: str) -> str:
        """
        Shorten a full URI to prefixed form using registered namespaces.

        Args:
            uri: Full URI string (with or without angle brackets)

        Returns:
            Shortened prefixed form (e.g., "ns28:Patient")
        """
        # Remove angle brackets if present
        if uri.startswith('<') and uri.endswith('>'):
            uri = uri[1:-1]

        namespaces = self.config.namespaces
        for prefix, ns in namespaces.items():
            if uri.startswith(ns):
                return f"{prefix}:{uri[len(ns):]}"

        return uri

    def shorten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply URI shortening to all columns in a DataFrame.

        Args:
            df: DataFrame with URI values

        Returns:
            DataFrame with shortened URIs
        """
        result = df.copy()
        for col in result.columns:
            result[col] = result[col].apply(
                lambda val: self.shorten_uri(str(val)) if val is not None else val
            )
        return result

    def count(self, sparql: str) -> int:
        """
        Execute a query that returns a count.

        Args:
            sparql: SPARQL query with COUNT in SELECT

        Returns:
            Integer count
        """
        df = self.query(sparql)
        if len(df) > 0 and len(df.columns) > 0:
            return int(df.iloc[0, 0])
        return 0

    def execute_update(self, sparql: str) -> bool:
        """
        Execute a SPARQL UPDATE query.

        Args:
            sparql: SPARQL UPDATE query

        Returns:
            True if the store was modified
        """
        if self.conn is None:
            raise RuntimeError("Not connected. Call connect() first.")

        return self.conn.executeUpdate(sparql)


# Convenience function for quick queries
@contextmanager
def patient_graph_connection(config: Optional[Config] = None):
    """
    Context manager for PatientGraph connections.

    Usage:
        with patient_graph_connection() as client:
            df = client.query("SELECT ...")
    """
    client = PatientGraphClient(config)
    try:
        client.connect()
        yield client
    finally:
        client.close()
