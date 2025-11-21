"""
Utility functions for persisting the consolidated fx_features dataset.
"""

from typing import Iterable

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import project_paths  # noqa: F401
from config import PG_CONFIG, SQLALCHEMY_ECHO


def _build_connection_string() -> str:
    return (
        f"postgresql://{PG_CONFIG['user']}:{PG_CONFIG['password']}"
        f"@{PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}"
    )


ENGINE: Engine = create_engine(_build_connection_string(), echo=SQLALCHEMY_ECHO, future=True)

CREATE_FX_FEATURES_SQL = """
CREATE TABLE IF NOT EXISTS fx_features (
    date DATE PRIMARY KEY,
    usd_thb DOUBLE PRECISION,
    dxy DOUBLE PRECISION,
    gold DOUBLE PRECISION,
    oil DOUBLE PRECISION,
    vix DOUBLE PRECISION,
    sp500 DOUBLE PRECISION,
    set_index DOUBLE PRECISION,
    th_10y DOUBLE PRECISION,
    th_cpi DOUBLE PRECISION,
    th_policy_rate DOUBLE PRECISION,
    us_10y DOUBLE PRECISION,
    us_cpi DOUBLE PRECISION,
    us_fed_rate DOUBLE PRECISION,
    news_sentiment DOUBLE PRECISION
);
"""


def _build_upsert_sql(table_name: str, columns: list) -> str:
    """Build UPSERT SQL that updates existing values with new non-null values."""
    insert_cols = ", ".join(columns)
    insert_vals = ", ".join([f":{col}" for col in columns])
    
    # Update columns with new values (if provided), otherwise keep existing

    set_clauses = []
    for col in columns[1:]:  # Skip 'date' (primary key)
        set_clauses.append(f"{col} = COALESCE(:{col}, {table_name}.{col})")
    
    return f"""
        INSERT INTO {table_name} ({insert_cols})
        VALUES ({insert_vals})
        ON CONFLICT (date) DO UPDATE SET
        {', '.join(set_clauses)}
    """


def _ensure_table_exists():
    """Create fx_features table if it doesn't exist."""
    with ENGINE.begin() as conn:
        conn.execute(text(CREATE_FX_FEATURES_SQL))


def save_fx_features(df: pd.DataFrame, table_name: str = "fx_features") -> None:
    if df is None or df.empty:
        print("⚠️  No FX features to save.")
        return

    # Only ensure production table exists for production saves
    if table_name == "fx_features":
        _ensure_table_exists()

    data = df.copy()
    data.index = pd.to_datetime(data.index)
    data.index.name = "date"
    data = data.reset_index()  # Convert index to column
    
    # Use actual columns from DataFrame instead of hardcoded schema
    columns = ["date"] + [col for col in data.columns if col != "date"]
    
    # Convert date to string format
    data["date"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    
    # Build UPSERT SQL
    upsert_sql = _build_upsert_sql(table_name, columns)
    
    # Execute UPSERT row by row (PostgreSQL ON CONFLICT requires this approach)
    with ENGINE.begin() as conn:
        upsert_stmt = text(upsert_sql)
        
        for _, row in data.iterrows():
            # Convert row to dict and replace NaN with None
            record = {}
            for col in columns:
                value = row[col]
                record[col] = None if pd.isna(value) else value
            
            # Execute upsert for this row
            conn.execute(upsert_stmt, record)

    print(f"✅ Upserted {len(data)} fx feature rows (overwrote existing values).")


def get_engine() -> Engine:
    return ENGINE


def create_test_table(table_name: str, columns: list[str]) -> None:
    """
    Create a test table with specified columns.
    
    Args:
        table_name: Name of the test table (e.g., "test_gold")
        columns: List of column names (date will be added automatically)
    """
    # Build column definitions (all DOUBLE PRECISION except date)
    col_defs = ["date DATE PRIMARY KEY"]
    for col in columns:
        if col != "date":
            col_defs.append(f"{col} DOUBLE PRECISION")
    
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        {', '.join(col_defs)}
    );
    """
    
    with ENGINE.begin() as conn:
        conn.execute(text(create_sql))
    
    print(f"✅ Created table '{table_name}' with columns: {columns}")


def drop_table(table_name: str, confirm: bool = True) -> None:
    """
    Drop a table from the database.
    
    Args:
        table_name: Name of the table to drop
        confirm: If True, prompt for confirmation before dropping
    """
    if confirm:
        response = input(f"⚠️  Are you sure you want to drop table '{table_name}'? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("❌ Cancelled.")
            return
    
    drop_sql = f"DROP TABLE IF EXISTS {table_name};"
    
    with ENGINE.begin() as conn:
        conn.execute(text(drop_sql))
    
    print(f"✅ Dropped table '{table_name}'")


def list_test_tables() -> list[str]:
    """
    List all tables starting with 'test_'.
    
    Returns:
        List of test table names
    """
    query = """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    AND table_name LIKE 'test_%';
    """
    
    with ENGINE.begin() as conn:
        result = conn.execute(text(query))
        tables = [row[0] for row in result]
    
    return tables


__all__ = ["save_fx_features", "get_engine", "ENGINE", "create_test_table", "drop_table", "list_test_tables"]
