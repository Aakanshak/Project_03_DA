"""Clean Rossmann CSV files and load them into normalized PostgreSQL tables."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
STORE_PATH = PROJECT_ROOT / "data" / "raw" / "store.csv"
SCHEMA_PATH = PROJECT_ROOT / "sql" / "schema.sql"


def require_columns(
    dataframe: pd.DataFrame, required_columns: set[str], source_name: str
) -> None:
    """Raise a useful error when a source file does not match the expected schema."""
    missing = required_columns.difference(dataframe.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"{source_name} is missing required columns: {missing_list}")


def read_source_data(
    train_path: Path = TRAIN_PATH, store_path: Path = STORE_PATH
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the Rossmann sales and store metadata CSV files."""
    missing_files = [path for path in (train_path, store_path) if not path.exists()]
    if missing_files:
        paths = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing Rossmann source file(s): {paths}")

    sales = pd.read_csv(train_path, low_memory=False)
    stores = pd.read_csv(store_path, low_memory=False)
    return sales, stores


def clean_sales_data(sales: pd.DataFrame) -> pd.DataFrame:
    """Clean daily sales and remove explicitly closed zero-sales observations."""
    required = {
        "Store",
        "Date",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
    }
    require_columns(sales, required, "train.csv")

    cleaned = sales.copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="raise")

    numeric_columns = ["Store", "Sales", "Customers", "Open", "Promo", "SchoolHoliday"]
    cleaned[numeric_columns] = cleaned[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )

    closed_zero_sales = cleaned["Open"].eq(0) & cleaned["Sales"].eq(0)
    cleaned = cleaned.loc[~closed_zero_sales].copy()

    cleaned["StateHoliday"] = (
        cleaned["StateHoliday"].fillna("0").astype(str).str.strip().str.lower()
    )
    cleaned["Open"] = cleaned["Open"].astype(bool)
    cleaned["Promo"] = cleaned["Promo"].astype(bool)
    cleaned["SchoolHoliday"] = cleaned["SchoolHoliday"].astype(bool)

    cleaned = cleaned.rename(
        columns={
            "Store": "store_id",
            "Date": "sales_date",
            "Sales": "sales",
            "Customers": "customers",
            "Open": "open",
            "Promo": "promo",
            "StateHoliday": "state_holiday",
            "SchoolHoliday": "school_holiday",
        }
    )

    output_columns = [
        "store_id",
        "sales_date",
        "sales",
        "customers",
        "open",
        "promo",
        "state_holiday",
        "school_holiday",
    ]
    return cleaned[output_columns].sort_values(["store_id", "sales_date"])


def clean_store_data(stores: pd.DataFrame) -> pd.DataFrame:
    """Clean store attributes and impute missing competition distance with the median."""
    required = {
        "Store",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "PromoInterval",
    }
    require_columns(stores, required, "store.csv")

    cleaned = stores.copy()
    cleaned["CompetitionDistance"] = pd.to_numeric(
        cleaned["CompetitionDistance"], errors="coerce"
    )
    competition_median = cleaned["CompetitionDistance"].median()
    if pd.isna(competition_median):
        raise ValueError("CompetitionDistance contains no usable numeric values.")
    cleaned["CompetitionDistance"] = cleaned["CompetitionDistance"].fillna(
        competition_median
    )

    nullable_integer_columns = [
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]
    for column in nullable_integer_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype(
            "Int64"
        )

    cleaned["Promo2"] = pd.to_numeric(cleaned["Promo2"], errors="raise").astype(bool)
    no_promo2 = ~cleaned["Promo2"]
    cleaned.loc[
        no_promo2, ["Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"]
    ] = pd.NA

    cleaned["StoreType"] = cleaned["StoreType"].astype(str).str.strip().str.lower()
    cleaned["Assortment"] = cleaned["Assortment"].astype(str).str.strip().str.lower()
    cleaned["PromoInterval"] = cleaned["PromoInterval"].replace(
        {"": pd.NA, "nan": pd.NA}
    )

    return cleaned.rename(
        columns={
            "Store": "store_id",
            "StoreType": "store_type",
            "Assortment": "assortment",
            "CompetitionDistance": "competition_distance",
            "CompetitionOpenSinceMonth": "competition_open_since_month",
            "CompetitionOpenSinceYear": "competition_open_since_year",
            "Promo2": "promo2",
            "Promo2SinceWeek": "promo2_since_week",
            "Promo2SinceYear": "promo2_since_year",
            "PromoInterval": "promo_interval",
        }
    )


def build_engine() -> Engine:
    """Create a SQLAlchemy engine from environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")

    required_variables = [
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    ]
    missing = [name for name in required_variables if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing database environment variables: " + ", ".join(missing)
        )

    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ["POSTGRES_PORT"]),
        database=os.environ["POSTGRES_DB"],
    )
    return create_engine(url, pool_pre_ping=True)


def create_schema(engine: Engine) -> None:
    """Execute the PostgreSQL schema file before loading data."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    raw_connection = engine.raw_connection()
    try:
        with raw_connection.cursor() as cursor:
            cursor.execute(schema_sql)
        raw_connection.commit()
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        raw_connection.close()


def load_to_postgres(
    engine: Engine, sales: pd.DataFrame, store_metadata: pd.DataFrame
) -> None:
    """Replace source-backed table contents in one transaction."""
    store_ids = pd.DataFrame(
        {"store_id": sorted(store_metadata["store_id"].astype(int).unique())}
    )

    with engine.begin() as connection:
        connection.execute(
            text("TRUNCATE TABLE daily_sales, store_metadata, stores CASCADE")
        )
        store_ids.to_sql(
            "stores", connection, if_exists="append", index=False, method="multi"
        )
        store_metadata.to_sql(
            "store_metadata",
            connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=1_000,
        )
        sales.to_sql(
            "daily_sales",
            connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5_000,
        )


def main() -> None:
    """Run the Rossmann CSV-to-PostgreSQL ingestion workflow."""
    raw_sales, raw_stores = read_source_data()
    sales = clean_sales_data(raw_sales)
    store_metadata = clean_store_data(raw_stores)

    engine = build_engine()
    create_schema(engine)
    load_to_postgres(engine, sales, store_metadata)
    print(
        f"Loaded {len(store_metadata):,} stores and {len(sales):,} daily sales rows."
    )


if __name__ == "__main__":
    main()
