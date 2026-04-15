import os

import duckdb
import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Charge les données depuis un chemin local, une URL publique ou S3.

    Supporte les formats CSV et Parquet. Pour les fichiers Parquet, utilise
    DuckDB pour une lecture efficace. Pour les fichiers S3 privés, les
    credentials sont lus depuis les variables d'environnement AWS_*.
    """
    if filepath.endswith(".parquet"):
        con = duckdb.connect(":memory:")
        return con.sql(f"SELECT * FROM read_parquet('{filepath}')").df()

    if filepath.startswith("s3://"):
        storage_options = {
            "endpoint_url": os.getenv("AWS_S3_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "token": os.getenv("AWS_SESSION_TOKEN"),
            "anon": not os.getenv("AWS_ACCESS_KEY_ID"),
        }
        return pd.read_csv(filepath, storage_options=storage_options)

    return pd.read_csv(filepath)
