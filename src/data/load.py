import os

import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """Charge les données depuis un chemin local, une URL publique ou S3 (s3://bucket/key)."""
    if filepath.startswith("s3://"):
        storage_options = {
            "endpoint_url": os.getenv("AWS_S3_ENDPOINT_URL", "https://minio.lab.sspcloud.fr"),
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "token": os.getenv("AWS_SESSION_TOKEN"),
            "anon": not os.getenv("AWS_ACCESS_KEY_ID"),  # lecture anonyme si pas de credentials
        }
        return pd.read_csv(filepath, storage_options=storage_options)
    return pd.read_csv(filepath)
