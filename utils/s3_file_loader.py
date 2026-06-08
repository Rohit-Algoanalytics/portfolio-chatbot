import pandas as pd
import boto3
import json
import os 
from dotenv import load_dotenv

load_dotenv(override=True)

s3 = boto3.client("s3",aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),)

def _flatten_filter_col(series: pd.Series, prefix: str) -> pd.DataFrame:
    """
    Flatten a filter column where each value is either a dict of filter results
    or False/None (meaning all filters are inactive).
    Collects all possible filter keys from every row so columns are consistent,
    then fills missing / non-dict rows with False.
    """
    all_keys: list[str] = []
    seen: set[str] = set()
    for val in series:
        if isinstance(val, dict):
            for k in val:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

    rows = []
    for val in series:
        if isinstance(val, dict):
            rows.append({k: val.get(k, False) for k in all_keys})
        else:
            rows.append({k: False for k in all_keys})

    expanded = pd.DataFrame(rows, index=series.index)
    expanded.columns = [f"{prefix}_{c}" for c in expanded.columns]
    return expanded


def load_portfolio_data(user_id: str) -> pd.DataFrame:
    bucket_name = "devfabric.algoanalytics.com"
    base_prefix = "Json/multi-user/latest-portfolio-data/multi-user-portfolio-data"

    s3_key = f"{base_prefix}/{user_id}/portfolio_data.json"

    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    raw = json.loads(obj["Body"].read().decode("utf-8"))

    holdings = raw.get("holdings", [])
    if not holdings:
        raise ValueError(f"No holdings data found for user: {user_id}")

    df = pd.DataFrame(holdings)

    # Flatten momentum_filters and value_filters with consistent columns even
    # when a row has False instead of a dict.
    for col in ["momentum_filters", "value_filters"]:
        if col in df.columns:
            expanded = _flatten_filter_col(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), expanded], axis=1)

    # Flatten other nested dict columns (mtf, authorisation) the simple way
    for col in ["mtf", "authorisation"]:
        if col in df.columns:
            expanded = df[col].apply(
                lambda x: x if isinstance(x, dict) else {}
            ).apply(pd.Series)
            if not expanded.empty:
                expanded.columns = [f"{col}_{c}" for c in expanded.columns]
                df = pd.concat([df.drop(columns=[col]), expanded], axis=1)
                df = df.loc[:, ~(df.isna().all())]
            else:
                df = df.drop(columns=[col])

    return df


def get_user_list() -> list[str]:
    """
    Fetch list of users from S3 bucket.
    Returns list of user folder names (user IDs).
    """
    bucket_name = "devfabric.algoanalytics.com"
    base_prefix = "Json/multi-user/latest-portfolio-data/multi-user-portfolio-data/"
    
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=base_prefix,
        Delimiter="/"
    )
    

    print(response)
    users = []
    for prefix in response.get("CommonPrefixes", []):
        # Extract folder name (user ID) from full prefix
        folder = prefix["Prefix"].replace(base_prefix, "").rstrip("/")
        if folder:
            users.append(folder)
    
    return users