from pathlib import Path
import pandas as pd

def test_curated_sales_weather_exists_and_valid():
    path = Path("data/processed/curated_sales_weather.parquet")
    assert path.exists(), "curated_sales_weather.parquet missing"
    df = pd.read_parquet(path)
    assert df.shape[0] == 3000888
    assert df["family"].nunique() == 33
    assert df["city"].nunique() == 22
    assert df["store_nbr"].nunique() == 54
    required_cols = {"date", "store_nbr", "family", "sales", "city", "precipitation_sum", "temperature_2m_max"}
    assert required_cols.issubset(df.columns)
