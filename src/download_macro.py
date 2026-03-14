from fredapi import Fred
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# ---------------------------------------
# LOAD ENV VARIABLES
# ---------------------------------------

load_dotenv()

# ---------------------------------------
# CONFIG
# ---------------------------------------

START_DATE = "2000-01-01"

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

print("Downloading macroeconomic indicators from FRED...")

# ---------------------------------------
# DOWNLOAD DATA
# ---------------------------------------

us10y = fred.get_series("DGS10", observation_start=START_DATE)
us2y = fred.get_series("DGS2", observation_start=START_DATE)

fedfunds = fred.get_series("FEDFUNDS", observation_start=START_DATE)
cpi = fred.get_series("CPIAUCSL", observation_start=START_DATE)
unemployment = fred.get_series("UNRATE", observation_start=START_DATE)
industrial_prod = fred.get_series("INDPRO", observation_start=START_DATE)

# ---------------------------------------
# CREATE DATAFRAME
# ---------------------------------------

df = pd.DataFrame({
    "US10Y": us10y,
    "US2Y": us2y,
    "FEDFUNDS": fedfunds,
    "CPI": cpi,
    "UNEMPLOYMENT": unemployment,
    "INDPRO": industrial_prod
})

df["YIELD_SPREAD"] = df["US10Y"] - df["US2Y"]

df = df.sort_index().ffill()

# ---------------------------------------
# SAVE DATA
# ---------------------------------------

df.to_parquet(DATA_DIR / "macro_data.parquet")

print("Macro dataset saved.")
print(df.tail())