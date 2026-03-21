# Download macroeconomic indicators from FRED API and save as a single parquet file

from fredapi import Fred
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file to access the FRED API key securely
load_dotenv()

# Define start date for all series and set up the raw data output directory
START_DATE = "2000-01-01"
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Initialize FRED client using API key from environment
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

print("Downloading macroeconomic indicators from FRED...")

# Download each macroeconomic series from FRED starting from START_DATE
us10y = fred.get_series("DGS10", observation_start=START_DATE)        # 10-Year Treasury Yield
us2y = fred.get_series("DGS2", observation_start=START_DATE)          # 2-Year Treasury Yield
fedfunds = fred.get_series("FEDFUNDS", observation_start=START_DATE)  # Federal Funds Rate
cpi = fred.get_series("CPIAUCSL", observation_start=START_DATE)       # Consumer Price Index
unemployment = fred.get_series("UNRATE", observation_start=START_DATE) # Unemployment Rate
industrial_prod = fred.get_series("INDPRO", observation_start=START_DATE) # Industrial Production Index

# Combine all series into a single DataFrame
df = pd.DataFrame({
    "US10Y": us10y,
    "US2Y": us2y,
    "FEDFUNDS": fedfunds,
    "CPI": cpi,
    "UNEMPLOYMENT": unemployment,
    "INDPRO": industrial_prod
})

# Compute yield spread as the difference between 10Y and 2Y Treasury yields (recession indicator)
df["YIELD_SPREAD"] = df["US10Y"] - df["US2Y"]

# Sort by date and forward-fill missing values to handle weekends and holidays
df = df.sort_index().ffill()

# Save the final macro dataset to parquet
df.to_parquet(DATA_DIR / "macro_data.parquet")

print("Macro dataset saved.")
print(df.tail())