import pandas_datareader.data as web
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

start = "2000-01-01"
end = "2024-12-31"

print("Downloading Treasury Data")

dgs10 = web.DataReader("DGS10", "fred", start, end)
dgs2 = web.DataReader("DGS2", "fred", start, end)

spread = dgs10["DGS10"] - dgs2["DGS2"]

df = pd.DataFrame({
    "US10Y": dgs10["DGS10"],
    "US2Y": dgs2["DGS2"],
    "yield_spread": spread
})

df.to_parquet(DATA_DIR / "treasury_spread.parquet")

print("Saved treasury spread data")