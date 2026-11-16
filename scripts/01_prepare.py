"""Prepare the public flight-operations sample.

Input : data/raw/kaggle_delay/flights.csv
Output: data/processed/flights.parquet
        outputs/tables/data_profile.csv

The source is the public Kaggle dataset
"Airline Flight Delay Dataset (100K Records)" (kesinisoujanya),
a 2018 BTS-derived flight-level sample.  The downloaded file contains
1,048,575 rows covering 2018-01-01 to 2018-02-26.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "kaggle_delay" / "flights.csv"
OUT = ROOT / "data" / "processed" / "flights.parquet"
TABLES = ROOT / "outputs" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

USECOLS = [
    "FL_DATE",
    "OP_CARRIER",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DELAY",
    "ARR_DELAY",
    "CANCELLED",
    "DIVERTED",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "CARRIER_DELAY",
    "WEATHER_DELAY",
    "NAS_DELAY",
    "SECURITY_DELAY",
    "LATE_AIRCRAFT_DELAY",
]


def hhmm_to_minutes(value: float) -> float:
    """Convert HHMM (e.g. 1517) to minutes after midnight."""
    if pd.isna(value):
        return float("nan")
    value = int(value)
    return (value // 100) * 60 + (value % 100)


def main() -> None:
    df = pd.read_csv(RAW, usecols=USECOLS, low_memory=False)
    df = df.rename(
        columns={
            "FL_DATE": "flight_date",
            "OP_CARRIER": "carrier",
            "OP_CARRIER_FL_NUM": "flight_num",
            "ORIGIN": "origin",
            "DEST": "dest",
            "CRS_DEP_TIME": "crs_dep_time",
            "DEP_DELAY": "dep_delay",
            "ARR_DELAY": "arr_delay",
            "CANCELLED": "cancelled",
            "DIVERTED": "diverted",
            "CRS_ELAPSED_TIME": "crs_elapsed_time",
            "DISTANCE": "distance",
            "CARRIER_DELAY": "carrier_delay",
            "WEATHER_DELAY": "weather_delay",
            "NAS_DELAY": "nas_delay",
            "SECURITY_DELAY": "security_delay",
            "LATE_AIRCRAFT_DELAY": "late_aircraft_delay",
        }
    )

    # The file uses DD-MM-YYYY consistently (e.g. 13-01-2018).
    df["flight_date"] = pd.to_datetime(df["flight_date"], dayfirst=True, errors="raise")
    df["month"] = df["flight_date"].dt.month.astype("int8")
    df["day"] = df["flight_date"].dt.day.astype("int8")
    df["day_of_week"] = df["flight_date"].dt.day_of_week.astype("int8")  # Monday=0
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    df["dep_hour"] = (df["crs_dep_time"] // 100).clip(0, 23).astype("int8")
    df["dep_minute"] = (df["crs_dep_time"] % 100).clip(0, 59).astype("int8")
    df["sched_dep_min"] = df["crs_dep_time"].map(hhmm_to_minutes).astype("float32")
    df["route"] = df["origin"] + "-" + df["dest"]

    # Target: DOT-style "arrival delayed 15+ minutes", only for completed flights.
    df["completed"] = ((df["cancelled"] == 0) & (df["diverted"] == 0)).astype("int8")
    df["is_delayed"] = pd.NA
    completed_mask = df["completed"].eq(1) & df["arr_delay"].notna()
    df.loc[completed_mask, "is_delayed"] = (df.loc[completed_mask, "arr_delay"] > 15).astype("int8")

    # Keep the five DOT delay-cause columns; they are populated only for delayed flights.
    cause_cols = [
        "carrier_delay",
        "weather_delay",
        "nas_delay",
        "security_delay",
        "late_aircraft_delay",
    ]
    df["has_cause"] = df[cause_cols].notna().any(axis=1).astype("int8")

    df["origin"] = df["origin"].astype("category")
    df["dest"] = df["dest"].astype("category")
    df["carrier"] = df["carrier"].astype("category")
    df["route"] = df["route"].astype("category")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    profile = pd.DataFrame(
        {
            "metric": [
                "rows",
                "flights_completed",
                "flights_cancelled",
                "flights_diverted",
                "date_min",
                "date_max",
                "carriers",
                "airports",
                "arr_delay_15_rate_completed",
                "mean_arr_delay_completed_min",
                "median_arr_delay_completed_min",
                "cause_rows",
            ],
            "value": [
                len(df),
                int(df["completed"].sum()),
                int(df["cancelled"].sum()),
                int(df["diverted"].sum()),
                str(df["flight_date"].min().date()),
                str(df["flight_date"].max().date()),
                int(df["carrier"].nunique()),
                int(df["origin"].nunique()),
                round(float(df.loc[completed_mask, "is_delayed"].mean()), 4),
                round(float(df.loc[completed_mask, "arr_delay"].mean()), 2),
                round(float(df.loc[completed_mask, "arr_delay"].median()), 2),
                int(df["has_cause"].sum()),
            ],
        }
    )
    profile.to_csv(TABLES / "data_profile.csv", index=False)
    print(profile.to_string(index=False))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
