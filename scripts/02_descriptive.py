"""Descriptive analytics for the flight-operations sample.

Run after 01_prepare.py:
    python scripts/02_descriptive.py

Outputs
-------
outputs/tables/kpi_overall.csv
outputs/tables/delay_by_carrier.csv
outputs/tables/top_origin_airports.csv
outputs/tables/top_routes.csv
outputs/tables/delay_causes.csv
outputs/figures/delay_rate_by_carrier.png
outputs/figures/top_origin_airports.png
outputs/figures/delay_heatmap_dow_hour.png
outputs/figures/delay_causes.png
outputs/figures/airport_delay_map.png
outputs/metrics_descriptive.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "flights.parquet"
AIRPORTS = ROOT / "data" / "raw" / "airports.csv"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 180,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 9,
    }
)

CAUSE_COLS = [
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]


def load() -> pd.DataFrame:
    return pd.read_parquet(DATA)


def overall_kpis(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["completed"].eq(1) & df["arr_delay"].notna()]
    kpis = {
        "flights_total": int(len(df)),
        "flights_completed": int(len(completed)),
        "flights_cancelled": int(df["cancelled"].sum()),
        "flights_diverted": int(df["diverted"].sum()),
        "cancellation_rate": round(float(df["cancelled"].mean()), 4),
        "delay_15_rate": round(float((completed["arr_delay"] > 15).mean()), 4),
        "mean_arr_delay_min": round(float(completed["arr_delay"].mean()), 2),
        "median_arr_delay_min": round(float(completed["arr_delay"].median()), 2),
        "p90_arr_delay_min": round(float(completed["arr_delay"].quantile(0.90)), 2),
        "date_min": str(df["flight_date"].min().date()),
        "date_max": str(df["flight_date"].max().date()),
    }
    out = pd.DataFrame({"metric": list(kpis), "value": list(kpis.values())})
    out.to_csv(TABLES / "kpi_overall.csv", index=False)
    return out


def delay_by_carrier(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["completed"].eq(1) & df["arr_delay"].notna()]
    by_carrier = (
        completed.groupby("carrier", observed=True)
        .agg(
            flights=("arr_delay", "size"),
            delay_15_rate=("arr_delay", lambda s: (s > 15).mean()),
            mean_arr_delay_min=("arr_delay", "mean"),
            median_arr_delay_min=("arr_delay", "median"),
        )
        .reset_index()
    )
    canc = df.groupby("carrier", observed=True)["cancelled"].mean().rename("cancellation_rate")
    by_carrier = by_carrier.merge(canc, on="carrier", how="left")
    by_carrier = by_carrier.sort_values("delay_15_rate", ascending=False).round(4)
    by_carrier.to_csv(TABLES / "delay_by_carrier.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    order = by_carrier.sort_values("delay_15_rate")
    ax.barh(order["carrier"], order["delay_15_rate"] * 100, color="#3b6ea5")
    ax.set_xlabel("Flights arriving 15+ min late (%)")
    ax.set_title("Delay rate by carrier (completed flights)")
    for i, v in enumerate(order["delay_15_rate"] * 100):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "delay_rate_by_carrier.png")
    plt.close(fig)
    return by_carrier


def top_origin_airports(df: pd.DataFrame, min_flights: int = 2000) -> pd.DataFrame:
    completed = df[df["completed"].eq(1) & df["arr_delay"].notna()]
    by_origin = (
        completed.groupby("origin", observed=True)
        .agg(
            flights=("arr_delay", "size"),
            delay_15_rate=("arr_delay", lambda s: (s > 15).mean()),
            mean_arr_delay_min=("arr_delay", "mean"),
        )
        .reset_index()
    )
    by_origin = by_origin[by_origin["flights"] >= min_flights]
    by_origin = by_origin.sort_values("delay_15_rate", ascending=False).round(4)
    by_origin.to_csv(TABLES / "top_origin_airports.csv", index=False)

    top = by_origin.head(15).sort_values("delay_15_rate")
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh(top["origin"], top["delay_15_rate"] * 100, color="#b5653a")
    ax.set_xlabel("Flights arriving 15+ min late (%)")
    ax.set_title(f"Highest-delay origin airports (>= {min_flights:,} flights)")
    for i, v in enumerate(top["delay_15_rate"] * 100):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "top_origin_airports.png")
    plt.close(fig)
    return by_origin


def delay_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    completed = df[df["completed"].eq(1) & df["arr_delay"].notna()].copy()
    completed["late"] = (completed["arr_delay"] > 15).astype(int)
    pivot = (
        completed.pivot_table(
            index="day_of_week", columns="dep_hour", values="late", aggfunc="mean"
        )
        .reindex(index=range(7), columns=range(24))
    )
    counts = (
        completed.pivot_table(
            index="day_of_week", columns="dep_hour", values="late", aggfunc="size"
        )
        .reindex(index=range(7), columns=range(24))
    )
    pivot.to_csv(TABLES / "delay_heatmap_dow_hour.csv")
    counts.to_csv(TABLES / "delay_heatmap_counts.csv")

    # Mask low-traffic cells (e.g. 03:00) so they do not dominate the colour scale.
    plot_values = pivot.where(counts >= 200)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    im = ax.imshow(plot_values.values * 100, aspect="auto", cmap=cmap, vmin=0)
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)], fontsize=7)
    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], fontsize=8)
    ax.set_xlabel("Scheduled departure hour")
    ax.set_title("Delay rate (%) by day of week and scheduled departure hour")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Delay rate (%)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "delay_heatmap_dow_hour.png")
    plt.close(fig)
    return plot_values


def delay_causes(df: pd.DataFrame) -> pd.DataFrame:
    cause = df[df["has_cause"].eq(1)].copy()
    totals = cause[CAUSE_COLS].sum()
    share = (totals / totals.sum()).rename("share_of_delay_minutes")
    flights_with = (cause[CAUSE_COLS] > 0).sum().rename("flights_with_cause")
    out = pd.concat([totals.rename("delay_minutes"), share, flights_with], axis=1).reset_index()
    out = out.rename(columns={"index": "cause"}).sort_values(
        "delay_minutes", ascending=False
    )
    out.round(4).to_csv(TABLES / "delay_causes.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    order = out.sort_values("delay_minutes")
    labels = {
        "carrier_delay": "Carrier",
        "weather_delay": "Weather",
        "nas_delay": "NAS",
        "security_delay": "Security",
        "late_aircraft_delay": "Late aircraft",
    }
    ax.barh(
        [labels.get(x, x) for x in order["cause"]],
        order["share_of_delay_minutes"] * 100,
        color="#5c7f67",
    )
    ax.set_xlabel("Share of attributed delay minutes (%)")
    ax.set_title("Attributed delay minutes by cause")
    for i, v in enumerate(order["share_of_delay_minutes"] * 100):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "delay_causes.png")
    plt.close(fig)
    return out


def airports_with_coords(df: pd.DataFrame, min_flights: int = 1000) -> pd.DataFrame:
    completed = df[df["completed"].eq(1) & df["arr_delay"].notna()]
    by_origin = (
        completed.groupby("origin", observed=True)
        .agg(
            flights=("arr_delay", "size"),
            delay_15_rate=("arr_delay", lambda s: (s > 15).mean()),
        )
        .reset_index()
    )
    by_origin = by_origin[by_origin["flights"] >= min_flights]

    ap = pd.read_csv(AIRPORTS, usecols=["iata_code", "name", "iso_country", "latitude_deg", "longitude_deg"])
    ap = ap[ap["iso_country"].eq("US") & ap["iata_code"].notna()]
    ap = ap.drop_duplicates("iata_code")
    merged = by_origin.merge(ap, left_on="origin", right_on="iata_code", how="inner")
    return merged


def airport_map(df: pd.DataFrame) -> pd.DataFrame:
    merged = airports_with_coords(df)
    merged.to_csv(TABLES / "airport_delay_map_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    sc = ax.scatter(
        merged["longitude_deg"],
        merged["latitude_deg"],
        c=merged["delay_15_rate"] * 100,
        s=np.clip(merged["flights"] / 400, 8, 120),
        cmap="YlOrRd",
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Origin-airport delay rate (bubble size = flights)")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Delay rate (%)")
    for _, row in merged.sort_values("delay_15_rate", ascending=False).head(10).iterrows():
        ax.annotate(
            row["origin"],
            (row["longitude_deg"], row["latitude_deg"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    fig.tight_layout()
    fig.savefig(FIGURES / "airport_delay_map.png")
    plt.close(fig)
    return merged


def top_routes(df: pd.DataFrame, min_flights: int = 500) -> pd.DataFrame:
    completed = df[df["completed"].eq(1) & df["arr_delay"].notna()]
    by_route = (
        completed.groupby("route", observed=True)
        .agg(
            flights=("arr_delay", "size"),
            delay_15_rate=("arr_delay", lambda s: (s > 15).mean()),
            mean_arr_delay_min=("arr_delay", "mean"),
        )
        .reset_index()
    )
    by_route = by_route[by_route["flights"] >= min_flights]
    by_route = by_route.sort_values("delay_15_rate", ascending=False).round(4)
    by_route.to_csv(TABLES / "top_routes.csv", index=False)
    return by_route


def main() -> None:
    df = load()
    kpis = overall_kpis(df)
    by_carrier = delay_by_carrier(df)
    by_origin = top_origin_airports(df)
    heat = delay_heatmap(df)
    causes = delay_causes(df)
    mapped = airport_map(df)
    routes = top_routes(df)

    metrics = {
        "kpis": dict(zip(kpis["metric"], kpis["value"])),
        "carrier_highest_delay_rate": by_carrier.head(1).to_dict(orient="records"),
        "origin_highest_delay_rate": by_origin.head(1).to_dict(orient="records"),
        "route_highest_delay_rate": routes.head(1).to_dict(orient="records"),
        "heatmap_peak": {
            "day_of_week": int(np.nanargmax(heat.values) // heat.shape[1]),
            "dep_hour": int(np.nanargmax(heat.values) % heat.shape[1]),
            "delay_rate": round(float(np.nanmax(heat.values)), 4),
        },
        "delay_cause_share": dict(
            zip(causes["cause"], causes["share_of_delay_minutes"].round(4))
        ),
        "airports_mapped": int(len(mapped)),
    }
    with open(ROOT / "outputs" / "metrics_descriptive.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
