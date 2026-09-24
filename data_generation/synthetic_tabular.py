import os
import pandas as pd
import numpy as np
from pathlib import Path

from config import Paths


def _extend_demand_history(seed_df: pd.DataFrame, target_rows: int = 2000) -> pd.DataFrame:
    """Generate extended demand history preserving SKU-level patterns."""
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata

        num_cols = ["units_sold", "units_returned", "net_units", "revenue_inr"]
        seed = seed_df[num_cols].copy()

        meta = SingleTableMetadata()
        meta.detect_from_dataframe(seed)
        syn = GaussianCopulaSynthesizer(meta)
        syn.fit(seed)
        synthetic_nums = syn.sample(num_rows=target_rows)
        synthetic_nums = synthetic_nums.abs().round(0).astype(int)
    except Exception:
        # Fallback: bootstrap with small noise
        synthetic_nums = seed_df[["units_sold","units_returned","net_units","revenue_inr"]].sample(
            n=target_rows, replace=True).reset_index(drop=True)
        noise = np.random.normal(1, 0.12, synthetic_nums.shape)
        synthetic_nums = (synthetic_nums * noise).abs().round(0).astype(int)

    # Re-attach categorical columns
    skus = seed_df[["sku_id","product_name","abc_class"]].drop_duplicates()
    channels = seed_df["channel"].unique().tolist()

    import itertools
    from datetime import date
    periods = pd.date_range("2022-09", "2025-05", freq="MS").strftime("%Y-%m").tolist()

    rows = []
    for period in periods:
        yr, mo = period.split("-")
        mon = date(int(yr), int(mo), 1).strftime("%b")
        for _, sku_row in skus.iterrows():
            rows.append({
                "period": period, "month": mon, "year": int(yr),
                "sku_id": sku_row.sku_id, "product_name": sku_row.product_name,
                "abc_class": sku_row.abc_class,
                "channel": np.random.choice(channels),
            })

    base = pd.DataFrame(rows).reset_index(drop=True)
    n_grid = len(base)

    # If target_rows exceeds the natural (month × SKU) grid size, cycle the grid
    # rather than silently truncating to whichever is smaller.
    if len(synthetic_nums) > n_grid:
        reps = (len(synthetic_nums) // n_grid) + 1
        base = pd.concat([base] * reps, ignore_index=True)

    n = min(len(base), len(synthetic_nums))
    result = base.iloc[:n].copy()
    result["units_sold"]     = synthetic_nums["units_sold"].values[:n]
    result["units_returned"] = synthetic_nums["units_returned"].values[:n]
    result["net_units"]      = (result["units_sold"] - result["units_returned"]).clip(lower=0)
    result["revenue_inr"]    = synthetic_nums["revenue_inr"].values[:n]
    result.insert(0, "record_id", [f"SYN-{i:05d}" for i in range(n)])
    result.drop(columns=["index"], errors="ignore", inplace=True)
    return result


def _extend_purchase_orders(seed_df: pd.DataFrame, target_rows: int = 800) -> pd.DataFrame:
    """Bootstrap purchase orders with realistic variation, including dates/delays."""
    result = seed_df.sample(n=target_rows, replace=True).reset_index(drop=True)

    price_noise = np.random.normal(1, 0.08, len(result))
    qty_noise   = np.random.normal(1, 0.15, len(result))
    result["qty_ordered"]   = (result["qty_ordered"]   * qty_noise).clip(lower=10).round(0).astype(int)
    result["rate_per_unit_inr"] = (result["rate_per_unit_inr"] * price_noise).round(2)
    result["po_value_inr"]  = (result["qty_ordered"] * result["rate_per_unit_inr"]).round(2)
    result["po_number"]     = [f"SYN-PO-{i:05d}" for i in range(len(result))]

    # Jitter dates so synthetic rows aren't just copies of the same 225 delay patterns
    if "po_date" in result.columns:
        result["po_date"] = pd.to_datetime(result["po_date"]) + pd.to_timedelta(
            np.random.randint(-15, 15, len(result)), unit="D"
        )
    if "delay_days" in result.columns:
        delay_noise = np.random.normal(0, 3, len(result)).round(0).astype(int)
        result["delay_days"] = (result["delay_days"] + delay_noise).clip(lower=-5)

    return result


def main():
    raw = Paths.DATA_RAW
    out = Paths.DATA_PROCESSED
    os.makedirs(out, exist_ok=True)

    print("Loading seed data...")
    demand_seed = pd.read_csv(raw / "demand_history.csv")
    po_seed     = pd.read_csv(raw / "purchase_orders.csv")

    print("Extending demand history → 2,000 rows...")
    extended_demand = _extend_demand_history(demand_seed, target_rows=2000)
    dest = out / "demand_history_extended.csv"
    extended_demand.to_csv(dest, index=False)
    print(f"  Saved {len(extended_demand)} rows → {dest}")

    print("Extending purchase orders → 800 rows...")
    extended_po = _extend_purchase_orders(po_seed, target_rows=800)
    dest2 = out / "purchase_orders_extended.csv"
    extended_po.to_csv(dest2, index=False)
    print(f"  Saved {len(extended_po)} rows → {dest2}")

    print("✓ Synthetic tabular data generation complete.")


if __name__ == "__main__":
    main()
