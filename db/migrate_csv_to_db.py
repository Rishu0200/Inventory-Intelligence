"""
db/migrate_csv_to_db.py — ONE-TIME migration: data/raw/*.csv → database.

Run this once (locally, against Neon/Supabase) after setting DATABASE_URL.
After this, the live app never reads the CSVs again — db/session.py is the
single source of truth. Safe to re-run: it wipes and reloads each table.

Usage:
    python -m db.migrate_csv_to_db
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from datetime import datetime

from config import Paths
from db.session import engine, init_db, get_session
from db.models import (
    Base, DemandRecord, InventorySnapshot, Supplier, SupplierTerm,
    PurchaseOrder,
)


def _to_date(series: pd.Series):
    return pd.to_datetime(series, errors="coerce").dt.date


def migrate_demand():
    df = pd.read_csv(Paths.DATA_RAW / "demand_history.csv")
    df["period"] = _to_date(df["period"])
    rows = df.to_dict("records")
    with get_session() as s:
        s.query(DemandRecord).delete()
        s.bulk_insert_mappings(DemandRecord, rows)
    print(f"  ✓ demand_records: {len(rows)} rows")


def migrate_inventory():
    df = pd.read_csv(Paths.DATA_RAW / "inventory_history.csv")
    df["snapshot_date"] = _to_date(df["snapshot_date"])
    rows = df.to_dict("records")
    with get_session() as s:
        s.query(InventorySnapshot).delete()
        s.bulk_insert_mappings(InventorySnapshot, rows)
    print(f"  ✓ inventory_snapshots: {len(rows)} rows")


def migrate_suppliers():
    df = pd.read_csv(Paths.DATA_RAW / "supplier_directory.csv")
    rows = df.to_dict("records")
    with get_session() as s:
        s.query(Supplier).delete()
        s.bulk_insert_mappings(Supplier, rows)
    print(f"  ✓ suppliers: {len(rows)} rows")


def migrate_supplier_terms():
    df = pd.read_csv(Paths.DATA_RAW / "supplier_terms.csv")
    df["price_valid_until"] = _to_date(df.get("price_valid_until", pd.Series(dtype=str)))
    df["last_audit_date"]   = _to_date(df.get("last_audit_date", pd.Series(dtype=str)))
    rows = df.to_dict("records")
    with get_session() as s:
        s.query(SupplierTerm).delete()
        s.bulk_insert_mappings(SupplierTerm, rows)
    print(f"  ✓ supplier_terms: {len(rows)} rows")


def migrate_purchase_orders():
    df = pd.read_csv(Paths.DATA_RAW / "purchase_orders.csv")
    for col in ["po_date", "expected_delivery", "actual_delivery"]:
        df[col] = _to_date(df[col])
    rows = df.to_dict("records")
    with get_session() as s:
        s.query(PurchaseOrder).delete()
        s.bulk_insert_mappings(PurchaseOrder, rows)
    print(f"  ✓ purchase_orders: {len(rows)} rows")


def main():
    print(f"Target database: {engine.url}")
    print("Creating tables (if not present)...")
    init_db()

    print("\nMigrating suppliers first (FK dependency for supplier_terms/purchase_orders)...")
    migrate_suppliers()
    migrate_supplier_terms()
    migrate_purchase_orders()
    migrate_demand()
    migrate_inventory()

    print("\n✅ Migration complete.")


if __name__ == "__main__":
    main()