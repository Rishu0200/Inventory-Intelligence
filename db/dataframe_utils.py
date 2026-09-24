from __future__ import annotations
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import inspect


def query_to_df(session: Session, model) -> pd.DataFrame:
    """Load every row of `model` into a DataFrame with the same column names."""
    rows = session.query(model).all()
    cols = [c.key for c in inspect(model).mapper.column_attrs]
    if not rows:
        return pd.DataFrame(columns=cols)
    data = [{c: getattr(r, c) for c in cols} for r in rows]
    return pd.DataFrame(data)