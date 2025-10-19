import os
import pandas as pd
import unicodedata
import re


def _norm(s: str) -> str:
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def load_sales(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"fecha", "date"}:
            rename[c] = "date"
        elif lc in {"sku", "item", "product", "codigo"}:
            rename[c] = "sku"
        elif lc in {"unidades", "units", "qty"}:
            rename[c] = "units"
    df = df.rename(columns=rename)[["date", "sku", "units"]].copy()
    df["sku"] = df["sku"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["units"] = pd.to_numeric(df["units"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    return df


def load_lead_times(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {_norm(c): c for c in df.columns}

    # CASO A: observaciones individuales "sku, lead_time_days"
    if "lead_time_days" in cols and any(k in cols for k in ["sku", "item", "product", "codigo"]):
        sku_col = cols.get("sku") or cols.get("item") or cols.get("product") or cols.get("codigo")
        lt_col = cols["lead_time_days"]

        tmp = df[[sku_col, lt_col]].copy()
        tmp.columns = ["sku", "lead_time_days"]
        tmp["sku"] = tmp["sku"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        tmp["lead_time_days"] = pd.to_numeric(tmp["lead_time_days"], errors="coerce")

        agg = tmp.groupby("sku", as_index=False).agg(
            lead_time_mean=("lead_time_days", "mean"),
            lead_time_std=("lead_time_days", "std"),
        )
        agg["lead_time_mean"] = agg["lead_time_mean"].fillna(3)
        agg["lead_time_std"] = agg["lead_time_std"].fillna(1.0)
        return agg

    # CASO B: ya vienen mean/std
    lt_mean = cols.get("lead_time_mean") or cols.get("lt_mean") or cols.get("lt_avg") or cols.get("mean")
    lt_std = cols.get("lead_time_std") or cols.get("lt_std") or cols.get("std")
    sku_col = cols.get("sku") or cols.get("item") or cols.get("product") or cols.get("codigo")

    if sku_col and lt_mean:
        out = df[[sku_col, lt_mean]].copy()
        out.columns = ["sku", "lead_time_mean"]
        out["sku"] = out["sku"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        out["lead_time_mean"] = pd.to_numeric(out["lead_time_mean"], errors="coerce").fillna(3)
        if lt_std:
            out["lead_time_std"] = pd.to_numeric(df[lt_std], errors="coerce").fillna(1.0)
        else:
            out["lead_time_std"] = 1.0
        return out

    raise ValueError("lead_times.csv debe tener (sku + lead_time_days) o (sku + lead_time_mean [+ lead_time_std]).")


def load_cogs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {_norm(c): c for c in df.columns}

    sku = cols.get("sku") or cols.get("item") or cols.get("product") or cols.get("codigo")
    if not sku:
        raise ValueError("cogs.csv debe contener columna 'sku' (o item/product/codigo)")

    cogs = (
        cols.get("cogs")
        or cols.get("cogs_per_unit")
        or cols.get("unit_cost")
        or cols.get("unit_cost_usd")
        or cols.get("unitcost")
        or cols.get("costo_unitario")
    )
    if not cogs:
        raise ValueError("cogs.csv debe contener columna de costo (COGS/cogs_per_unit/unit_cost/costo_unitario)")

    out = df[[sku, cogs]].copy()
    out.columns = ["sku", "COGS"]
    out["sku"] = out["sku"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    out["COGS"] = pd.to_numeric(out["COGS"], errors="coerce").fillna(0.0)

    # Defaults
    defaults = {
        "h": 0.04 / 30,
        "K": 60.0,
        "penalty": 5.0,
        "DSO": 15,
        "DPO": 45,
        "min_order_qty": 0,
        "pallet_size": 0,
        "tarifa_pallet": 0.0,
        "v_unit": 0.0,
        "backorder_mode": 0,
    }
    for k, v in defaults.items():
        if k not in df.columns:
            out[k] = v
        else:
            out[k] = df[cols.get(k, k)]

    keep = [
        "sku",
        "COGS",
        "h",
        "K",
        "penalty",
        "DSO",
        "DPO",
        "min_order_qty",
        "pallet_size",
        "tarifa_pallet",
        "v_unit",
        "backorder_mode",
    ]
    return out[keep].copy()


def load_all(data_dir: str):
    sales = load_sales(os.path.join(data_dir, "sales.csv"))
    lts = load_lead_times(os.path.join(data_dir, "lead_times.csv"))
    cogs = load_cogs(os.path.join(data_dir, "cogs.csv"))
    cost = cogs.merge(lts, on="sku", how="left").fillna({"lead_time_mean": 3, "lead_time_std": 1})
    return sales, cost


def load_prices(path: str) -> pd.DataFrame:
    """Lee prices.csv con columnas: sku, date, unit_price"""
    df = pd.read_csv(path, parse_dates=["date"])
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc in {"sku", "item", "product", "codigo"}:
            rename[c] = "sku"
        elif lc in {"date", "fecha"}:
            rename[c] = "date"
        elif lc in {"unit_price", "price", "precio"}:
            rename[c] = "unit_price"
    df = df.rename(columns=rename)[["sku", "date", "unit_price"]].copy()
    df["sku"] = df["sku"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0.0)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df
