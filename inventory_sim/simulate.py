import math
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class SimResult:
    cost: float
    fill_rate: float
    inv_avg: float
    dio: float
    ccc: float
    orders_year: float
    holding_cost: float
    shortage_cost: float
    order_cost: float
    transport_cost: float
    total_units_ordered: float
    total_units_demanded: float
    policy_name: str
    policy_type: str
    sku: str
    scenario: str
    backorder_mode: int


def _prepare_demand_model(sales_df: pd.DataFrame, sku: str):
    df = sales_df[sales_df["sku"] == sku].copy()
    if df.empty:
        return {"type": "poisson", "lambda": 0.1, "start": pd.Timestamp("2024-01-01")}
    df["month"] = df["date"].dt.month
    groups = df.groupby("month")["units"].apply(list).to_dict()
    sufficient = all(len(groups.get(m, [])) >= 10 for m in range(1, 13))
    if sufficient:
        return {"type": "bootstrap", "by_month": groups, "start": df["date"].min()}
    lam = max(df["units"].mean(), 0.01)
    return {"type": "poisson", "lambda": lam, "start": df["date"].min()}


def _sample_demand(model, day) -> int:
    if model["type"] == "bootstrap":
        pool = model["by_month"].get(day.month, [])
        if len(pool) == 0:
            lam = np.mean([np.mean(v) for v in model["by_month"].values() if len(v) > 0] or [0.1])
            return int(np.random.poisson(max(lam, 0.01)))
        return int(random.choice(pool))
    return int(np.random.poisson(model["lambda"]))


def _sample_lt(p) -> int:
    return int(max(1, round(np.random.normal(p["lead_time_mean"], p["lead_time_std"]))))


def simulate(sales_df: pd.DataFrame, sku: str, p: dict, policy, ptype: str, horizon_days: int, transport_on: bool) -> SimResult:
    dm = _prepare_demand_model(sales_df, sku)
    start = dm.get("start")
    inv_pos = 0.0
    on_hand = 0.0
    pipe = []
    backlog = 0.0 if int(p.get("backorder_mode", 0)) == 1 else None

    holding = short = order = transp = 0.0
    u_ord = u_dem = u_fill = 0.0
    n_orders = 0

    for t in range(horizon_days):
        # Recepciones
        arr = sum(q for (tt, q) in pipe if tt == t)
        if arr > 0:
            on_hand += arr
            inv_pos += arr
            if backlog is not None and backlog > 0:
                served_back = min(on_hand, backlog)
                on_hand -= served_back
                backlog -= served_back
                u_fill += served_back
        pipe = [(tt, q) for (tt, q) in pipe if tt != t]

        # Demanda
        day = start + timedelta(days=t)
        d = _sample_demand(dm, day)
        u_dem += d
        if backlog is None:  # lost sales
            served = min(on_hand, d)
            on_hand -= served
            u_fill += served
            lost = d - served
            if lost > 0:
                short += p["penalty"] * lost
            inv_pos -= served
        else:  # backorder
            if d <= on_hand:
                on_hand -= d
                u_fill += d
                inv_pos -= d
            else:
                served = on_hand
                u_fill += served
                inv_pos -= served
                backlog += d - served
                on_hand = 0.0
                short += 0.1 * p["penalty"] * (d - served)

        # Reposición
        if ptype == "rQ":
            if inv_pos <= policy.r:
                Q = policy.Q
                lt = _sample_lt(p)
                pipe.append((t + lt, Q))
                inv_pos += Q
                n_orders += 1
                extra = 0.0  # transporte OFF
                order += p["K"]
                transp += extra
                u_ord += Q
        else:
            if inv_pos <= policy.s:
                Q = max(0.0, policy.S - inv_pos)
                if Q > 0:
                    if int(p.get("min_order_qty", 0)):
                        Q = max(Q, p["min_order_qty"])
                    if int(p.get("pallet_size", 0)):
                        Q = math.ceil(Q / p["pallet_size"]) * p["pallet_size"]
                    lt = _sample_lt(p)
                    pipe.append((t + lt, Q))
                    inv_pos += Q
                    n_orders += 1
                    extra = 0.0  # transporte OFF
                    order += p["K"]
                    transp += extra
                    u_ord += Q

        holding += p["h"] * on_hand

    inv_avg = holding / max(p["h"], 1e-9)
    cogs_daily = max(p["COGS"] * max(u_dem, 1e-9) / horizon_days, 1e-9)
    dio = inv_avg / cogs_daily
    fill = u_fill / max(u_dem, 1e-9)
    ccc = dio + p["DSO"] - p["DPO"]
    total = holding + short + order + transp

    return SimResult(
        total,
        fill,
        inv_avg,
        dio,
        ccc,
        float(n_orders),
        holding,
        short,
        order,
        transp,
        u_ord,
        u_dem,
        getattr(policy, "name", "?"),
        ptype,
        sku,
        scenario="TRANSP_OFF",
        backorder_mode=int(p.get("backorder_mode", 0)),
    )
