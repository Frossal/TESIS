from dataclasses import dataclass
import math
import pandas as pd


@dataclass
class PolicyRQ:
    r: float
    Q: float
    name: str


@dataclass
class PolicySS:
    s: float
    S: float
    name: str


def _daily_std(series: pd.Series) -> float:
    return float(series.std(ddof=1) if series.nunique() > 1 else max(0.2, series.mean() ** 0.5))


def generate_candidates(sales_df: pd.DataFrame, sku: str, params: dict):
    df = sales_df[sales_df["sku"] == sku]
    dm = df["units"].mean() if not df.empty else 0.3
    ds = _daily_std(df["units"]) if not df.empty else 0.6
    muL = dm * params["lead_time_mean"]
    sigmaL = ((ds ** 2) * params["lead_time_mean"] + (dm ** 2) * (params["lead_time_std"] ** 2)) ** 0.5

    # nivel de servicio para el punto de reorden (SLT via z)
    z_map = {0.95: 1.645, 0.98: 2.054, 0.99: 2.326}

    demand_annual = max(dm * 365, 1e-6)
    eoq = (2 * params["K"] * demand_annual / max(params["h"] * 365, 1e-6)) ** 0.5

    rq, ss = [], []
    for alpha in [0.95, 0.98, 0.99]:
        z = z_map[alpha]
        r = max(0.0, muL + z * sigmaL)
        for mult in [0.6, 1.0, 1.5]:
            Q = max(1.0, eoq * mult)
            if params["min_order_qty"]:
                Q = max(Q, params["min_order_qty"])
            if params["pallet_size"]:
                from math import ceil

                Q = ceil(Q / params["pallet_size"]) * params["pallet_size"]
            rq.append(PolicyRQ(r, Q, f"rQ_a{alpha:.2f}_Qx{mult:.1f}"))
        for mult in [1.0, 1.5, 2.0]:
            from math import ceil

            jump = max(1.0, eoq * mult)
            s = r
            S = s + jump
            if params["pallet_size"]:
                S = s + ceil(jump / params["pallet_size"]) * params["pallet_size"]
            if params["min_order_qty"]:
                S = max(S, s + params["min_order_qty"])
            ss.append(PolicySS(s, S, f"sS_a{alpha:.2f}_Jumpx{mult:.1f}"))
    return {"rQ": rq, "sS": ss}
