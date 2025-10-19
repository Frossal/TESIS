# inventory_sim/pipeline.py
import numpy as np, random, pandas as pd
from datetime import datetime
from calendar import monthrange

from inventory_sim.config import SimConfig
from inventory_sim.dataio import load_all, load_prices
from inventory_sim.policies import generate_candidates
from inventory_sim.simulate import simulate
from inventory_sim.rs import rank_and_select


# ------------------------------------------------------------
# Utilidades de ventana temporal (Año = Y, Mes = M)
# ------------------------------------------------------------

def _calc_window(y_target: int, month_cut: int | None):
    """
    Devuelve (base_start, base_end, m_end) según el contrato:
      - Ventana = [Y-1-01-01 .. fin de (Y, M-1)]
      - Si M es None:
          * Si Y == año actual -> m_end = mes actual
          * Si Y < año actual   -> m_end = 12
      - Si m_end == 1 -> fin = 31/dic/(Y-1)
    """
    today = datetime.today()
    if month_cut is None:
        m_end = today.month if y_target == today.year else 12
    else:
        m_end = int(month_cut)

    if m_end == 1:
        base_end = datetime(y_target - 1, 12, 31)
    else:
        prev_m = m_end - 1
        last_day = monthrange(y_target, prev_m)[1]
        base_end = datetime(y_target, prev_m, last_day)

    base_start = datetime(y_target - 1, 1, 1)
    return base_start, base_end, m_end


def _filter_by_window(df: pd.DataFrame, date_col: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Filtra un DataFrame por fecha [start..end] si existe la columna date_col."""
    if df is None or df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out[(out[date_col] >= start) & (out[date_col] <= end)].reset_index(drop=True)


# ------------------------------------------------------------
# Auxiliares existentes (se mantienen)
# ------------------------------------------------------------

def pick_topk_off(winners_off: pd.DataFrame, k: int, service_min: float) -> pd.DataFrame:
    ok  = winners_off[winners_off["fill_rate"] >= service_min].sort_values("cost_mean", ascending=True)
    bad = winners_off[winners_off["fill_rate"] <  service_min].sort_values("cost_mean", ascending=True)
    top = pd.concat([ok, bad], ignore_index=True).head(k)
    top = top.rename(columns={
        "type":"type_OFF","name":"name_OFF","cost_mean":"cost_mean_OFF",
        "fill_rate":"fill_rate_OFF","dio":"dio_OFF","ccc":"ccc_OFF"
    })
    return top[["sku","type_OFF","name_OFF","cost_mean_OFF","fill_rate_OFF","dio_OFF","ccc_OFF"]]


def _price_for_year(prices: pd.DataFrame | None, sku: str, year: int) -> float:
    if prices is None:
        return 0.0
    dfy = prices[(prices["sku"] == sku) & (prices["year"] == year)]
    if not dfy.empty:
        return float(dfy["unit_price"].mean())
    dfh = prices[prices["sku"] == sku]
    if not dfh.empty:
        return float(dfh["unit_price"].mean())
    return 0.0


# ------------------------------------------------------------
# Pipeline principal OFF usando ventana [Y-1 .. fin (Y, M-1)]
# ------------------------------------------------------------

def run_off_pipeline(
    data_dir: str,
    cfg: SimConfig,
    max_skus: int = 0,
    log_every: int = 10,
    y_target: int | None = None,     # << nuevo: año objetivo requerido por UI
    month_cut: int | None = None,    # << nuevo: mes de corte (None -> reglas por defecto)
):
    """
    Corre R&S OFF usando SOLO la ventana temporal definida por (y_target, month_cut).
    Devuelve dict con winners_off, cands_off, rep_df, y_target, month_cut y base_window.
    Si hay prices.csv, computa utilidad simulada para el año y_target.
    """
    # Semillas
    np.random.seed(cfg.seed_numpy)
    random.seed(cfg.seed_random)

    # 1) Cargar datos crudos
    sales, costs = load_all(data_dir)

    # 1b) Cargar precios (si existen)
    price_path = f"{data_dir}/prices.csv"
    prices_df = None
    try:
        prices_df = load_prices(price_path)
    except Exception:
        prices_df = None

    # 2) Parámetros temporales
    if y_target is None:
        y_target = datetime.today().year  # seguridad (UI ya fija default 2025)
    base_start, base_end, m_end = _calc_window(y_target, month_cut)

    # 3) Filtrar por ventana
    sales_hist = _filter_by_window(sales, "date", base_start, base_end)
    # Si prices/cogs/lead_times tuvieran fecha transaccional, filtrarlos igual:
    # (en este proyecto costs es maestro por SKU, sin 'date')
    # prices_df podría tener 'date' o 'year'; si tiene 'date', filtramos,
    # si no, lo dejamos para lookup por 'year' en _price_for_year.

    # 4) SKUs a procesar
    skus = sorted(costs["sku"].unique().tolist())
    if max_skus and max_skus > 0:
        skus = skus[:max_skus]

    # 5) Loop de simulación OFF
    cands_off_list, rep_all, winners_off = [], [], []

    def sim_off(sku, params, pol, ptype, sales_base):
        return simulate(sales_base, sku, params, pol, ptype, cfg.horizon_days, transport_on=False)

    for idx, sku in enumerate(skus, 1):
        params = costs[costs["sku"] == sku].iloc[0].to_dict()

        # Generar candidatos usando SOLO el histórico de la ventana
        cands = generate_candidates(sales_hist, sku, params)
        cand_list = [("rQ", p) for p in cands["rQ"]] + [("sS", p) for p in cands["sS"]]

        # Selección con rank & select (simulando SIEMPRE sobre la ventana filtrada)
        perf_off, rep_off = rank_and_select(lambda s, pr, pol, pt: sim_off(s, pr, pol, pt, sales_hist),
                                            sku, params, cand_list, cfg)

        out = perf_off.copy()
        out["cost_ci_semi"] = 1.96 * out["sd"].fillna(0) / (out["n"] ** 0.5)
        out["cost_ci_lo"] = out["mean"] - out["cost_ci_semi"]
        out["cost_ci_hi"] = out["mean"] + out["cost_ci_semi"]
        out = out.rename(columns={"mean": "cost_mean", "fill": "fill_rate"})
        out["sku"] = sku
        cands_off_list.append(
            out[["sku", "type", "name", "n", "cost_mean", "cost_ci_lo", "cost_ci_hi", "cost_ci_semi", "fill_rate", "dio", "ccc"]]
        )

        rep_all.append(rep_off.assign(sku=sku, scenario="OFF"))

        # Ganador por score (penaliza no cumplir servicio)
        pf = perf_off.copy()
        pf["score"] = pf["mean"] + np.where(pf["fill"] >= cfg.service_min, 0, (cfg.service_min - pf["fill"]) * 1e6)
        row = pf.sort_values("score").iloc[0]

        winner = {
            "sku": sku, "type": row["type"], "name": row["name"],
            "cost_mean": row["mean"], "fill_rate": row["fill"],
            "dio": row["dio"], "ccc": row["ccc"], "n": row["n"], "sd": row.get("sd", 0.0)
        }

        # Si hay año objetivo, calculamos métricas anuales simuladas + profit
        pol_obj = next(p for (t, p) in cand_list if (t == row["type"] and p.name == row["name"]))
        r = sim_off(sku, params, pol_obj, row["type"], sales_hist)
        dem = float(r.total_units_demanded)
        sold = float(r.fill_rate * r.total_units_demanded)
        unit_price = _price_for_year(prices_df, sku, y_target)
        unit_cost  = float(params.get("COGS", 0.0))
        profit = (unit_price - unit_cost) * sold
        winner.update({
            "sim_year": y_target,
            "units_demand_sim": dem,
            "units_sold_sim": sold,
            "unit_price_sim": unit_price,
            "unit_cost": unit_cost,
            "profit_sim": profit
        })

        winners_off.append(winner)

        if log_every and idx % log_every == 0:
            print(f"[OFF] {idx}/{len(skus)} procesados…")

    # 6) Empaquetar resultados
    cands_off = pd.concat(cands_off_list, ignore_index=True)
    rep_df = pd.concat(rep_all, ignore_index=True)
    winners = pd.DataFrame(winners_off)

    return {
        "cands_off": cands_off,
        "rep_df": rep_df,
        "winners_off": winners,
        "prices": prices_df,
        "sales_hist": sales_hist,                 # histórico realmente usado
        "y_target": y_target,
        "month_cut": month_cut,
        "base_window": (_ts(base_start), _ts(base_end)),
    }


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


# ===================== FORECAST MENSUAL (con mes de inicio/fin) =====================

def _month_weights_for_sku(sales_hist: pd.DataFrame, sku: str) -> np.ndarray:
    """
    Estacionalidad mensual (12 pesos que suman 1) por SKU, usando el histórico filtrado (sales_hist).
    Si no hay histórico para el SKU, devuelve uniforme (1/12).
    """
    df = sales_hist[sales_hist["sku"] == sku].copy()
    if df.empty:
        return np.ones(12) / 12.0
    df["month"] = df["date"].dt.month
    m = df.groupby("month")["units"].mean()
    w = np.array([m.get(i, 0.0) for i in range(1, 13)], dtype=float)
    if w.sum() <= 0:
        return np.ones(12) / 12.0
    return w / w.sum()


def build_monthly_forecast(
    res: dict,
    top_skus: list[str],
    year: int,
    month_start: int | None = None,   # << nuevo: mes inicial (ej. M seleccionado)
    month_end: int = 12,              # << nuevo: mes final (por defecto diciembre)
    use_sold: bool = True
) -> pd.DataFrame:
    """
    Proyección mensual por SKU para 'year' en el rango [month_start..month_end].
      - Volumen anual objetivo: units_sold_sim (si use_sold) o units_demand_sim.
      - Distribución mensual: estacionalidad desde sales_hist (ya filtrado por ventana).
    Columnas: sku, date, forecast_units, basis ('sold'/'demand').
    """
    winners = res["winners_off"].copy()
    sales_hist = res["sales_hist"]

    basis_col = "units_sold_sim" if use_sold else "units_demand_sim"
    ms = 1 if (month_start is None or month_start < 1) else int(month_start)
    me = 12 if (month_end   is None or month_end   > 12) else int(month_end)
    months = list(range(ms, me + 1))

    rows = []
    for sku in top_skus:
        row = winners[winners["sku"] == sku]
        if row.empty:
            continue
        total = float(row[basis_col].iloc[0]) if basis_col in row.columns else None
        if total is None or np.isnan(total):
            # fallback: usar suma del último año disponible en el histórico filtrado
            df_s = sales_hist[(sales_hist["sku"] == sku)]
            if df_s.empty:
                continue
            last_year = df_s["date"].dt.year.max()
            total = float(df_s[df_s["date"].dt.year == last_year]["units"].sum())
        w = _month_weights_for_sku(sales_hist, sku)

        # Repartimos total en 12 y luego filtramos a [ms..me]
        monthly = (w * total).astype(float)
        for m in months:
            rows.append({
                "sku": sku,
                "date": pd.Timestamp(year=int(year), month=m, day=1),
                "forecast_units": monthly[m-1],
                "basis": "sold" if use_sold else "demand"
            })
    return pd.DataFrame(rows)
