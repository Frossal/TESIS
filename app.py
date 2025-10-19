import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from inventory_sim.config import SimConfig
from inventory_sim.pipeline import run_off_pipeline, build_monthly_forecast
from inventory_sim.reports import save_reports_off

# ======================== Config de página ========================
st.set_page_config(page_title="Simulador Inventario — OFF", layout="wide")
st.title("🧮 Simulador de Inventario — OFF (sin transporte)")

# ======================== Sidebar (parámetros) ====================
with st.sidebar:
    st.header("⚙️ Parámetros")
    data_dir  = st.text_input("Carpeta de datos", value="data")
    service   = st.slider("Meta de Fill_rate", 0.80, 0.99, 0.95, 0.01)
    horizon   = st.number_input("Horizonte (días)", 7, 730, 365, 1)
    init_reps = st.number_input("Réplicas iniciales", 2, 50, 12, 1)
    max_reps  = st.number_input("Réplicas máx. por candidato", 5, 200, 60, 5)
    ci_rel    = st.number_input("IC95 relativo (costo)", 0.005, 0.2, 0.03, 0.005, format="%.3f")
    seed      = st.number_input("Semilla", 0, 999999, 42, 1)

    st.markdown("---")
    st.subheader("🎯 Alcance")
    max_skus = st.number_input("SKUs a procesar (0 = todos)", 0, 100000, 0, 1)

    st.markdown("---")
    st.subheader("🗓️ Año de simulación")
    sim_year = st.selectbox("Año objetivo", options=[2024, 2025], index=1)

    st.subheader("🗓️ Mes objetivo (corte)")
    month_cut = st.selectbox("Mes de corte", options=list(range(1,13)), index=0, format_func=lambda m: f"{m:02d}")

    st.markdown("---")
    st.subheader("📊 Salida")
    top_table_n = st.number_input("Top en Tabla", 5, 200, 50, 5)
    top_charts  = st.number_input("Top en Gráficos", 5, 50, 10, 5)
    out_dir     = st.text_input("Carpeta de exportación", value="out")

    st.markdown("---")
    st.subheader("⚖️ Pesos del ranking")
    w_cost = st.slider("Peso costo", 0.0, 1.0, 0.5, 0.05)
    w_fill = st.slider("Peso penalización fill", 0.0, 1.0, 0.3, 0.05)
    w_dio  = st.slider("Peso DIO", 0.0, 1.0, 0.2, 0.05)

    st.session_state["w_cost"] = w_cost
    st.session_state["w_fill_pen"] = w_fill
    st.session_state["w_dio"] = w_dio
    st.session_state["target_fill_rate"] = service * 100.0

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    run_btn    = c1.button("▶️ Ejecutar")
    all_btn    = c2.button("▶️ Todo")
    export_btn = c3.button("💾 Exportar")

# ======================== Estado ================================
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "last_cfg" not in st.session_state:
    st.session_state.last_cfg = None

def run_now(proc_all: bool):
    cfg = SimConfig(
        horizon_days=int(horizon), service_min=float(service),
        init_reps=int(init_reps), max_reps=int(max_reps), ci_rel_target=float(ci_rel),
        seed_numpy=int(seed), seed_random=int(seed)
    )
    st.session_state.last_cfg = cfg.__dict__ | {"y_target": sim_year, "month_cut": int(month_cut)}
    sel = 0 if proc_all else int(max_skus)
    with st.status("Calculando…", expanded=True) as status:
        st.write(f"📥 Leyendo datos desde `{data_dir}`")
        res = run_off_pipeline(
            data_dir=data_dir,
            cfg=cfg,
            max_skus=sel,
            log_every=10,
            y_target=sim_year,
            month_cut=int(month_cut)
        )
        st.session_state.last_results = res
        status.update(label="✅ Listo", state="complete")

if run_btn:
    run_now(proc_all=False)
if all_btn:
    run_now(proc_all=True)

res = st.session_state.last_results
if res is None:
    st.info("Configura parámetros y pulsa **Ejecutar** o **Todo**.")
    st.stop()

# ======================== KPI de cabecera ======================
winners = res["winners_off"].copy()

def _minmax(s: pd.Series) -> pd.Series:
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())

rank_df_base = winners.rename(columns={
    "type":"type_OFF","name":"name_OFF",
    "cost_mean":"cost_mean_OFF","fill_rate":"fill_rate_OFF",
    "dio":"dio_OFF","ccc":"ccc_OFF"
}).copy()

rank_df_base["cost_norm"] = _minmax(rank_df_base["cost_mean_OFF"])
rank_df_base["dio_norm"]  = _minmax(rank_df_base["dio_OFF"])
rank_df_base["fill_penalty"] = (service - rank_df_base["fill_rate_OFF"]).clip(lower=0.0)
rank_df_base["score"] = (
    w_cost*rank_df_base["cost_norm"]
  + w_fill*rank_df_base["fill_penalty"]
  + w_dio *rank_df_base["dio_norm"]
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("SKUs procesados", len(winners))
porc_ok = 100.0 * (winners["fill_rate"] >= service).mean() if not winners.empty else 0.0
k2.metric("SKUs con Fill_rate ≥ meta", f"{porc_ok:.1f}%")
k3.metric("Meta de Fill_rate", f"{service:.0%}")
k4.metric("Año/Mes de corte", f"{sim_year}-{int(month_cut):02d}")

# ======================== Helpers de fecha/KPI ===================
def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    elif "dt" in d.columns:
        d["date"] = pd.to_datetime(d["dt"], errors="coerce")
    elif "period" in d.columns:
        d["date"] = pd.to_datetime(d["period"], errors="coerce")
    elif "periodo" in d.columns:
        d["date"] = pd.to_datetime(d["periodo"], errors="coerce")
    elif {"year", "month"}.issubset(d.columns):
        d["date"] = pd.to_datetime(dict(year=d["year"].astype(int), month=d["month"].astype(int), day=1), errors="coerce")
    else:
        d["date"] = pd.Timestamp(f"{sim_year}-{int(month_cut):02d}-01")
    d["date"] = d["date"].fillna(pd.Timestamp(f"{sim_year}-{int(month_cut):02d}-01"))
    return d

def _safe_norm(x: pd.Series) -> pd.Series:
    if x.empty or (x.max() == x.min()):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - x.min()) / (x.max() - x.min())

def _compute_score_from_aggregates(df_agg: pd.DataFrame) -> pd.DataFrame:
    out = df_agg.copy()
    
    # 💡 ESTO YA DEBE ESTAR HECHO: Se rellena con 0.0 ANTES de normalizar para que el score sea numérico.
    cost_norm = _safe_norm(out["cost_mean_OFF"].fillna(0.0))
    dio_norm  = _safe_norm(out["dio_OFF"].fillna(0.0))
    fill_norm = _safe_norm(out["fill_rate_OFF"].fillna(0.0))
    
    out["score"] = (
        st.session_state["w_cost"] * cost_norm
      + st.session_state["w_fill_pen"] * (1.0 - fill_norm)  # penaliza fill bajo
      + st.session_state["w_dio"]  * dio_norm
    )
    # descomposición para "why_not"
    out["_c_cost"] = st.session_state["w_cost"] * cost_norm
    out["_c_fill"] = st.session_state["w_fill_pen"] * (1.0 - fill_norm)
    out["_c_dio"]  = st.session_state["w_dio"]  * dio_norm
    return out

def _normalize_kpi_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    rename_map = {"cost_mean":"cost_mean_OFF", "fill_rate":"fill_rate_OFF", "dio":"dio_OFF", "type":"type_OFF"}
    for src, dst in rename_map.items():
        if (src in d.columns) and (dst not in d.columns):
            d.rename(columns={src: dst}, inplace=True)
    d = _ensure_date_column(d)
    for col in ["cost_mean_OFF", "fill_rate_OFF", "dio_OFF"]:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if "type_OFF" not in d.columns:
        d["type_OFF"] = d.get("type", np.nan)
    return d

def _aggregate_months_view(base_df: pd.DataFrame, months_selected: list[int]) -> pd.DataFrame:
    """
    - Promedio por SKU de cost_mean_OFF, fill_rate_OFF y dio_OFF (ignora NaN).
    - type_OFF: último no nulo.
    - Luego se **recalcula el score** desde estos agregados.
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame(columns=["sku","score","cost_mean_OFF","fill_rate_OFF","dio_OFF","type_OFF"])

    d = base_df.copy()

    # mes numérico para filtrar
    if "month" not in d.columns:
        d["month"] = d["date"].dt.month

    # filtrar meses (si hay)
    if months_selected:
        d = d[d["month"].isin(months_selected)]

    # si no hay filas, devolvemos todos los SKUs con NaN
    if d.empty:
        skus = base_df["sku"].drop_duplicates().sort_values() if "sku" in base_df.columns else pd.Series(dtype=str)
        out = pd.DataFrame({
            "sku": skus,
            "cost_mean_OFF": np.nan,
            "fill_rate_OFF": np.nan,
            "dio_OFF": np.nan,
            "type_OFF": ""
        })
        out = _compute_score_from_aggregates(out)
        for col in ("cost_mean_OFF", "fill_rate_OFF", "dio_OFF"):
            if col in out.columns:
                out[col] = out[col].fillna(0.0).astype(float)
        return out[["sku","score","cost_mean_OFF","fill_rate_OFF","dio_OFF","type_OFF"]]

    def _nanmean(x):
        return np.nanmean(x.values.astype(float)) if len(x) else np.nan

    agg = d.groupby("sku", as_index=False).agg({
        "cost_mean_OFF": _nanmean,
        "fill_rate_OFF": _nanmean,
        "dio_OFF": _nanmean,
        "type_OFF": lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else ""
    })

    # asegurar que todos los SKUs existan en el agregado (aunque con NaN)
    all_skus = base_df["sku"].drop_duplicates()
    agg = all_skus.to_frame().merge(agg, on="sku", how="left")

    # 💡 CORRECCIÓN CLAVE: Rellenar NaN en las métricas numéricas para que se muestren '0'
    # en lugar de 'None' cuando el SKU no tiene datos en los meses seleccionados.
    agg["cost_mean_OFF"] = agg["cost_mean_OFF"].fillna(0.0).astype(float)
    agg["fill_rate_OFF"] = agg["fill_rate_OFF"].fillna(0.0).astype(float)
    agg["dio_OFF"] = agg["dio_OFF"].fillna(0.0).astype(float)
    
    # Rellenar type_OFF con un string vacío si es necesario.
    agg["type_OFF"] = agg["type_OFF"].fillna("")

    agg = _compute_score_from_aggregates(agg)
    for col in ("cost_mean_OFF", "fill_rate_OFF", "dio_OFF"):
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0).astype(float)
    return agg[["sku","score","cost_mean_OFF","fill_rate_OFF","dio_OFF","type_OFF"]]

# ===================== Helpers: agregación y score =====================

import numpy as np
import pandas as pd

def _coerce_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Asegura que df[col] sea datetime64; no falla si no existe."""
    if col in df.columns and not np.issubdtype(df[col].dtype, np.datetime64):
        with pd.option_context("mode.chained_assignment", None):
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _first_non_null(s: pd.Series):
    """Primer valor no nulo (para type_OFF)."""
    try:
        return next(x for x in s.values if pd.notna(x))
    except StopIteration:
        return None

def aggregate_for_months(
    base_df: pd.DataFrame,
    months_selected: list[int],
    cols_keep: list[str] = ("sku", "date", "cost_mean_OFF", "fill_rate_OFF", "dio_OFF", "type_OFF"),
) -> pd.DataFrame:
    """
    Devuelve un df por SKU con el promedio de las métricas de los meses seleccionados.
    Si un SKU sólo tiene un mes, se usa ese valor (no se rellena a 0).
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame(columns=["sku", "cost_mean_OFF", "fill_rate_OFF", "dio_OFF", "type_OFF"])

    # trabajar sólo con columnas relevantes si existen
    cols_exist = [c for c in cols_keep if c in base_df.columns]
    df = base_df[cols_exist].copy()

    # date -> month para filtrar
    df = _coerce_datetime(df, "date")
    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df = df[df["month"].isin(months_selected)]
    # Si no hay columna date, asumimos que ya viene filtrado arriba (no explota)

    if df.empty:
        # devolvemos la lista de SKUs con NaN en métricas (se completará más abajo)
        all_skus = sorted(base_df["sku"].dropna().unique().tolist())
        return pd.DataFrame({"sku": all_skus})

    # promedio por SKU ignorando NaN
    agg_map = {}
    if "cost_mean_OFF" in df.columns:  agg_map["cost_mean_OFF"] = "mean"
    if "fill_rate_OFF" in df.columns:  agg_map["fill_rate_OFF"] = "mean"
    if "dio_OFF" in df.columns:        agg_map["dio_OFF"] = "mean"
    if "type_OFF" in df.columns:       agg_map["type_OFF"] = _first_non_null

    grouped = (
        df.groupby("sku", as_index=False)
          .agg(agg_map) if agg_map else df[["sku"]].drop_duplicates()
    )

    return grouped

def normalize_minmax(s: pd.Series) -> pd.Series:
    """Min-Max robusto; si todo es igual devuelve 0s."""
    s = s.astype(float)
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def add_dynamic_score(
    df: pd.DataFrame,
    w_cost: float,
    w_fill_penalty: float,
    w_dio: float,
) -> pd.DataFrame:
    """
    Añade columna 'score' = w_cost*cost_norm + w_fill_penalty*(1-fill_rate)_norm + w_dio*dio_norm
    donde cada término está min-max normalizado.
    """
    out = df.copy()
    # columnas presentes => normalizamos
    cost = out["cost_mean_OFF"] if "cost_mean_OFF" in out.columns else pd.Series(0.0, index=out.index)
    dio  = out["dio_OFF"]        if "dio_OFF" in out.columns        else pd.Series(0.0, index=out.index)
    fr   = out["fill_rate_OFF"]  if "fill_rate_OFF" in out.columns  else pd.Series(0.0, index=out.index)

    # penalización por fill: mayor si el fill es bajo
    fr_penalty = 1.0 - fr.clip(lower=0.0, upper=1.0).astype(float)

    cost_n = normalize_minmax(cost.fillna(np.nan))
    dio_n  = normalize_minmax(dio.fillna(np.nan))
    fr_n   = normalize_minmax(fr_penalty.fillna(np.nan))

    out["score"] = (
        w_cost * cost_n +
        w_fill_penalty * fr_n +
        w_dio * dio_n
    ).astype(float)

    # ranking (menor score = más recomendable)
    out["ranking_general"] = out["score"].rank(method="dense", ascending=True).astype(int)
    out.sort_values(["score", "sku"], inplace=True, ascending=[True, True])
    return out


# ======================== Selector de meses ======================
st.subheader("📅 Meses a considerar (múltiple)")
if ("months_initialized" not in st.session_state) or (st.session_state.get("last_cut") != int(month_cut)):
    st.session_state.last_cut = int(month_cut)
    st.session_state.months_selected    = [int(month_cut)]
    st.session_state.months_multiselect = [f"{int(month_cut):02d}"]
    st.session_state.months_initialized = True

allowed_options = [f"{m:02d}" for m in range(int(month_cut), 13)]

def _select_all_months():
    st.session_state.months_selected    = list(range(int(month_cut), 13))
    st.session_state.months_multiselect = [f"{m:02d}" for m in st.session_state.months_selected]

c_ms, c_btn = st.columns([0.85, 0.15])
with c_ms:
    if "months_multiselect" not in st.session_state:
        st.session_state.months_multiselect = [f"{int(month_cut):02d}"]
    st.multiselect(" ", options=allowed_options, key="months_multiselect", label_visibility="hidden")
with c_btn:
    st.button("Seleccionar todo", on_click=_select_all_months, use_container_width=True)

st.session_state.months_selected = sorted({int(m) for m in st.session_state.months_multiselect})
if not st.session_state.months_selected:
    st.warning("Selecciona al menos un mes (desde el corte en adelante).")
    st.stop()
months_selected: list[int] = st.session_state.months_selected[:]

# =================== Base normalizada + vista agregada ==============
base_df = _normalize_kpi_columns(res["rep_df"].copy())   # 👈 normaliza nombres y fecha
view_df = _aggregate_months_view(base_df, months_selected)
view_df = view_df.sort_values(["score", "sku"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
view_df["ranking_general"] = np.arange(1, len(view_df) + 1)

# ======================= Tablas =======================
TOP_N = int(top_table_n)

st.subheader("🏁 Listado General")
search_general = st.text_input("Buscar SKU en General:", key="search_general")
df_general = view_df.copy()
if search_general.strip():
    q = search_general.strip().lower()
    df_general = df_general[df_general["sku"].str.lower().str.contains(q, na=False)]
cols_general = ["ranking_general","sku","score","cost_mean_OFF","fill_rate_OFF","dio_OFF","type_OFF"]
st.dataframe(df_general[cols_general], use_container_width=True, hide_index=True)

st.subheader(f"📁 Top {TOP_N} recomendados (según ranking)")
search_top = st.text_input("Buscar SKU en Top N:", key="search_top")
df_top = view_df.nsmallest(TOP_N, "score").copy()
if search_top.strip():
    q = search_top.strip().lower()
    df_top = df_top[df_top["sku"].str.lower().str.contains(q, na=False)]
st.dataframe(df_top[cols_general], use_container_width=True, hide_index=True)

st.subheader("🚫 No elegidos (y motivo principal)")
def _why_not_row(row) -> str:
    drivers = {
        "Costo alto": row.get("_c_cost", 0.0),
        "Fill bajo":  row.get("_c_fill", 0.0),
        "DIO alto":   row.get("_c_dio",  0.0),
    }
    main_driver = max(drivers, key=drivers.get)
    extra = ""
    try:
        if float(row["fill_rate_OFF"]) < float(st.session_state["target_fill_rate"]):
            extra = " · Fill por debajo de la meta"
    except Exception:
        pass
    return f"Driver principal: {main_driver}{extra}"

df_not = view_df.iloc[TOP_N:].copy()
if not df_not.empty:
    df_not["why_not"] = df_not.apply(_why_not_row, axis=1)
cols_not = ["ranking_general","sku","why_not","score","cost_mean_OFF","fill_rate_OFF","dio_OFF","type_OFF"]
st.dataframe(df_not[cols_not] if not df_not.empty else df_not, use_container_width=True, hide_index=True)

# ======================= Gráficos (Top 10) =======================
st.subheader("📈 Gráficos (Top 10)")
g10 = df_top.head(10).copy()
col_g1, col_g2 = st.columns(2)

with col_g1:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(g10["sku"], g10["cost_mean_OFF"])
    ax.set_title("Costo por SKU (OFF)")
    ax.set_ylabel("Costo medio")
    ax.tick_params(axis='x', rotation=70)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    st.pyplot(fig)

with col_g2:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(g10["dio_OFF"], g10["fill_rate_OFF"])
    ax.set_title("Trade-off DIO vs Fill (OFF)")
    ax.set_xlabel("DIO (días)")
    ax.set_ylabel("Fill rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)

# ======================== Recomendaciones ========================
st.subheader("🧭 Recomendaciones")

def skus_con_actividad(skus_list: list[str], months: list[int]) -> set[str]:
    if not skus_list:
        return set()
    f_sold = build_monthly_forecast(res, skus_list, year=sim_year,
                                    month_start=min(months), month_end=max(months), use_sold=True)
    f_dmd  = build_monthly_forecast(res, skus_list, year=sim_year,
                                    month_start=min(months), month_end=max(months), use_sold=False)
    f = pd.concat([f_sold, f_dmd], ignore_index=True)
    f["month"] = f["date"].dt.month
    f = f[f["month"].isin(months)]
    act = f.groupby("sku")["forecast_units"].sum()
    return set(act[act > 0].index.tolist())

activos_top = skus_con_actividad(df_top["sku"].tolist(), months_selected)
vista_recs = df_top[df_top["sku"].isin(activos_top)].copy()

def _msg_fill(sku, pol):
    return (f"• 📦 **{sku}**: está por debajo del nivel de servicio esperado. "
            f"Revisa su política {pol} y aumenta stock de seguridad o frecuencia de pedido.")
def _msg_dio(sku):
    return (f"• ⏳ **{sku}**: inventario retenido más de lo deseado (DIO alto). "
            f"Ajusta cantidades por pedido o aumenta la frecuencia; promueve rotación.")
def _msg_ccc(sku):
    return (f"• 💰 **{sku}**: ciclo de caja prolongado. "
            f"Revisa ventas/stock; negocia plazos de pago y mejora rotación.")

def make_recs_v1_explicada(df_view: pd.DataFrame, service_min: float) -> list[str]:
    out = []
    if df_view.empty: return out
    p75_dio = df_top["dio_OFF"].quantile(0.75) if not df_top.empty else 0.0
    med_ccc = base_df["ccc_OFF"].median() if "ccc_OFF" in base_df.columns else 0.0
    for r in df_view.itertuples(index=False):
        sku, fill, dio, pol = r.sku, r.fill_rate_OFF, r.dio_OFF, r.type_OFF
        ccc = getattr(r, "ccc_OFF", med_ccc)
        fill_low = fill < service_min
        dio_high = dio > p75_dio
        ccc_long = ccc > med_ccc
        if not (fill_low or dio_high or ccc_long):
            continue
        parts = []
        if fill_low:
            main = _msg_fill(sku, pol)
            if dio_high: parts.append("inventario inmovilizado (DIO alto)")
            if ccc_long: parts.append("ciclo de caja prolongado (CCC)")
        elif dio_high:
            main = _msg_dio(sku)
            if fill_low: parts.append("nivel de servicio bajo")
            if ccc_long: parts.append("ciclo de caja prolongado (CCC)")
        else:
            main = _msg_ccc(sku)
            if fill_low: parts.append("nivel de servicio bajo")
            if dio_high: parts.append("inventario inmovilizado (DIO alto)")
        if parts: main += " Además, " + " y ".join(parts) + "."
        out.append(main)
    return out

recs = make_recs_v1_explicada(vista_recs, float(service))
if recs:
    with st.expander(f"Ver recomendaciones accionables ({len(recs)})", expanded=True):
        for r in recs:
            st.markdown(r)
else:
    st.info("Sin recomendaciones accionables para el Top en los meses seleccionados.")

# ======================== Proyección =============================
st.subheader("📋 Tabla de proyección (meses seleccionados)")

all_skus = view_df["sku"].tolist()
f_sold = build_monthly_forecast(
    res, all_skus, year=sim_year,
    month_start=min(months_selected), month_end=max(months_selected),
    use_sold=True
)
f_dmd = build_monthly_forecast(
    res, all_skus, year=sim_year,
    month_start=min(months_selected), month_end=max(months_selected),
    use_sold=False
)

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["sku","date","periodo","unidades"])
    d = df.copy()
    d["month"] = d["date"].dt.month
    d = d[d["month"].isin(months_selected)]
    out = (d.groupby(["sku","date"])["forecast_units"]
             .sum().reset_index()
             .rename(columns={"forecast_units":"unidades"}))
    out["periodo"] = out["date"].dt.strftime("%Y-%m")
    return out[["sku","date","periodo","unidades"]]

tab_sold = _prep(f_sold)
tab_dmd  = _prep(f_dmd)

tabla_proj = (
    tab_sold.merge(tab_dmd, on=["sku","date","periodo"], how="outer", suffixes=("_sold","_dmd"))
    .fillna(0.0)
    .rename(columns={"unidades_sold":"ventas_proyectadas", "unidades_dmd":"demanda_proyectada"})
    .sort_values(["sku","date"]).reset_index(drop=True)
)[["sku","periodo","ventas_proyectadas","demanda_proyectada"]]
tabla_proj["ventas_proyectadas"] = tabla_proj["ventas_proyectadas"].round().astype(int)
tabla_proj["demanda_proyectada"] = tabla_proj["demanda_proyectada"].round().astype(int)
tabla_proj = tabla_proj[(tabla_proj["ventas_proyectadas"]>0) | (tabla_proj["demanda_proyectada"]>0)].copy()

q_proj = st.text_input("Buscar SKU en proyección:", key="proj_search").strip()
tabla_proj_view = (tabla_proj[
    tabla_proj["sku"].astype(str).str.contains(q_proj, case=False, na=False)
    | tabla_proj["periodo"].str.contains(q_proj, case=False, na=False)
].copy() if q_proj else tabla_proj.copy())
st.dataframe(tabla_proj_view.reset_index(drop=True), use_container_width=True)

# ======================== Gráfico proyección =====================
st.subheader("📈 Proyección (solo meses seleccionados)")

skus_plot = set(map(str, tabla_proj_view["sku"].dropna().unique()))
if not skus_plot or not months_selected:
    st.info("No hay SKUs o meses para graficar con el filtro actual.")
else:
    def _series_sum(df: pd.DataFrame, months: list[int], skus: set[str]) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype=float)
        d = df.copy()
        d["month"] = d["date"].dt.month
        d = d[d["month"].isin(months)]
        d = d[d["sku"].astype(str).isin(skus)]
        return d.groupby("month")["forecast_units"].sum()

    s_sold = _series_sum(f_sold, months_selected, skus_plot)
    s_dmd  = _series_sum(f_dmd, months_selected, skus_plot)

    months = [m for m in sorted(set(s_sold.index).union(s_dmd.index))
              if float(s_sold.get(m, 0)) + float(s_dmd.get(m, 0)) > 0]

    if not months:
        st.info("Los meses seleccionados no tienen proyección para los SKUs filtrados.")
    else:
        y1 = [int(round(float(s_sold.get(m, 0)))) for m in months]
        y2 = [int(round(float(s_dmd.get(m, 0))))  for m in months]

        fig, ax = plt.subplots(figsize=(11, 4))
        color_ventas, color_demanda = "#1f77b4", "#ff7f0e"
        ax.plot(months, y1, marker="o", label="Ventas proyectadas",  color=color_ventas)
        ax.plot(months, y2, marker="o", label="Demanda proyectada", color=color_demanda)

        max_val = max((y1 + y2) or [0])
        offset  = max(6, int(max_val * 0.015))
        ax.margins(y=0.12)

        for x, v in zip(months, y1):
            if v != 0:
                ax.text(x, v + offset, f"{v}", ha="center", va="bottom",
                        fontsize=9, color=color_ventas, fontweight="bold")
        for x, v in zip(months, y2):
            if v != 0:
                ax.text(x, v - offset, f"{v}", ha="center", va="top",
                        fontsize=9, color=color_demanda, fontweight="bold")

        ax.set_xticks(months)
        ax.set_xticklabels([f"{m:02d}" for m in months])
        ax.set_xlabel("Mes"); ax.set_ylabel("Unidades")
        ax.set_title(f"Curva agregada — {sim_year} (meses {months[0]:02d}-{months[-1]:02d})")
        ax.grid(alpha=0.25); ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

# ======================== Exportar ===============================
if export_btn:
    os.makedirs(out_dir, exist_ok=True)
    paths = save_reports_off(
        out_dir,
        df_top,               # Top base ya filtrado por meses
        res["cands_off"],
        res["rep_df"],
        config=st.session_state.last_cfg,
    )
    st.success(f"✅ Exportado correctamente: {paths['xlsx']}")
    st.caption("🗂️ Archivos guardados en: out/plots/")
