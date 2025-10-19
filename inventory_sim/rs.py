import numpy as np
import pandas as pd


def mean_ci(values, alpha=0.05):
    arr = np.array(values, dtype=float)
    m = float(np.mean(arr))
    n = len(arr)
    sd = float(np.std(arr, ddof=1) if n > 1 else 0.0)
    semi = 1.96 * sd / max(np.sqrt(n), 1e-9) if n > 1 else 0.0
    return m, m - semi, m + semi, semi


def rank_and_select(sim_fn, sku: str, params: dict, candidates, cfg):
    rows = []
    # réplicas iniciales
    for ctype, pol in candidates:
        for _ in range(cfg.init_reps):
            r = sim_fn(sku, params, pol, ctype)
            rows.append((sku, ctype, pol.name, r.cost, r.fill_rate, r.dio, r.ccc))

    rep_df = pd.DataFrame(rows, columns=["sku", "type", "name", "cost", "fill_rate", "dio", "ccc"])
    perf = rep_df.groupby(["sku", "type", "name"]).agg(
        n=("cost", "count"),
        mean=("cost", "mean"),
        sd=("cost", "std"),
        fill=("fill_rate", "mean"),
        dio=("dio", "mean"),
        ccc=("ccc", "mean"),
    ).reset_index()
    perf["semi"] = 1.96 * perf["sd"].fillna(0) / np.sqrt(perf["n"].clip(lower=1))

    # bucle adaptativo simple: top-3 hasta cumplir criterio
    while True:
        perf["score"] = perf["mean"] + np.where(perf["fill"] >= cfg.service_min, 0, (cfg.service_min - perf["fill"]) * 1e6)
        best = perf.sort_values("score").iloc[0]
        rel_ok = best["semi"] <= cfg.ci_rel_target * max(best["mean"], 1e-6)
        if rel_ok or (perf["n"].sum() >= cfg.max_reps * len(candidates)):
            break
        for _, row in perf.sort_values("score").head(3).iterrows():
            pol = next(p for (t, p) in candidates if (t == row["type"] and p.name == row["name"]))
            r = sim_fn(sku, params, pol, row["type"])
            rep_df.loc[len(rep_df)] = [sku, row["type"], row["name"], r.cost, r.fill_rate, r.dio, r.ccc]
        perf = rep_df.groupby(["sku", "type", "name"]).agg(
            n=("cost", "count"),
            mean=("cost", "mean"),
            sd=("cost", "std"),
            fill=("fill_rate", "mean"),
            dio=("dio", "mean"),
            ccc=("ccc", "mean"),
        ).reset_index()
        perf["semi"] = 1.96 * perf["sd"].fillna(0) / np.sqrt(perf["n"].clip(lower=1))

    return perf, rep_df
