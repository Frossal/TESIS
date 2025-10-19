import os
import argparse
import numpy as np
import random
import pandas as pd

from inventory_sim.config import SimConfig
from inventory_sim.dataio import load_all
from inventory_sim.policies import generate_candidates
from inventory_sim.simulate import simulate
from inventory_sim.rs import rank_and_select
from inventory_sim.reports import save_reports_off


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Simulador de inventarios (r,Q) vs (s,S) — OFF (sin transporte), Top-N y recomendaciones"
    )
    ap.add_argument("--data-dir", required=True, help="Carpeta con sales.csv, lead_times.csv, cogs.csv")
    ap.add_argument("--out-dir", default="out", help="Carpeta de salida")
    ap.add_argument("--service", type=float, default=0.95, help="Meta de fill rate")
    ap.add_argument("--horizon", type=int, default=365, help="Horizonte de simulación (días)")
    ap.add_argument("--init-reps", type=int, default=12, help="Réplicas iniciales por candidato")
    ap.add_argument("--max-reps", type=int, default=60, help="Máximo total de réplicas por candidato")
    ap.add_argument("--ci-rel", type=float, default=0.03, help="Semi-ancho relativo del IC95 objetivo (costo)")
    ap.add_argument("--topk", type=int, default=50, help="Cuántos SKUs mostrar en tabla y gráficos")
    ap.add_argument("--max-skus", type=int, default=0, help="Limita cuántos SKUs procesar (0=todos)")
    ap.add_argument("--log-every", type=int, default=10, help="Log de progreso cada N SKUs")
    ap.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    ap.add_argument("--flush", action="store_true", help="Forzar impresión sin buffer (consola)")
    return ap


def pick_topk_off(winners_off: pd.DataFrame, k: int, service_min: float) -> pd.DataFrame:
    """Prioriza los que cumplen servicio y menor costo; completa con los siguientes."""
    ok = winners_off[winners_off["fill_rate"] >= service_min].sort_values("cost_mean", ascending=True)
    bad = winners_off[winners_off["fill_rate"] < service_min].sort_values("cost_mean", ascending=True)
    top = pd.concat([ok, bad], ignore_index=True).head(k)
    top = top.rename(
        columns={
            "type": "type_OFF",
            "name": "name_OFF",
            "cost_mean": "cost_mean_OFF",
            "fill_rate": "fill_rate_OFF",
            "dio": "dio_OFF",
            "ccc": "ccc_OFF",
        }
    )
    return top[["sku", "type_OFF", "name_OFF", "cost_mean_OFF", "fill_rate_OFF", "dio_OFF", "ccc_OFF"]]


def main():
    args = build_argparser().parse_args()

    cfg = SimConfig(
        horizon_days=args.horizon,
        service_min=args.service,
        init_reps=args.init_reps,
        max_reps=args.max_reps,
        ci_rel_target=args.ci_rel,
        seed_numpy=args.seed,
        seed_random=args.seed,
    )
    np.random.seed(cfg.seed_numpy)
    random.seed(cfg.seed_random)

    # Lee datos
    sales, costs = load_all(args.data_dir)
    skus = sorted(costs["sku"].unique().tolist())
    if args.max_skus and args.max_skus > 0:
        skus = skus[: args.max_skus]

    print(f"Iniciando simulación (OFF) para {len(skus)} SKUs...")
    if args.flush:
        import sys

        sys.stdout.flush()

    cands_off_list = []
    rep_all = []
    winners_off = []

    for idx, sku in enumerate(skus, 1):
        if (idx == 1) or (idx % max(1, args.log_every) == 0):
            print(f"[{idx}/{len(skus)}] SKU: {sku}")
            if args.flush:
                import sys

                sys.stdout.flush()

        params = costs[costs["sku"] == sku].iloc[0].to_dict()
        cands = generate_candidates(sales, sku, params)
        cand_list = [("rQ", p) for p in cands["rQ"]] + [("sS", p) for p in cands["sS"]]

        # escenario OFF (sin transporte)
        sim_off = lambda s, pr, pol, pt: simulate(
            sales, s, pr, pol, pt, cfg.horizon_days, transport_on=False
        )

        perf_off, rep_off = rank_and_select(lambda s, pr, pol, pt: sim_off(s, pr, pol, pt), sku, params, cand_list, cfg)

        # tabla de candidatos con IC95
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

        # ganador OFF
        pf = perf_off.copy()
        pf["score"] = pf["mean"] + np.where(pf["fill"] >= cfg.service_min, 0, (cfg.service_min - pf["fill"]) * 1e6)
        row = pf.sort_values("score").iloc[0]
        winners_off.append(
            {
                "sku": sku,
                "type": row["type"],
                "name": row["name"],
                "cost_mean": row["mean"],
                "fill_rate": row["fill"],
                "dio": row["dio"],
                "ccc": row["ccc"],
                "n": row["n"],
                "sd": row.get("sd", 0.0),
            }
        )

    cands_off = pd.concat(cands_off_list, ignore_index=True)
    rep_df = pd.concat(rep_all, ignore_index=True)
    off = pd.DataFrame(winners_off)

    # Top-N OFF
    top_over_off = pick_topk_off(off, args.topk, cfg.service_min)

    # Guardar reportes (Excel + PNGs + recomendaciones .txt)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = save_reports_off(args.out_dir, top_over_off, cands_off, rep_df, config=cfg.__dict__)

    print("OK.")
    print("XLSX:", paths["xlsx"])
    for p in paths["plots"]:
        print("PLOT:", p)
    print("TXT :", paths["recs"])


if __name__ == "__main__":
    main()
