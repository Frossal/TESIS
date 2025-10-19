import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_reports_off(out_dir: str, overview_off_top, cands_off, rep_df, config: dict):
    """Genera Excel + gráficos SOLO OFF (sin transporte) y recomendaciones (Versión 1, solo accionables)."""
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    cfg_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

    # --- limpieza de columna técnica
    for _df in [overview_off_top, cands_off, rep_df]:
        if "name_OFF" in _df.columns:
            _df.drop(columns=["name_OFF"], inplace=True)

    # --- formateo de política para visibilidad
    if "type_OFF" in overview_off_top.columns:
        overview_off_top["type_OFF"] = (
            overview_off_top["type_OFF"].astype(str)
            .str.strip().str.lower()
            .replace({"rq": "(r, Q)", "ss": "(s, S)", "rq_off": "(r, Q)", "ss_off": "(s, S)"})
        )

    # === Excel ===
    xlsx_path = os.path.join(out_dir, f"sim_results_off_{cfg_hash}.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
        overview_off_top.to_excel(w, sheet_name="overview_top_off", index=False)
        cands_off.to_excel(w, sheet_name="candidates_OFF", index=False)
        rep_df.to_excel(w, sheet_name="replicas_raw", index=False)
        pd.DataFrame([{"config_json": json.dumps(config)}]).to_excel(w, sheet_name="config", index=False)

    # === GRÁFICOS ===
    # 1) Costo OFF
    plt.figure(figsize=(12, 5))
    x = np.arange(len(overview_off_top))
    plt.bar(x, overview_off_top["cost_mean_OFF"], width=0.6)
    plt.xticks(x, overview_off_top["sku"], rotation=60, ha="right")
    plt.ylabel("Costo medio")
    plt.title("Costo por SKU (OFF)")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, f"cost_off_{cfg_hash}.png")
    plt.savefig(p1, dpi=160)
    plt.close()

    # 2) Trade-off DIO vs Fill_rate
    plt.figure(figsize=(7, 5))
    plt.scatter(overview_off_top["dio_OFF"], overview_off_top["fill_rate_OFF"])
    plt.xlabel("DIO (días)")
    plt.ylabel("Fill_rate")
    plt.title("Trade-off DIO vs Fill_rate (OFF)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, f"tradeoff_off_{cfg_hash}.png")
    plt.savefig(p2, dpi=160)
    plt.close()

    # 3) CCC por SKU
    plt.figure(figsize=(12, 5))
    plt.bar(overview_off_top["sku"], overview_off_top["ccc_OFF"])
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("CCC (días)")
    plt.title("CCC por SKU (OFF)")
    plt.tight_layout()
    p3 = os.path.join(plots_dir, f"ccc_off_{cfg_hash}.png")
    plt.savefig(p3, dpi=160)
    plt.close()

        # === Recomendaciones (Versión 1, explicada y unificada por SKU) ===
    def _recs_v1_explicada(ov: pd.DataFrame, service_min: float) -> list[str]:
        out = []
        dio_p75 = ov["dio_OFF"].quantile(0.75) if len(ov) else 0.0
        ccc_med = ov["ccc_OFF"].median()       if len(ov) else 0.0

        def _msg_fill(sku, pol):
            return (f"• 📦 {sku}: está por debajo del nivel de servicio esperado. "
                    f"Esto significa que los clientes pueden no encontrar el producto cuando lo solicitan. "
                    f"Revisa su política {pol} y aumenta el stock de seguridad o la frecuencia de pedido; "
                    f"también ayuda coordinar con compras/proveedor para reducir el tiempo de entrega.")

        def _msg_dio(sku):
            return (f"• ⏳ {sku}: tiene inventario acumulado por más tiempo del deseado (DIO alto). "
                    f"Esto inmoviliza capital y ocupa espacio. Ajusta las cantidades por pedido o aumenta la frecuencia de reposición, "
                    f"y promueve la rotación (p. ej., ofertas, sustituciones o mejores exhibiciones).")

        def _msg_ccc(sku):
            return (f"• 💰 {sku}: está demorando demasiado en convertir inventario en efectivo (ciclo de caja largo). "
                    f"Revisa si las ventas están lentas o si hay sobrestock; conversa con tu proveedor para mejorar plazos de pago "
                    f"y acelera el cobro si aplica. Reducir inventario y mejorar la rotación ayuda a que el dinero regrese antes.")

        for r in ov.itertuples(index=False):
            sku, fill, dio, ccc = r.sku, r.fill_rate_OFF, r.dio_OFF, r.ccc_OFF
            pol = getattr(r, "type_OFF", "")

            fill_low = fill < service_min
            dio_high = dio > dio_p75
            ccc_long = ccc > ccc_med

            if not (fill_low or dio_high or ccc_long):
                continue

            parts = []
            if fill_low:
                main = _msg_fill(sku, pol)
                if dio_high: parts.append("inventario inmovilizado (DIO alto)")
                if ccc_long: parts.append("ciclo de caja prolongado (CCC)")
            elif dio_high:
                main = _msg_dio(sku)
                if fill_low: parts.append("nivel de servicio por debajo de la meta")
                if ccc_long: parts.append("ciclo de caja prolongado (CCC)")
            else:
                main = _msg_ccc(sku)
                if fill_low: parts.append("nivel de servicio por debajo de la meta")
                if dio_high: parts.append("inventario inmovilizado (DIO alto)")

            if parts:
                main += " Además, " + " y ".join(parts) + "."

            out.append(main)

        return out

    def _explicacion(ov, service_min):
        porc_ok = 100.0 * (ov["fill_rate_OFF"] >= service_min).mean() if len(ov) else 0.0
        return (
            f"Se muestran {len(ov)} SKUs recomendados OFF. "
            f"{porc_ok:.1f}% cumplen la meta de Fill_rate ({service_min:.2f}). "
            "Las líneas listan únicamente acciones necesarias (cuando las hay)."
        )

    recs = _recs_v1_explicada(overview_off_top, config.get("service_min", 0.95))
    resumen = _explicacion(overview_off_top, config.get("service_min", 0.95))
    rec_path = os.path.join(out_dir, f"recommendations_off_{cfg_hash}.txt")

    with open(rec_path, "w", encoding="utf-8") as f:
        f.write("Resumen:\n" + resumen + "\n\n")
        if recs:
            f.write("Recomendaciones accionables:\n")
            for r in recs:
                f.write(r + "\n")
        else:
            f.write("No se detectaron recomendaciones accionables para los SKUs seleccionados.\n")


    return {"xlsx": xlsx_path, "plots": [p1, p2, p3], "recs": rec_path}
