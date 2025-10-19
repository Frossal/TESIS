# procesar_data.py
import os
import re
import pandas as pd
from datetime import datetime

EXCEL_FILE = "Data Tesisxlsx.xlsx"  # cámbialo si tu archivo se llama distinto
OUTPUT_DIR = os.path.join("..", "data")  # genera CSVs en ../data desde /procesamiento

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Leyendo archivo {EXCEL_FILE}...")

# ---------- 1) Cargar Excel y detectar fila de encabezados ----------
raw = pd.read_excel(EXCEL_FILE, header=None)

# Busca en las primeras 10 filas una celda que diga "PRODUCTO"
header_row = None
for i in range(min(10, len(raw))):
    if raw.iloc[i].astype(str).str.strip().str.upper().eq("PRODUCTO").any():
        header_row = i
        break

if header_row is None:
    raise ValueError("No se encontró la fila de encabezados (no aparece 'PRODUCTO' en las primeras filas).")

df = pd.read_excel(EXCEL_FILE, header=header_row)
df.columns = [str(c).strip() for c in df.columns]  # limpia espacios

# ---------- 2) Detectar columnas clave ----------
# Columna de producto (acepta varias variantes)
prod_cols_try = ["PRODUCTO", "Producto", "producto", "sku", "SKU"]
sku_col = next((c for c in prod_cols_try if c in df.columns), None)
if sku_col is None:
    raise ValueError("No se encontró la columna de producto (PRODUCTO/sku).")

# Columnas de meses tipo "2023/Ene", "2024/Feb", etc.
def is_month_col(col: str) -> bool:
    col = str(col).strip()
    m = re.match(r"^\s*(\d{4})/([A-Za-zÁÉÍÓÚáéíóú]{3})\s*$", col)
    return m is not None

month_cols = [c for c in df.columns if is_month_col(c)]
if not month_cols:
    raise ValueError("No se encontraron columnas con formato 'YYYY/Mon' (ej. 2023/Ene, 2024/Feb).")

# Columnas opcionales para precio unitario
cant_col_try  = ["CANT. TOTAL", "CANTIDAD TOTAL", "CANT_TOTAL", "CANT TOTAL"]
total_col_try = ["TOTAL (S/)", "TOTAL S/", "TOTAL_S", "TOTAL"]

cant_col  = next((c for c in cant_col_try  if c in df.columns), None)
total_col = next((c for c in total_col_try if c in df.columns), None)

# ---------- 3) Calcular precio unitario (si hay totales) ----------
if cant_col and total_col:
    df["precio_unitario"] = pd.to_numeric(df[total_col], errors="coerce") / \
                            pd.to_numeric(df[cant_col],  errors="coerce").replace(0, pd.NA)
else:
    df["precio_unitario"] = pd.NA

# ---------- 4) Transformar a formato largo ----------
id_vars = [sku_col, "precio_unitario"]
melt = df.melt(id_vars=id_vars, value_vars=month_cols,
               var_name="col_mes", value_name="qty")

# ---------- 5) Parsear fecha YYYY/Mon (meses en español) ----------
mes_map = {
    "ENE":1, "FEB":2, "MAR":3, "ABR":4, "MAY":5, "JUN":6,
    "JUL":7, "AGO":8, "SET":9, "SEP":9, "OCT":10, "NOV":11, "DIC":12
}

def parse_yyyy_mon(s: str):
    s = str(s).strip()
    if "/" not in s:
        return pd.NaT
    y, mon = s.split("/", 1)
    mon = mon.strip().upper()[:3]
    m = mes_map.get(mon)
    try:
        y = int(y)
    except:
        return pd.NaT
    if m is None:
        return pd.NaT
    return datetime(y, m, 1)

melt["date"] = melt["col_mes"].apply(parse_yyyy_mon)
melt.drop(columns=["col_mes"], inplace=True)
melt = melt.dropna(subset=["date"])

# Asegura tipos numéricos
melt["qty"] = pd.to_numeric(melt["qty"], errors="coerce").fillna(0)

# Renombrar a esquema del simulador
melt.rename(columns={sku_col: "sku"}, inplace=True)

# ---------- 6) Exportar CSVs ----------
sales = melt[["date", "sku", "qty"]].copy()
sales.sort_values(["sku", "date"], inplace=True)
sales.to_csv(os.path.join(OUTPUT_DIR, "sales.csv"), index=False)
print(f"✅ Generado: {os.path.join(OUTPUT_DIR, 'sales.csv')}")

if melt["precio_unitario"].notna().any():
    prices = melt[["date", "sku", "precio_unitario"]].rename(columns={"precio_unitario": "unit_price"})
    prices.sort_values(["sku", "date"], inplace=True)
    prices.to_csv(os.path.join(OUTPUT_DIR, "prices.csv"), index=False)
    print(f"✅ Generado: {os.path.join(OUTPUT_DIR, 'prices.csv')}")
else:
    print("ℹ️ No se encontraron columnas de totales; se generó solo sales.csv (sin prices.csv).")

print("Listo.")
