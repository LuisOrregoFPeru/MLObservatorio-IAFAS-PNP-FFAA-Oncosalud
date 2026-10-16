"""
data_processing.py
-------------------
Carga y filtrado de datos del dashboard de consumo por IAFAS (INEN).

Fuente: data/fact_iafas.parquet (generado con build_data.py a partir de la
extracción cruda INEN.V_CONSUMO_CUENTA_PACIENTE_IAFAS). Una fila por
IAFAS x mes x paciente (seudónimo) x diagnóstico x ítem x sexo x grupo etario x
departamento; el DNI no se almacena.

Definiciones
  - Gasto (S/)      = CANTIDAD x PRECIO de cada línea facturada a la IAFAS. Es el
                      siniestro incurrido por la IAFAS en el INEN (valorizado).
  - Tipo de consumo = MEDICAMENTO (medicina), INSUMO (material médico) o
                      PROCEDIMIENTO (consultas, laboratorio, imágenes, radioterapia,
                      hospitalización, etc.).
  - Paciente activo = paciente (seudónimo) con al menos una línea en el mes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import alto_costo as AC

DATA_DIR = Path(__file__).parent / "data"
FACT_FILE = DATA_DIR / "fact_iafas.parquet"
ITEMS_FILE = DATA_DIR / "items.csv"
DX_FILE = DATA_DIR / "dx_catalog.csv"
PRIMAS_FILE = DATA_DIR / "primas.csv"
AFIL_FILE = DATA_DIR / "afiliados.csv"

TIPO_LIST = ["MEDICAMENTO", "INSUMO", "PROCEDIMIENTO"]
TIPO_LABEL = {"MEDICAMENTO": "Medicamentos", "INSUMO": "Insumos", "PROCEDIMIENTO": "Procedimientos"}
EDAD_ORDER = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+", "Sin dato"]
SD_GROUP = "Sin diagnóstico registrado"
# Ámbito de atención: clasificación directa (procedimientos) e inferida por paciente-mes (medicamentos e insumos)
AMBITO_DIRECTO = ["Consulta externa", "Apoyo al diagnóstico", "Quimioterapia y radioterapia", "Emergencia", "Hospitalización",
                  "Centro quirúrgico", "Procedimientos y terapias ambulatorias", "No clínico"]
AMBITO_INFERIDO = ["Ambulatorio (inferido)", "Emergencia (inferido)", "Hospitalización (inferido)", "Centro quirúrgico (inferido)",
                   "Sin procedimiento asociado en el mes"]
AMBITO_ORDER = AMBITO_DIRECTO + AMBITO_INFERIDO
# Agrupación para ver todo el consumo con las mismas categorías
AMBITO_MACRO = {
    "Consulta externa": "Ambulatorio", "Apoyo al diagnóstico": "Ambulatorio", "Quimioterapia y radioterapia": "Ambulatorio",
    "Procedimientos y terapias ambulatorias": "Ambulatorio", "Ambulatorio (inferido)": "Ambulatorio",
    "Emergencia": "Emergencia", "Emergencia (inferido)": "Emergencia",
    "Hospitalización": "Hospitalización", "Hospitalización (inferido)": "Hospitalización",
    "Centro quirúrgico": "Centro quirúrgico", "Centro quirúrgico (inferido)": "Centro quirúrgico",
    "No clínico": "No clínico", "Sin procedimiento asociado en el mes": "Sin procedimiento asociado",
}
MACRO_ORDER = ["Ambulatorio", "Emergencia", "Hospitalización", "Centro quirúrgico", "Sin procedimiento asociado", "No clínico"]


@st.cache_data(show_spinner="Cargando datos de consumo por IAFAS...")
def load_fact() -> pd.DataFrame:
    fact = pd.read_parquet(FACT_FILE)
    items = pd.read_csv(ITEMS_FILE, dtype={"cod_item": str}).fillna({"subtipo": ""})
    dx = pd.read_csv(DX_FILE)
    items = AC.marcar(items).set_index("item_id")
    df = fact.copy()
    df["descripcion"] = df["item_id"].map(items["descripcion"]).astype("category")
    df["principio_activo"] = df["item_id"].map(items["principio_activo"]).astype("category")
    df["subtipo"] = df["item_id"].map(items["subtipo"]).astype("category")
    df["alto_costo"] = df["item_id"].map(items["alto_costo"]).astype("category")
    df["molecula"] = df["item_id"].map(items["molecula_norm"]).astype("category")
    df["grupo_tarifario"] = df["item_id"].map(items["grupo_tarifario"]).fillna("").astype("category")
    dxi = dx.set_index("cod_dx")
    df["dx_grupo"] = df["cod_dx"].astype(str).map(dxi["grupo"]).astype("category")
    df["dx_etiqueta"] = df["cod_dx"].astype(str).map(dxi["etiqueta"]).astype("category")
    df["tiempo"] = pd.to_datetime(df["tiempo"])
    df["anio"] = df["tiempo"].dt.year.astype("int16")
    df["ambito_macro"] = df["ambito"].astype(str).map(AMBITO_MACRO).astype("category")
    return df


@st.cache_data(show_spinner=False)
def load_primas() -> pd.DataFrame:
    if PRIMAS_FILE.exists():
        p = pd.read_csv(PRIMAS_FILE)
    else:
        p = pd.DataFrame(columns=["iafas", "anio", "prima_anual"])
    p["anio"] = pd.to_numeric(p["anio"], errors="coerce")
    p["prima_anual"] = pd.to_numeric(p["prima_anual"], errors="coerce")
    return p.dropna(subset=["iafas", "anio"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_afiliados() -> pd.DataFrame:
    if AFIL_FILE.exists():
        p = pd.read_csv(AFIL_FILE)
    else:
        p = pd.DataFrame(columns=["iafas", "anio", "afiliados"])
    p["anio"] = pd.to_numeric(p["anio"], errors="coerce")
    p["afiliados"] = pd.to_numeric(p["afiliados"], errors="coerce")
    return p.dropna(subset=["iafas", "anio"]).reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame, *, iafas=None, tipos=None, date_lo=None, date_hi=None,
    dx_grupos=None, sexo=None, edad=None, deptos=None, excluir_sd: bool = False,
    excluir_anios=None, ambitos=None,
) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    if iafas:
        m &= df["iafas"].isin(iafas)
    if tipos:
        m &= df["tipo"].isin(tipos)
    if date_lo is not None:
        m &= df["tiempo"] >= date_lo
    if date_hi is not None:
        m &= df["tiempo"] <= date_hi
    if dx_grupos:
        m &= df["dx_grupo"].isin(dx_grupos)
    if sexo:
        m &= df["sexo"].isin(sexo)
    if edad:
        m &= df["edad_grupo"].isin(edad)
    if deptos:
        m &= df["departamento"].isin(deptos)
    if excluir_sd:
        m &= df["dx_grupo"] != SD_GROUP
    if ambitos:
        m &= df["ambito"].isin(ambitos)
    if excluir_anios:
        m &= ~df["anio"].isin(excluir_anios)
    return df[m]


def month_index(lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(lo, hi, freq="MS")


def monthly_metrics(d: pd.DataFrame, idx: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Serie mensual: gasto, líneas, cantidad, pacientes activos, costo por paciente activo."""
    if d.empty:
        return pd.DataFrame(columns=["gasto", "lineas", "cantidad", "pacientes", "costo_paciente"])
    g = d.groupby("tiempo", observed=True).agg(gasto=("gasto", "sum"), lineas=("lineas", "sum"), cantidad=("cantidad", "sum"))
    pac = d[d["pid"] >= 0].groupby("tiempo", observed=True)["pid"].nunique().rename("pacientes")
    g = g.join(pac).fillna({"pacientes": 0})
    if idx is None:
        idx = month_index(g.index.min(), g.index.max())
    g = g.reindex(idx, fill_value=0.0)
    g.index.name = "tiempo"
    g["costo_paciente"] = np.where(g["pacientes"] > 0, g["gasto"] / g["pacientes"].replace(0, np.nan), np.nan)
    return g


def annual_metrics(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=["gasto", "lineas", "pacientes", "meses", "costo_paciente"])
    g = d.groupby("anio", observed=True).agg(gasto=("gasto", "sum"), lineas=("lineas", "sum"), meses=("tiempo", "nunique"))
    pac = d[d["pid"] >= 0].groupby("anio", observed=True)["pid"].nunique().rename("pacientes")
    g = g.join(pac).fillna({"pacientes": 0})
    g["costo_paciente"] = np.where(g["pacientes"] > 0, g["gasto"] / g["pacientes"].replace(0, np.nan), np.nan)
    g.index = g.index.astype(int)
    return g


def top_table(d: pd.DataFrame, by: str, extra_cols: list[str] | None = None) -> pd.DataFrame:
    """Tabla agregada por `by` con gasto, cantidad, líneas, pacientes, precio medio, % y % acumulado."""
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(by, observed=True).agg(
        gasto=("gasto", "sum"), cantidad=("cantidad", "sum"), lineas=("lineas", "sum"),
        primer_mes=("tiempo", "min"), ultimo_mes=("tiempo", "max"),
    )
    pac = d[d["pid"] >= 0].groupby(by, observed=True)["pid"].nunique().rename("pacientes")
    g = g.join(pac).fillna({"pacientes": 0})
    g["precio_medio"] = np.where(g["cantidad"] > 0, g["gasto"] / g["cantidad"].replace(0, np.nan), np.nan)
    g["gasto_paciente"] = np.where(g["pacientes"] > 0, g["gasto"] / g["pacientes"].replace(0, np.nan), np.nan)
    g = g.sort_values("gasto", ascending=False)
    tot = g["gasto"].sum()
    g["pct_gasto"] = g["gasto"] / tot * 100 if tot else 0.0
    g["pct_acum"] = g["pct_gasto"].cumsum()
    g = g.reset_index()
    return g


def patient_table(d: pd.DataFrame) -> pd.DataFrame:
    """Una fila por paciente (seudónimo): gasto total, meses activos, líneas."""
    dd = d[d["pid"] >= 0]
    if dd.empty:
        return pd.DataFrame(columns=["pid", "gasto", "meses_activos", "lineas"])
    g = dd.groupby("pid").agg(gasto=("gasto", "sum"), meses_activos=("tiempo", "nunique"), lineas=("lineas", "sum"),
                              primer_mes=("tiempo", "min"), ultimo_mes=("tiempo", "max"))
    return g.sort_values("gasto", ascending=False).reset_index()
