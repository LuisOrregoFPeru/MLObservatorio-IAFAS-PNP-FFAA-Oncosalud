"""
build_data.py
-------------
Convierte la extracción cruda de consumo por IAFAS (Excel con la hoja de trabajo
INEN.V_CONSUMO_CUENTA_PACIENTE_IAFAS) en tres archivos compactos y ANONIMIZADOS:

  data/fact_iafas.parquet   una fila por IAFAS x mes x paciente(seudónimo) x dx x ítem
  data/items.csv            catálogo de ítems (descripción canónica, principio activo, tipo)
  data/dx_catalog.csv       catálogo de diagnósticos CIE-10 con grupo de sitio

Uso:
    python build_data.py RUTA/DATA_IAFAS.xlsx

El DNI nunca se guarda: se reemplaza por un identificador entero derivado de un
hash con sal (PID_SALT), suficiente para contar pacientes únicos.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "data"
PID_SALT = os.environ.get("PID_SALT", "iafas-dashboard-v1")

TIPO_MAP = {"MEDICINA": "MEDICAMENTO", "MATERIAL MEDICO": "INSUMO", "PROCEDIMIENTO": "PROCEDIMIENTO"}
IAFAS_SHORT = {
    "SALUDPOL": "SALUDPOL",
    "ONCOSALUD S.A.C.": "ONCOSALUD",
    "IAFAS DEL EJERCITO DEL PERU (FOSPEME)": "FOSPEME (Ejército)",
    "IAFAS DE LA MARINA DE GUERRA DEL PERU": "FOSMAR (Marina de Guerra)",
    "IAFAS DE LA FUERZA AEREA DEL PERU (FOSFAP)": "FOSFAP (Fuerza Aérea)",
}

# Grupos de sitio según bloques CIE-10 (3 primeros caracteres)
def dx_group(cod: str) -> str:
    c = (cod or "").strip().upper()
    if c in ("", "SD"):
        return "Sin diagnóstico registrado"
    if not re.match(r"^[A-Z]\d{2}", c):
        return "Código no CIE-10 / otros"
    L, n = c[0], int(c[1:3])
    if L == "C":
        if n <= 14: return "Cavidad oral, labio y faringe"
        if n == 15: return "Esófago"
        if n == 16: return "Estómago"
        if n == 17: return "Intestino delgado"
        if n == 18: return "Colon"
        if 19 <= n <= 21: return "Recto y ano"
        if n == 22: return "Hígado y vías biliares intrahepáticas"
        if n in (23, 24): return "Vesícula y vías biliares"
        if n == 25: return "Páncreas"
        if n == 26: return "Otros digestivos"
        if n == 34: return "Pulmón y bronquios"
        if 30 <= n <= 39: return "Otros respiratorios y torácicos"
        if n in (40, 41): return "Hueso y cartílago"
        if n == 43: return "Melanoma"
        if n == 44: return "Otros tumores de piel"
        if 45 <= n <= 49: return "Tejidos blandos y mesotelio"
        if n == 50: return "Mama"
        if n == 53: return "Cuello uterino"
        if n in (54, 55): return "Cuerpo del útero"
        if n == 56: return "Ovario"
        if 51 <= n <= 58: return "Otros genitales femeninos"
        if n == 61: return "Próstata"
        if n == 62: return "Testículo"
        if n in (60, 63): return "Otros genitales masculinos"
        if n in (64, 65): return "Riñón y pelvis renal"
        if n == 67: return "Vejiga"
        if 66 <= n <= 68: return "Otros urinarios"
        if 69 <= n <= 72: return "Ojo, encéfalo y SNC"
        if n == 73: return "Tiroides"
        if 74 <= n <= 75: return "Otras glándulas endocrinas"
        if 76 <= n <= 80 or n == 97: return "Mal definidos, secundarios y sitio primario desconocido"
        if n == 81: return "Linfoma de Hodgkin"
        if 82 <= n <= 86: return "Linfomas (no Hodgkin y otros)"
        if 88 <= n <= 90: return "Mieloma y neoplasias inmunoproliferativas"
        if n == 91: return "Leucemia linfoide"
        if n == 92: return "Leucemia mieloide"
        if 93 <= n <= 95: return "Otras leucemias"
        if n == 96: return "Otros linfoides y hematopoyéticos"
        return "Otros tumores malignos"
    if L == "D":
        if n <= 9: return "Carcinoma in situ"
        if n <= 36: return "Tumores benignos"
        if n <= 48: return "Tumores de comportamiento incierto"
        if 50 <= n <= 89: return "Enfermedades de la sangre (no neoplásicas)"
        return "Otras enfermedades (CIE-10 D)"
    if L == "Z":
        return "Factores que influyen en la salud (Z)"
    return "Otras enfermedades no neoplásicas"



# Ámbito de atención ---------------------------------------------------------
TARIFARIO = OUT / "tarifario_inen.csv"
AMB_AMB = "Procedimientos y terapias ambulatorias"


def ambito_grupo(g: str) -> str | None:
    """Ámbito a partir del grupo del tarifario INEN (filas de agrupamiento del tarifario)."""
    u = str(g).upper()
    if u.startswith("CONSULTA EXTERNA") or u in ("UNIDAD FUNCIONAL DE SOPORTE A TELESALUD", "ESTRATEGIA SANITARIA", "UNIDAD FUNCIONAL DE ODONTOLOGÍA",
                                                   "UNIDAD FUNCIONAL DE OFTALMOLOGÍA") or u.startswith("DEPARTAMENTO DE PREVENCION"):
        return "Consulta externa"
    if u.startswith("ESPECIALIDADES MÉDICAS"):
        return "Apoyo al diagnóstico" if any(k in u for k in ("CARDIOLOGÍA", "NEUMOLOGÍA", "NEUROLOGÍA")) else AMB_AMB
    if u in ("HOSPITALIZACIÓN", "PROCEDIMIENTOS DE HOSPITALIZACIÓN"):
        return "Hospitalización"
    if u == "MEDICINA CRÍTICA":
        return "Emergencia"
    if u in ("QUIMIOTERAPIA", "RADIOTERAPIA"):
        return "Quimioterapia y radioterapia"
    if u.startswith("ANESTESIA") or u.startswith("CIRUGIA") or u == "UNIDAD FUNCIONAL DE PROCEDIMIENTOS VASCULARES":
        return "Centro quirúrgico"
    if u in ("ENFERMERIA", "TRANSFUSIONES", "PROCEDIMIENTOS ESPECIALES", "PROCEDIMIENTOS MÉDICOS", "LABORATORIO DENTAL"):
        return AMB_AMB
    if u.startswith("LABORATORIO") or u.startswith("EQUIPO FUNCIONAL") or u.startswith("UNIDAD FUNCIONAL") or u == "MEDICINA NUCLEAR" \
            or u.startswith("AREA DE TRABAJO DE TOMA"):
        return "Apoyo al diagnóstico"
    return None


def ambito_prefijo(cod: str) -> str:
    """Respaldo para códigos que no figuran en el tarifario: prefijo del código INEN."""
    p = str(cod).strip()[:2]
    if p in ("03", "04", "05", "06"):
        return "Centro quirúrgico"
    if p in ("12", "13", "40"):
        return "Consulta externa"
    if p in ("25", "24", "29", "31", "21", "18", "17", "19"):
        return "Apoyo al diagnóstico"
    if p in ("09", "30"):
        return "Quimioterapia y radioterapia"
    if p == "02":
        return "Hospitalización"
    if p == "41":
        return "No clínico"
    return AMB_AMB


def ambito_proc(cod: str, grupo: str | None) -> str:
    if grupo:
        a = ambito_grupo(grupo)
        if a:
            return a
    return ambito_prefijo(cod)


def ambito_inferido(df: pd.DataFrame) -> pd.Series:
    """Para medicamentos e insumos (sin campo de ámbito): ámbito del paciente en el mes, según sus procedimientos.
    Prioridad: Hospitalización > Emergencia > Centro quirúrgico > Ambulatorio. Es una inferencia a nivel paciente-mes."""
    proc = df[df.tipo == "PROCEDIMIENTO"]
    k = ["iafas", "tiempo", "pid"]
    pm = proc[proc.pid >= 0].groupby(k)["ambito"].agg(set)
    def pick(sv):
        if "Hospitalización" in sv: return "Hospitalización (inferido)"
        if "Emergencia" in sv: return "Emergencia (inferido)"
        if "Centro quirúrgico" in sv: return "Centro quirúrgico (inferido)"
        return "Ambulatorio (inferido)"
    lab = pm.map(pick)
    m = df.set_index(k).index
    out = pd.Series(lab.reindex(m).values, index=df.index)
    return out.fillna("Sin procedimiento asociado en el mes")


def principio_activo(desc: str) -> str:
    d = re.sub(r"\s+", " ", str(desc)).strip()
    toks = d.split(" ")
    out = []
    for t in toks:
        if re.match(r"^[\d(\[]", t):
            break
        out.append(t)
    s = " ".join(out).strip(" +,-")
    return (s or d).title() if s.isupper() else (s or d)


def mode(s: pd.Series):
    m = s.dropna().mode()
    return m.iloc[0] if len(m) else None


def edad_grupo(e: pd.Series) -> pd.Series:
    e = pd.to_numeric(e, errors="coerce")
    e = e.where((e >= 0) & (e <= 110))
    bins = [-1, 14, 29, 44, 59, 74, 200]
    labels = ["0-14", "15-29", "30-44", "45-59", "60-74", "75+"]
    out = pd.cut(e, bins=bins, labels=labels).astype("object")
    return out.fillna("Sin dato")


def main(path: str) -> None:
    raw = pd.read_excel(path, sheet_name=0, dtype=str)
    print("filas crudas:", len(raw))
    df = pd.DataFrame()
    df["anio"] = raw["ANIO"].astype(int)
    df["mes"] = raw["MES"].astype(int)
    df["tiempo"] = pd.to_datetime(dict(year=df.anio, month=df.mes, day=1))
    df["iafas"] = raw["IAFAS"].map(IAFAS_SHORT).fillna(raw["IAFAS"])
    df["tipo"] = raw["TIPO_REGISTRO"].map(TIPO_MAP)
    cant = pd.to_numeric(raw["CANTIDAD"], errors="coerce").fillna(0.0)
    prec = pd.to_numeric(raw["PRECIO"], errors="coerce").fillna(0.0)
    df["cantidad"] = cant
    df["gasto"] = cant * prec
    dni = raw["DNI"].fillna("-").str.strip()
    h = dni.map(lambda x: int(hashlib.sha256((PID_SALT + x).encode()).hexdigest()[:12], 16))
    df["pid"] = np.where(dni.isin(["-", ""]), -1, h)
    df["cod_dx"] = raw["COD_DX"].fillna("SD").str.strip().str.upper()
    df["cod_item"] = raw["COD_ITEM"].astype(str).str.strip()
    df["sexo"] = raw["SEXO"].fillna("Sin dato").str.strip().replace({"M": "Masculino", "F": "Femenino"})
    df["edad_grupo"] = edad_grupo(raw["EDAD"])
    df["departamento"] = raw["DES_DEPARTAMENTO"].fillna("Sin dato").str.replace(r"\s*\(.*\)", "", regex=True).str.strip().str.title()

    tar = pd.read_csv(TARIFARIO, dtype={"cod": str}).set_index("cod")["grupo_tarifario"] if TARIFARIO.exists() else pd.Series(dtype=str)
    cods = df["cod_item"].unique()
    amb_map = {c: ambito_proc(c, tar.get(c)) for c in cods}
    df["ambito"] = np.where(df.tipo == "PROCEDIMIENTO", df.cod_item.map(amb_map), None)
    inf = ambito_inferido(df)
    df["ambito"] = df["ambito"].where(df.tipo == "PROCEDIMIENTO", inf)

    # catálogos ---------------------------------------------------------
    items = (
        pd.DataFrame({"cod_item": df.cod_item, "tipo": df.tipo, "descripcion": raw["DESCRIPCION"].str.strip(),
                      "subtipo": raw["TIPO_MEDICINA"], "cod_digemid": raw["COD_DIGEMID"]})
        .groupby("cod_item")
        .agg(tipo=("tipo", mode), descripcion=("descripcion", mode), subtipo=("subtipo", mode), cod_digemid=("cod_digemid", mode))
        .reset_index()
    )
    items["descripcion"] = items["descripcion"].str.replace(r"\s+", " ", regex=True)
    items["subtipo"] = items["subtipo"].fillna("").str.title().replace({"Medicinas Genericas": "Genéricas", "Medicinas Oncologicas": "Oncológicas",
                                                                       "Medicinas-Formula Magistral": "Fórmula magistral", "Material Medico": ""})
    items["principio_activo"] = np.where(items.tipo == "MEDICAMENTO", items.descripcion.map(principio_activo), items.descripcion)
    items["grupo_tarifario"] = np.where(items["tipo"] == "PROCEDIMIENTO", items["cod_item"].map(lambda c: tar.get(c, "Sin grupo en el tarifario")), "")
    items["item_id"] = np.arange(len(items), dtype="int32")
    idmap = dict(zip(items.cod_item, items.item_id))
    df["item_id"] = df.cod_item.map(idmap).astype("int32")

    dxd = (
        pd.DataFrame({"cod_dx": df.cod_dx, "des": raw["DES_DX"].str.strip()})
        .groupby("cod_dx").agg(descripcion=("des", mode)).reset_index()
    )
    dxd["descripcion"] = dxd["descripcion"].fillna("Sin descripción")
    dxd["descripcion"] = dxd["descripcion"].str.replace(r"^TM ", "Tumor maligno de ", regex=True).str.capitalize()
    dxd.loc[dxd.cod_dx == "SD", "descripcion"] = "Sin diagnóstico registrado"
    dxd["grupo"] = dxd.cod_dx.map(dx_group)
    dxd["etiqueta"] = dxd.cod_dx + " · " + dxd.descripcion

    # tabla de hechos ----------------------------------------------------
    key = ["iafas", "tiempo", "pid", "cod_dx", "item_id", "sexo", "edad_grupo", "departamento", "tipo", "ambito"]
    fact = df.groupby(key, observed=True, as_index=False).agg(gasto=("gasto", "sum"), cantidad=("cantidad", "sum"), lineas=("gasto", "size"))
    for c in ["iafas", "cod_dx", "sexo", "edad_grupo", "departamento", "tipo", "ambito"]:
        fact[c] = fact[c].astype("category")
    fact["pid"] = fact["pid"].astype("int64")
    fact["gasto"] = fact["gasto"].astype("float32")
    fact["cantidad"] = fact["cantidad"].astype("float32")
    fact["lineas"] = fact["lineas"].astype("int32")

    OUT.mkdir(exist_ok=True)
    fact.to_parquet(OUT / "fact_iafas.parquet", index=False, compression="zstd")
    items.drop(columns=["cod_digemid"]).to_csv(OUT / "items.csv", index=False)
    dxd.to_csv(OUT / "dx_catalog.csv", index=False)
    print("fact:", fact.shape, "| items:", len(items), "| dx:", len(dxd))
    print("gasto total S/ {:,.0f}".format(df.gasto.sum()))
    print(fact.groupby("iafas", observed=True).gasto.sum().round(0))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Uso: python build_data.py RUTA/DATA_IAFAS.xlsx")
    main(sys.argv[1])
