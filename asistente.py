"""
asistente.py
------------
Asistente de consultas en lenguaje natural sobre los datos del observatorio (barra lateral).

Dos modos:
  - Modo IA (LLM): si hay una clave de Anthropic (st.secrets["ANTHROPIC_API_KEY"], variable de entorno
    o ingresada en la barra lateral), Claude interpreta la pregunta y llama a la herramienta
    `consultar_datos`, que ejecuta agregaciones seguras con pandas (no se ejecuta código generado).
    Al modelo solo le llegan tablas agregadas; nunca filas individuales ni identificadores.
  - Modo básico (sin clave): un intérprete de reglas en español reconoce IAFAS, años, tipo de consumo,
    métrica, agrupación y ranking, y responde con la misma herramienta.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from data_processing import SD_GROUP, TIPO_LABEL
from ui import STRETCH, mes, num

DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_ROWS_LLM = 40          # filas máximas que se devuelven al modelo por consulta
MAX_TOOL_ROUNDS = 6

# Dimensiones permitidas: nombre expuesto -> columna del DataFrame
DIMS = {
    "iafas": "iafas", "anio": "anio", "mes": "tiempo", "tipo": "tipo", "dx_grupo": "dx_grupo",
    "diagnostico": "dx_etiqueta", "item": "descripcion", "principio_activo": "principio_activo",
    "ambito": "ambito", "ambito_macro": "ambito_macro", "grupo_tarifario": "grupo_tarifario",
    "sexo": "sexo", "edad_grupo": "edad_grupo", "departamento": "departamento", "alto_costo": "alto_costo",
}
DIM_LABEL = {
    "iafas": "IAFAS", "anio": "Año", "mes": "Mes", "tipo": "Tipo de consumo", "dx_grupo": "Grupo de diagnóstico",
    "diagnostico": "Diagnóstico (CIE-10)", "item": "Ítem", "principio_activo": "Principio activo", "ambito": "Ámbito",
    "ambito_macro": "Ámbito (agrupado)", "grupo_tarifario": "Grupo del tarifario", "sexo": "Sexo",
    "edad_grupo": "Grupo etario", "departamento": "Departamento", "alto_costo": "Alto costo",
}
METRICS = {
    "gasto": "Gasto (S/)", "cantidad": "Cantidad", "lineas": "Líneas", "pacientes": "Pacientes únicos",
    "gasto_por_paciente": "Gasto por paciente (S/)", "precio_medio": "Precio medio (S/)",
}

TOOL = {
    "name": "consultar_datos",
    "description": (
        "Agrega el consumo facturado por el INEN a las IAFAS. Devuelve una tabla agrupada por las dimensiones pedidas, "
        "ordenada por la métrica, con el total general y el porcentaje de cada fila. Úsala tantas veces como haga falta "
        "(por ejemplo, una consulta para el total y otra para el ranking). Los filtros de texto aceptan coincidencias parciales "
        "sin distinguir mayúsculas ni tildes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metrica": {"type": "string", "enum": list(METRICS), "description": "Métrica a calcular. Por defecto gasto."},
            "agrupar_por": {"type": "array", "items": {"type": "string", "enum": list(DIMS)}, "maxItems": 3,
                            "description": "Dimensiones de agrupación (0 a 3). Vacío = solo total."},
            "top": {"type": "integer", "minimum": 1, "maximum": 100, "description": "Número de filas a devolver (por defecto 15)."},
            "orden": {"type": "string", "enum": ["desc", "asc", "natural"],
                      "description": "desc/asc por la métrica; natural = orden de la dimensión (útil para series de tiempo)."},
            "filtros": {
                "type": "object",
                "properties": {
                    "iafas": {"type": "array", "items": {"type": "string"}},
                    "tipo": {"type": "array", "items": {"type": "string", "enum": list(TIPO_LABEL)}},
                    "anio_desde": {"type": "integer"}, "anio_hasta": {"type": "integer"},
                    "mes_desde": {"type": "string", "description": "AAAA-MM"}, "mes_hasta": {"type": "string", "description": "AAAA-MM"},
                    "dx_grupo": {"type": "array", "items": {"type": "string"}},
                    "diagnostico_texto": {"type": "string", "description": "Texto o código CIE-10 contenido en la etiqueta del diagnóstico."},
                    "item_texto": {"type": "string", "description": "Texto contenido en la descripción del ítem o en el principio activo."},
                    "sexo": {"type": "array", "items": {"type": "string"}},
                    "edad_grupo": {"type": "array", "items": {"type": "string"}},
                    "departamento": {"type": "array", "items": {"type": "string"}},
                    "ambito_macro": {"type": "array", "items": {"type": "string"}},
                    "grupo_tarifario": {"type": "array", "items": {"type": "string"}},
                    "solo_alto_costo": {"type": "boolean", "description": "Solo medicamentos de la lista de alto costo no PNUME."},
                    "excluir_sin_diagnostico": {"type": "boolean"},
                },
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Motor de consulta (seguro: solo agregaciones predefinidas)
# ---------------------------------------------------------------------------
def _norm(s) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s).strip()


def _match_values(col: pd.Series, wanted) -> list:
    """Empareja valores pedidos con categorías reales (exacto o parcial, sin tildes)."""
    cats = col.astype(str).unique().tolist() if not hasattr(col, "cat") else [str(c) for c in col.cat.categories]
    out = []
    for w in wanted or []:
        nw = _norm(w)
        exact = [c for c in cats if _norm(c) == nw]
        out += exact or [c for c in cats if nw and nw in _norm(c)]
    return sorted(set(out))


def _contains(col: pd.Series, txt: str) -> pd.Series:
    cats = pd.Series(col.astype(str).unique())
    hit = set(cats[cats.map(_norm).str.contains(re.escape(_norm(txt)), regex=True)])
    return col.astype(str).isin(hit)


def consultar(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Ejecuta la consulta. Devuelve (tabla, resumen) con métricas ya calculadas."""
    f = spec.get("filtros") or {}
    m = pd.Series(True, index=df.index)
    aplicados = []
    for key, col in (("iafas", "iafas"), ("tipo", "tipo"), ("dx_grupo", "dx_grupo"), ("sexo", "sexo"),
                     ("edad_grupo", "edad_grupo"), ("departamento", "departamento"), ("ambito_macro", "ambito_macro"),
                     ("grupo_tarifario", "grupo_tarifario")):
        if f.get(key):
            vals = _match_values(df[col], f[key])
            m &= df[col].astype(str).isin(vals)
            aplicados.append(f"{DIM_LABEL.get(key, key)}: {', '.join(vals) if vals else '(sin coincidencias)'}")
    if f.get("anio_desde"):
        m &= df["anio"] >= int(f["anio_desde"]); aplicados.append(f"desde {f['anio_desde']}")
    if f.get("anio_hasta"):
        m &= df["anio"] <= int(f["anio_hasta"]); aplicados.append(f"hasta {f['anio_hasta']}")
    for key, op in (("mes_desde", ">="), ("mes_hasta", "<=")):
        if f.get(key):
            try:
                t = pd.Timestamp(str(f[key])[:7] + "-01")
                m &= (df["tiempo"] >= t) if op == ">=" else (df["tiempo"] <= t)
                aplicados.append(f"{'desde' if op == '>=' else 'hasta'} {mes(t)}")
            except Exception:  # noqa: BLE001
                pass
    if f.get("diagnostico_texto"):
        m &= _contains(df["dx_etiqueta"], f["diagnostico_texto"]); aplicados.append(f"diagnóstico contiene «{f['diagnostico_texto']}»")
    if f.get("item_texto"):
        m &= _contains(df["descripcion"], f["item_texto"]) | _contains(df["principio_activo"], f["item_texto"])
        aplicados.append(f"ítem contiene «{f['item_texto']}»")
    if f.get("solo_alto_costo"):
        m &= ~df["alto_costo"].astype(str).str.strip().isin(["", "nan", "None"])
        aplicados.append("solo alto costo")
    if f.get("excluir_sin_diagnostico"):
        m &= df["dx_grupo"].astype(str) != SD_GROUP; aplicados.append("sin «Sin diagnóstico registrado»")
    d = df[m]

    metric = spec.get("metrica") or "gasto"
    dims = [x for x in (spec.get("agrupar_por") or []) if x in DIMS][:3]
    keys = [DIMS[x] for x in dims]
    top = int(spec.get("top") or 15)
    orden = spec.get("orden") or ("natural" if any(x in ("anio", "mes") for x in dims) else "desc")

    def _agg(g):
        gasto = g["gasto"].sum()
        pac = g.loc[g["pid"] >= 0, "pid"].nunique()
        cant = g["cantidad"].sum()
        return {"gasto": gasto, "cantidad": cant, "lineas": g["lineas"].sum(), "pacientes": pac,
                "gasto_por_paciente": gasto / pac if pac else np.nan, "precio_medio": gasto / cant if cant else np.nan}

    tot = _agg(d) if len(d) else {k: 0 for k in METRICS}
    resumen = {"registros": int(len(d)), "filtros": aplicados, "metrica": metric, "total": tot,
               "periodo": f"{mes(d['tiempo'].min())} a {mes(d['tiempo'].max())}" if len(d) else "—",
               "ultimo_mes": d["tiempo"].max() if len(d) else None}
    if d.empty or not keys:
        return pd.DataFrame(), resumen

    g = d.groupby(keys, observed=True).agg(gasto=("gasto", "sum"), cantidad=("cantidad", "sum"), lineas=("lineas", "sum"))
    g = g.join(d[d["pid"] >= 0].groupby(keys, observed=True)["pid"].nunique().rename("pacientes")).fillna({"pacientes": 0})
    g["gasto_por_paciente"] = g["gasto"] / g["pacientes"].replace(0, np.nan)
    g["precio_medio"] = g["gasto"] / g["cantidad"].replace(0, np.nan)
    g = g[g["lineas"] > 0]
    if metric in ("gasto", "cantidad", "lineas"):
        s = g[metric].sum()
        g["pct"] = g[metric] / s * 100 if s else np.nan
    resumen["n_grupos"] = int(len(g))
    if orden == "natural":
        g = g.sort_index()
        if len(g) > top and any(x in ("anio", "mes") for x in dims):
            g = g.tail(top)
        else:
            g = g.head(top)
    else:
        g = g.sort_values(metric, ascending=(orden == "asc")).head(top)
    g = g.reset_index()
    if "tiempo" in g.columns:
        g["tiempo"] = pd.to_datetime(g["tiempo"]).dt.strftime("%Y-%m")
    cols = keys + [metric] + (["pct"] if "pct" in g.columns else [])
    extra = [c for c in ("gasto", "pacientes") if c != metric and c not in cols]
    g = g[cols + extra]
    g.columns = [DIM_LABEL[dims[keys.index(c)]] if c in keys else (METRICS.get(c) or ("% del total" if c == "pct" else c)) for c in g.columns]
    return g, resumen


def _fmt_val(v, metric):
    if metric in ("gasto", "gasto_por_paciente", "precio_medio"):
        return "S/ " + num(v, 0)
    return num(v, 0)


def tabla_a_texto(tb: pd.DataFrame, resumen: dict) -> str:
    """Serializa el resultado para el modelo (compacto, JSON)."""
    out = {"periodo": resumen["periodo"], "filtros_aplicados": resumen["filtros"], "registros": resumen["registros"],
           "total": {k: (None if (isinstance(v, float) and np.isnan(v)) else round(float(v), 2)) for k, v in resumen["total"].items()},
           "grupos_totales": resumen.get("n_grupos", 0)}
    if not tb.empty:
        t = tb.head(MAX_ROWS_LLM).copy()
        for c in t.columns:
            if pd.api.types.is_float_dtype(t[c]):
                t[c] = t[c].astype("float64").round(2)
        out["filas"] = json.loads(t.to_json(orient="records", force_ascii=False))
    return json.dumps(out, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Modo IA (Claude con uso de herramientas)
# ---------------------------------------------------------------------------
def _contexto(df: pd.DataFrame) -> str:
    def vals(c, n=60):
        return ", ".join(map(str, sorted(df[c].dropna().astype(str).unique().tolist())[:n]))
    return f"""Eres el asistente de datos del «Observatorio del Consumo por IAFAS SaludPol, FF.AA. y Oncosalud» del Instituto Nacional de
Enfermedades Neoplásicas (INEN, Perú). Respondes en español, con frases breves y claras, sin guiones largos y usando dos puntos solo si es imprescindible.

Datos disponibles: consumo facturado por el INEN a las IAFAS, serie mensual {mes(df['tiempo'].min())} a {mes(df['tiempo'].max())}.
Cada línea es un medicamento, insumo o procedimiento. Gasto = cantidad × precio (S/), es el siniestro incurrido por la IAFAS en el INEN,
no incluye lo que paga a otros prestadores. Paciente = identificador seudonimizado (no hay DNI). Los pacientes únicos no se suman entre filas.
El último mes puede estar incompleto por rezago de facturación; 2026 no es un año completo.

Valores válidos:
- IAFAS: {vals('iafas')}
- Tipo: MEDICAMENTO, INSUMO, PROCEDIMIENTO
- Grupos de diagnóstico: {vals('dx_grupo')}
- Ámbito agrupado: {vals('ambito_macro')}
- Sexo: {vals('sexo')} · Grupo etario: {vals('edad_grupo')}
- Departamentos: {vals('departamento', 30)}

Reglas:
1. Toda cifra debe salir de la herramienta consultar_datos; nunca la inventes ni la estimes. Si la pregunta no se puede responder con estos datos, dilo.
2. Para variaciones (crecimiento, comparación entre años) pide la serie y calcula tú la diferencia, indicando qué años comparas.
3. Montos en soles con separador de miles por espacio y coma decimal (S/ 1 234 567). Porcentajes con un decimal.
4. Cierra con una advertencia breve solo cuando aplique (año incompleto, pocos pacientes, «Sin diagnóstico registrado» con mucho peso).
5. No muestres JSON ni nombres de parámetros. Responde en 2 a 6 frases o una lista corta; la tabla de detalle se muestra aparte.
"""


def _api_key() -> str | None:
    k = st.session_state.get("asist_key") or None
    if not k:
        try:
            k = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:  # noqa: BLE001
            k = None
    return k or os.environ.get("ANTHROPIC_API_KEY")


def _modelo() -> str:
    try:
        return st.secrets.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    except Exception:  # noqa: BLE001
        return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


def responder_llm(df: pd.DataFrame, pregunta: str, historial: list[dict], key: str):
    import anthropic  # import diferido: solo se requiere en modo IA

    client = anthropic.Anthropic(api_key=key)
    msgs = []
    for h in historial[-6:]:              # memoria corta de la conversación (solo texto)
        msgs.append({"role": "user", "content": h["q"]})
        msgs.append({"role": "assistant", "content": h["a"]})
    msgs.append({"role": "user", "content": pregunta})
    ultima_tabla, ultimo_spec = pd.DataFrame(), None
    for _ in range(MAX_TOOL_ROUNDS):
        r = client.messages.create(model=_modelo(), max_tokens=1200, system=_contexto(df), tools=[TOOL], messages=msgs)
        if r.stop_reason != "tool_use":
            texto = "".join(b.text for b in r.content if b.type == "text").strip()
            return texto or "No obtuve respuesta.", ultima_tabla, ultimo_spec
        msgs.append({"role": "assistant", "content": r.content})
        res = []
        for b in r.content:
            if b.type == "tool_use":
                try:
                    tb, rs = consultar(df, b.input or {})
                    if not tb.empty:
                        ultima_tabla, ultimo_spec = tb, b.input
                    res.append({"type": "tool_result", "tool_use_id": b.id, "content": tabla_a_texto(tb, rs)})
                except Exception as e:  # noqa: BLE001
                    res.append({"type": "tool_result", "tool_use_id": b.id, "content": f"Error: {e}", "is_error": True})
        msgs.append({"role": "user", "content": res})
    return "La consulta requirió demasiados pasos; intenta formularla de forma más específica.", ultima_tabla, ultimo_spec


# ---------------------------------------------------------------------------
# Modo básico (reglas, sin clave)
# ---------------------------------------------------------------------------
_IAFAS_ALIAS = {"saludpol": "SALUDPOL", "policia": "SALUDPOL", "pnp": "SALUDPOL", "oncosalud": "ONCOSALUD",
                "fospeme": "FOSPEME", "ejercito": "FOSPEME", "fosmar": "FOSMAR", "marina": "FOSMAR",
                "fosfap": "FOSFAP", "fuerza aerea": "FOSFAP", "ffaa": "__MIL__", "ff.aa": "__MIL__", "fuerzas armadas": "__MIL__"}
_DIM_WORDS = [  # (patrón, dimensión) en orden de prioridad
    (r"por mes|mensual|cada mes", "mes"), (r"por ano|anual|cada ano|por anos|evolucion|tendencia", "anio"),
    (r"por iafas|entre iafas|cada iafas|compar", "iafas"), (r"principio activo|molecula", "principio_activo"),
    (r"medicamento|farmaco|insumo|procedimiento|item|producto", "item"),
    (r"grupo de diagnostico|tipo de cancer|sitio tumoral|por cancer|por diagnostico", "dx_grupo"),
    (r"\bcie\b|cie-?10|codigo de diagnostico|diagnosticos especificos", "diagnostico"),
    (r"ambito|hospitaliza|emergencia|ambulatori|quirurgic", "ambito_macro"), (r"tarifario", "grupo_tarifario"),
    (r"sexo|hombres|mujeres|genero", "sexo"), (r"edad|etario", "edad_grupo"),
    (r"departamento|region|residencia", "departamento"), (r"por tipo|tipo de consumo", "tipo"),
]


def interpretar(df: pd.DataFrame, q: str) -> dict:
    n = _norm(q)
    f: dict = {}
    # IAFAS
    iaf = []
    for k, v in _IAFAS_ALIAS.items():
        if re.search(rf"\b{re.escape(k)}\b", n):
            iaf += ["FOSPEME", "FOSMAR", "FOSFAP"] if v == "__MIL__" else [v]
    if iaf:
        f["iafas"] = sorted(set(iaf))
    # años
    anios = sorted({int(a) for a in re.findall(r"\b(20[12]\d)\b", n)})
    if len(anios) == 1 and not re.search(r"desde|a partir", n):
        f["anio_desde"] = f["anio_hasta"] = anios[0]
    elif len(anios) == 1:
        f["anio_desde"] = anios[0]
    elif len(anios) >= 2:
        f["anio_desde"], f["anio_hasta"] = anios[0], anios[-1]
    # tipo
    tipos = [t for t, pat in (("MEDICAMENTO", r"medicamento|farmaco|medicina"), ("INSUMO", r"insumo|material"),
                              ("PROCEDIMIENTO", r"procedimiento")) if re.search(pat, n)]
    if tipos:
        f["tipo"] = tipos
    if re.search(r"alto costo", n):
        f["solo_alto_costo"] = True
    if re.search(r"sin (el )?(diagnostico registrado|sd)\b|excluy\w* sin diagnostico", n):
        f["excluir_sin_diagnostico"] = True
    # grupos de diagnóstico mencionados
    _stop = {"otros", "otras", "tumores", "vias", "sitio", "primario", "desconocido", "definidos", "secundarios", "codigo",
             "comportamiento", "incierto", "neoplasias", "tejidos", "blandos", "cuerpo", "glandulas", "situ", "intestino"}
    dxg = [g for g in df["dx_grupo"].dropna().astype(str).unique() if g != SD_GROUP
           and any(re.search(rf"\b{w[:-1] if len(w) > 5 else w}", n) for w in re.findall(r"[a-z]{4,}", _norm(g)) if w not in _stop)]
    if dxg and not re.search(r"por (grupo de )?diagnostico|tipo de cancer", n):
        f["dx_grupo"] = dxg
    # sexo / departamento
    if re.search(r"\bmujeres\b|femenin", n) and "sexo" not in n:
        f["sexo"] = [s for s in df["sexo"].astype(str).unique() if _norm(s).startswith(("f", "muj"))]
    if re.search(r"\bhombres\b|masculin|varones", n) and "sexo" not in n:
        f["sexo"] = [s for s in df["sexo"].astype(str).unique() if _norm(s).startswith(("m", "hom", "var")) and not _norm(s).startswith("muj")]
    deps = [d for d in df["departamento"].dropna().astype(str).unique() if len(d) > 3 and re.search(rf"\b{re.escape(_norm(d))}\b", n)]
    if deps:
        f["departamento"] = deps
    # ítem textual: «de pembrolizumab», «del trastuzumab»
    mt = re.search(r"(?:gasto|consumo|cantidad|pacientes|precio)\s+(?:en|de|del|con)\s+([a-z][a-z\- ]{4,30}?)(?:\s+(?:en|por|de|del|entre|desde|durante)\b|[?.,]|$)", n)
    if mt:
        cand = mt.group(1).strip()
        stop = {"medicamentos", "insumos", "procedimientos", "la iafas", "las iafas", "cada iafas", "alto costo", "pacientes", "oncosalud", "saludpol"}
        dx_words = {w for g in f.get("dx_grupo", []) for w in re.findall(r"[a-z]{4,}", _norm(g))}
        if re.search(r"cancer|tumor|neoplasia|diagnostico", cand) or any(w[:5] in cand for w in dx_words):
            cand = ""
        if cand and cand not in stop and not any(cand.startswith(s) for s in ("medicament", "insumo", "procedim", "los ", "las ", "la ", "el ")):
            if _contains(df["descripcion"], cand).any() or _contains(df["principio_activo"], cand).any():
                f["item_texto"] = cand
    # métrica
    if re.search(r"por paciente|costo medio|costo promedio|severidad", n):
        metrica = "gasto_por_paciente"
    elif re.search(r"precio", n):
        metrica = "precio_medio"
    elif re.search(r"cuantos pacientes|numero de pacientes|pacientes unicos|pacientes atendidos|\bpacientes\b", n) and "gasto" not in n:
        metrica = "pacientes"
    elif re.search(r"cantidad|unidades|cuantas unidades", n):
        metrica = "cantidad"
    elif re.search(r"lineas|atenciones", n):
        metrica = "lineas"
    else:
        metrica = "gasto"
    # agrupación
    dims = []
    for pat, d in _DIM_WORDS:
        if re.search(pat, n) and d not in dims:
            if d == "item" and "principio_activo" in dims:
                continue
            if d == "item" and not re.search(r"top|mas|mayor|principales|ranking|cuales|que (medicamentos|insumos|procedimientos)|lista", n):
                continue
            if d == "ambito_macro" and "ambito_macro" in f:
                continue
            if d == "iafas" and len(f.get("iafas", [])) == 1:
                continue
            if d == "sexo" and "sexo" in f:
                continue
            if d == "anio" and "mes" in dims:
                continue
            dims.append(d)
    if not dims and (re.search(r"top|mas|mayor|principales|ranking", n)):
        dims = ["item"]
    mtop = re.search(r"\btop\s*(\d{1,3})|\b(\d{1,3})\s+(?:principales|primer|mayores|medicamentos|insumos|procedimientos|diagnosticos|items)", n)
    top = int(next(x for x in mtop.groups() if x)) if mtop else (100 if "mes" in dims else 15)
    orden = "asc" if re.search(r"menor|menos", n) and not re.search(r"al menos", n) else None
    spec = {"metrica": metrica, "agrupar_por": dims[:2], "top": top, "filtros": f}
    if orden:
        spec["orden"] = orden
    return spec


def describir(tb: pd.DataFrame, rs: dict) -> str:
    met = rs["metrica"]
    tot = rs["total"].get(met, np.nan)
    partes = [f"**{METRICS[met]}** · {rs['periodo']}"]
    if rs["filtros"]:
        partes.append("Filtros: " + "; ".join(rs["filtros"]) + ".")
    if rs["registros"] == 0:
        return "No hay registros con esos criterios. Prueba con menos filtros o revisa los del panel."
    if met != "pacientes" or tb.empty:
        partes.append(f"Total: **{_fmt_val(tot, met)}** ({num(rs['total']['pacientes'])} pacientes únicos).")
    else:
        partes.append(f"Total: **{num(tot)}** pacientes únicos (no se suman entre filas).")
    if not tb.empty:
        first, vcol = tb.iloc[0], tb.columns[len([c for c in tb.columns if c in DIM_LABEL.values()])]
        dcols = [c for c in tb.columns if c in DIM_LABEL.values()]
        if dcols and not any(c in ("Año", "Mes") for c in dcols):
            lbl = " · ".join(str(first[c]) for c in dcols)
            extra = f" ({num(first['% del total'], 1)}% del total)" if "% del total" in tb.columns else ""
            partes.append(f"Primer lugar: **{lbl}** con {_fmt_val(first[vcol], met)}{extra}.")
        elif dcols and len(tb) >= 2 and dcols == ["Año"]:
            a, b = tb.iloc[-2], tb.iloc[-1]
            um = rs.get("ultimo_mes")
            parcial = um is not None and int(b["Año"]) == um.year and um.month < 12
            if parcial and len(tb) >= 3:
                a, b = tb.iloc[-3], tb.iloc[-2]
                partes.append(f"{int(tb.iloc[-1]['Año'])} es un año incompleto (hasta {mes(um)}); se comparan los dos últimos años completos.")
                parcial = False
            if a[vcol]:
                var = (b[vcol] / a[vcol] - 1) * 100
                partes.append(f"Entre {int(a['Año'])} y {int(b['Año'])} la variación es de {num(var, 1)}%"
                              + (" (el último año está incompleto)." if parcial else "."))
    partes.append("_Modo básico: para preguntas libres activa el modo IA con una clave de Anthropic._")
    return "  \n".join(partes)


# ---------------------------------------------------------------------------
# Interfaz (barra lateral)
# ---------------------------------------------------------------------------
EJEMPLOS = [
    "¿Cuál es el gasto total por IAFAS?",
    "Top 10 medicamentos de SALUDPOL en 2025",
    "Evolución anual del gasto de ONCOSALUD",
    "¿Cuántos pacientes por grupo de diagnóstico en FF.AA.?",
    "Gasto por paciente según ámbito en 2024",
    "Medicamentos de alto costo por IAFAS",
]


def _grafico(tb: pd.DataFrame, height: int = 200):
    dcols = [c for c in tb.columns if c in DIM_LABEL.values()]
    if len(dcols) != 1 or len(tb) < 2:
        return
    import plotly.graph_objects as go
    from ui import PRIMARY
    vcol = tb.columns[1]
    if dcols[0] in ("Año", "Mes"):
        x = tb[dcols[0]].astype(str)
        fig = go.Figure(go.Scatter(x=x, y=tb[vcol], mode="lines+markers", line=dict(color=PRIMARY, width=2)))
        fig.update_xaxes(type="category")
    else:
        t = tb.head(12).iloc[::-1]
        lab = t[dcols[0]].astype(str).str.slice(0, 28)
        fig = go.Figure(go.Bar(x=t[vcol], y=lab, orientation="h", marker_color=PRIMARY))
        fig.update_yaxes(type="category", tickfont=dict(size=10))
        height = max(height, 22 * len(t) + 40)
    fig.update_layout(height=height, margin=dict(l=0, r=8, t=6, b=4), plot_bgcolor="white", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False, hovermode="closest")
    fig.update_xaxes(gridcolor="#eceff4", tickfont=dict(size=10), nticks=4, tickangle=0)
    st.plotly_chart(fig, config={"displayModeBar": False}, **STRETCH)


def _tabla(tb: pd.DataFrame, height: int | None = None):
    fm = {}
    for c in tb.columns:
        if pd.api.types.is_numeric_dtype(tb[c]) and c not in ("Año",):
            fm[c] = (lambda v: num(v, 1)) if c == "% del total" else (lambda v: num(v, 0))
    st.dataframe(tb.style.format(fm, na_rep="—"), hide_index=True, **STRETCH, height=height or "auto")


@st.dialog("Resultado de la consulta", width="large")
def _ampliar(i: int):
    h = st.session_state["asist_hist"][i]
    st.markdown(f"**Pregunta:** {h['q']}")
    st.markdown(h["a"])
    if h["tabla"] is not None and not h["tabla"].empty:
        _grafico(h["tabla"], 320)
        _tabla(h["tabla"], 420)
        st.download_button("Descargar CSV", h["tabla"].to_csv(index=False).encode("utf-8-sig"),
                           file_name="consulta_asistente_iafas.csv", mime="text/csv", key=f"asist_dl_{i}")


def render(df_filtrado: pd.DataFrame, df_total: pd.DataFrame):
    """Dibuja el asistente. Debe llamarse dentro del contexto de la barra lateral."""
    st.session_state.setdefault("asist_hist", [])
    with st.expander("💬 Pregunta a los datos", expanded=True):
        key = _api_key()
        usar_flt = st.toggle("Aplicar los filtros del panel", value=True, key="asist_flt",
                             help="Si está activo, el asistente responde sobre la selección actual (meses, tipo, diagnóstico, demografía).")
        with st.form("asist_form", clear_on_submit=True, border=False):
            q = st.text_area("Tu pregunta", placeholder="Ej.: ¿qué medicamentos explican más gasto en SALUDPOL en 2025?",
                             height=80, label_visibility="collapsed")
            c1, c2 = st.columns([1, 1])
            enviar = c1.form_submit_button("Consultar", type="primary", **STRETCH)
            borrar = c2.form_submit_button("Limpiar", **STRETCH)
        ej = st.selectbox("Ejemplos", ["Elige un ejemplo…"] + EJEMPLOS, index=0, key="asist_ej", label_visibility="collapsed")
        if borrar:
            st.session_state["asist_hist"] = []
        pregunta = (q or "").strip()
        if not pregunta and ej != "Elige un ejemplo…" and st.session_state.get("asist_ej_prev") != ej:
            pregunta = ej
        st.session_state["asist_ej_prev"] = ej

        base = df_filtrado if usar_flt else df_total
        if enviar or pregunta:
            if pregunta:
                with st.spinner("Consultando…"):
                    try:
                        if key:
                            a, tb, spec = responder_llm(base, pregunta, st.session_state["asist_hist"], key)
                            modo = "IA"
                        else:
                            spec = interpretar(base, pregunta)
                            tb, rs = consultar(base, spec)
                            a, modo = describir(tb, rs), "básico"
                    except Exception as e:  # noqa: BLE001
                        a, tb, spec, modo = f"No se pudo completar la consulta: {e}", pd.DataFrame(), None, "error"
                st.session_state["asist_hist"].append({"q": pregunta, "a": a, "tabla": tb, "spec": spec, "modo": modo})

        hist = st.session_state["asist_hist"]
        if hist:
            i = len(hist) - 1
            h = hist[i]
            st.markdown(f"<div class='asist-q'>🧑 {h['q']}</div>", unsafe_allow_html=True)
            st.markdown(h["a"])
            if h["tabla"] is not None and not h["tabla"].empty:
                _grafico(h["tabla"])
                _tabla(h["tabla"].head(10))
                if st.button("🔍 Ampliar y descargar", key="asist_amp", **STRETCH):
                    _ampliar(i)
            if h.get("spec"):
                with st.popover("Cómo se calculó", **STRETCH):
                    st.json(h["spec"], expanded=True)
            if len(hist) > 1:
                with st.popover(f"Historial ({len(hist) - 1})", **STRETCH):
                    for j in range(len(hist) - 2, -1, -1):
                        st.markdown(f"**{hist[j]['q']}**")
                        st.markdown(hist[j]["a"])
                        st.divider()

        st.caption(("🟢 Modo IA activo (Claude). " if key else "⚪ Modo básico por reglas. ")
                   + "Las cifras salen de agregaciones sobre los datos del observatorio; al modelo solo se envían tablas resumidas, sin identificadores.")
        if not key:
            with st.popover("Activar modo IA", **STRETCH):
                st.text_input("Clave de API de Anthropic", type="password", key="asist_key",
                              help="Se guarda solo en esta sesión. En Streamlit Cloud, colócala en Secrets como ANTHROPIC_API_KEY.")
