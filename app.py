"""
Observatorio del Consumo por IAFAS | INEN
Dashboard Streamlit: una pestaña por IAFAS con resumen e ingreso, ámbito de atención, diagnósticos, medicamentos,
insumos, procedimientos, gasto, tendencias y proyecciones. Misma arquitectura que el
Observatorio del Financiamiento Oncológico, sin el módulo de impacto normativo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from data_processing import (
    ATENCION_ORDER, ESTADIO_ORDER, VITAL_ORDER, EDAD_ORDER, SD_GROUP, TIPO_LABEL, TIPO_LIST, apply_filters, load_afiliados, load_fact, load_primas, month_index,
)
import panels as P
import reco as R
import asistente as A
import proyecciones_comp as PC
import desenlace as DS
from ui import CSS, mes, num, STRETCH, Opts, fmt_df, interp, warn

st.set_page_config(page_title="Observatorio del Consumo por IAFAS | INEN", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

fact = load_fact()
MIN_M, MAX_M = fact["tiempo"].min(), fact["tiempo"].max()
IAFAS_ORDER = fact.groupby("iafas", observed=True)["gasto"].sum().sort_values(ascending=False).index.tolist()

st.markdown(
    f"""<div class="app-header"><h1>Observatorio del Consumo por IAFAS SaludPol, FF.AA. y Oncosalud</h1>
<p>Instituto Nacional de Enfermedades Neoplásicas · Consumo facturado a {len(IAFAS_ORDER)} IAFAS · serie mensual {mes(MIN_M)} a {mes(MAX_M)}</p></div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Panel de control")
    asist_box = st.container()        # asistente de consultas: se llena cuando ya existen los filtros
    st.divider()
    st.markdown("**Filtros**")
    st.caption("Los filtros se aplican a todas las pestañas.")

    _meses = list(pd.date_range(MIN_M, MAX_M, freq="MS"))
    rng = st.select_slider("Rango de meses", options=_meses, value=(_meses[0], _meses[-1]), format_func=mes)
    excl_last = st.checkbox("Excluir el último mes de la serie", value=False,
                            help=f"{mes(MAX_M)} puede estar incompleto por rezago de facturación.")
    tipos = st.multiselect("Tipo de consumo", TIPO_LIST, default=TIPO_LIST, format_func=lambda x: TIPO_LABEL[x])

    aten_sel = st.multiselect("Tipo de atención (vacío = todos)", [x for x in ATENCION_ORDER if x in set(fact["atencion"].astype(str))], default=[],
                              help="Dato registrado por el INEN en cada línea (campo TIPO_ATENCION): ambulatorio u hospitalizado.")

    st.divider()
    st.markdown("**Diagnóstico**")
    dx_all = sorted(fact["dx_grupo"].dropna().unique().tolist())
    dx_sel = st.multiselect("Grupo de diagnóstico (vacío = todos)", dx_all, default=[])
    excl_sd = st.checkbox("Excluir «Sin diagnóstico registrado»", value=False,
                          help="Son líneas sin código CIE-10 en el origen; concentran una parte grande del gasto.")
    est_sel = st.multiselect("Estadío clínico (vacío = todos)", [x for x in ESTADIO_ORDER if x in set(fact["estadio_grupo"].astype(str))], default=[],
                             help="Estadío registrado en la línea (DX_ESTADIO), agrupado: IIIb cuenta como III. «Desconocido» = registrado como desc.")
    vit_sel = st.multiselect("Condición vital (vacío = todos)", VITAL_ORDER, default=[],
                             help="Según la fecha de deceso de la extracción. Al filtrar se excluyen las líneas sin paciente identificado.")
    with st.expander("Filtros demográficos"):
        sexo_sel = st.multiselect("Sexo", sorted(fact["sexo"].unique().tolist()), default=[])
        edad_sel = st.multiselect("Grupo etario", [e for e in EDAD_ORDER if e in set(fact["edad_grupo"].unique())], default=[])
        dep_sel = st.multiselect("Departamento de residencia", sorted(fact["departamento"].unique().tolist()), default=[])

    st.divider()
    st.markdown("**Visualización**")
    chart_type = st.radio("Tipo de gráfico", ["Línea", "Barras"], horizontal=True)
    markers = st.checkbox("Mostrar marcadores", value=True)
    labels = st.checkbox("Mostrar etiquetas de datos", value=False)
    lowess = st.checkbox("Línea suavizada (LOWESS)", value=False)
    log_scale = st.checkbox("Escala logarítmica", value=False)
    alpha = st.select_slider("Umbral de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05, format_func=lambda x: str(x).replace(".", ","))

    st.divider()
    st.markdown("**Alto costo**")
    umbral = st.number_input("Umbral de paciente de alto costo (S/ acumulado)", min_value=1000.0, value=50000.0, step=5000.0)

    with st.expander("💰 Ingreso y afiliados de la IAFAS"):
        st.caption("La extracción no incluye ingresos. Ingresa la prima o ingreso anual de cada IAFAS (en soles) para obtener el "
                   "peso del INEN en ese ingreso (gasto INEN / ingreso). Se puede dejar vacío.")
        up = st.file_uploader("Cargar CSV (iafas, anio, prima_anual)", type=["csv"], key="up_primas")
        base_primas = load_primas()
        if up is not None:
            try:
                base_primas = pd.read_csv(up)[["iafas", "anio", "prima_anual"]]
            except Exception as e:  # noqa: BLE001
                st.error(f"No se pudo leer el CSV: {e}")
        if "primas_df" not in st.session_state or up is not None:
            st.session_state["primas_df"] = base_primas
        edited = st.data_editor(
            st.session_state["primas_df"], num_rows="dynamic", hide_index=True, **STRETCH, key="primas_editor",
            column_config={
                "iafas": st.column_config.SelectboxColumn("IAFAS", options=IAFAS_ORDER, required=True),
                "anio": st.column_config.NumberColumn("Año", min_value=2019, max_value=2030, step=1, format="%d"),
                "prima_anual": st.column_config.NumberColumn("Prima anual (S/)", min_value=0.0, format="%.0f"),
            },
        )
        st.session_state["primas_df"] = edited
        st.caption("Afiliados por IAFAS y año (SUSALUD). Permiten calcular la tasa de utilización y el costo por afiliado. Los años sin dato usan el último disponible.")
        if "afil_df" not in st.session_state:
            st.session_state["afil_df"] = load_afiliados()
        afil_ed = st.data_editor(
            st.session_state["afil_df"], num_rows="dynamic", hide_index=True, **STRETCH, key="afil_editor",
            column_config={
                "iafas": st.column_config.SelectboxColumn("IAFAS", options=IAFAS_ORDER, required=True),
                "anio": st.column_config.NumberColumn("Año", min_value=2019, max_value=2030, step=1, format="%d"),
                "afiliados": st.column_config.NumberColumn("Afiliados", min_value=0, format="%d"),
            },
        )
        st.session_state["afil_df"] = afil_ed

date_lo, date_hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
if excl_last and date_hi >= MAX_M:
    date_hi = MAX_M - pd.DateOffset(months=1)

opts = Opts(
    date_lo=date_lo, date_hi=date_hi, chart_type=chart_type, markers=markers, labels=labels, lowess=lowess, log=log_scale,
    alpha=alpha, umbral_alto_costo=umbral,
    primas=edited.dropna(subset=["iafas", "anio", "prima_anual"]) if len(edited) else edited,
    afiliados=afil_ed.dropna(subset=["iafas", "anio", "afiliados"]) if len(afil_ed) else afil_ed,
)

flt = apply_filters(fact, tipos=tipos or None, date_lo=date_lo, date_hi=date_hi, dx_grupos=dx_sel or None,
                    sexo=sexo_sel or None, edad=edad_sel or None, deptos=dep_sel or None, excluir_sd=excl_sd,
                    atencion=aten_sel or None, estadios=est_sel or None, vital=vit_sel or None)

n_filters = sum(bool(x) for x in (dx_sel, sexo_sel, edad_sel, dep_sel, aten_sel, est_sel, vit_sel)) + int(excl_sd) + int(len(tipos) < 3)
st.caption(f"**Selección:** {mes(date_lo)} a {mes(date_hi)} · {len(tipos)} tipo(s) de consumo · "
           f"{'sin filtros adicionales' if n_filters == 0 else f'{n_filters} filtro(s) adicional(es) activo(s)'} · "
           f"{len(flt):,} registros agregados")

with st.sidebar, asist_box:
    A.render(flt, fact)

if flt.empty:
    st.warning("La selección no devuelve datos. Amplía el rango o quita filtros.")
    st.stop()

# ---------------------------------------------------------------------------
# Navegación por IAFAS
# ---------------------------------------------------------------------------
ONCO = "ONCOSALUD"
MIL = ["FOSPEME (Ejército)", "FOSMAR (Marina de Guerra)", "FOSFAP (Fuerza Aérea)"]
POOL = "Recomendaciones SALUDPOL y FF.AA."
IAFAS_TABS = [i for i in IAFAS_ORDER if i != ONCO] + [ONCO]   # ONCOSALUD al final
LIQ = "Producción vs liquidaciones"
NAV = ["Comparativo"] + IAFAS_TABS[:-1] + [POOL, ONCO, LIQ, "Tabla dinámica", "Metodología"]
choice = st.segmented_control("Pestañas", NAV, default="Comparativo", key="nav", label_visibility="collapsed")
choice = choice or "Comparativo"
st.divider()

SECTIONS = ["📌 Resumen e ingreso", "🏥 Tipo de atención", "🩺 Diagnósticos", "🎗️ Estadío y desenlace", "💊 Medicamentos", "🧰 Insumos", "🔬 Procedimientos",
            "💰 Gasto y evolución", "📈 Tendencias", "🔮 Proyecciones", "🧭 Recomendaciones"]

if choice == "Comparativo":
    st.subheader("Comparativo entre IAFAS")
    P.panel_comparativo(flt, IAFAS_TABS, opts)

elif choice == LIQ:
    import liquidaciones as LQ
    st.subheader("Producción vs liquidaciones")
    LQ.panel_liquidaciones(fact, opts, n_filters > 0)

elif choice == POOL:
    pool_names = ["SALUDPOL"] + MIL
    d_p = flt[flt["iafas"].isin(pool_names)]
    d_o = flt[~flt["iafas"].isin(pool_names)]
    st.subheader("Recomendaciones para SALUDPOL y FF.AA. (análisis conjunto)")
    if d_p.empty:
        st.warning("Estas IAFAS no tienen consumo con los filtros actuales.")
        st.stop()
    rows = []
    for nm in pool_names:
        x = d_p[d_p["iafas"] == nm]
        if len(x):
            rows.append({"IAFAS": nm, "Gasto": x["gasto"].sum(), "% del conjunto": x["gasto"].sum() / d_p["gasto"].sum() * 100,
                         "Pacientes": x.loc[x.pid >= 0, "pid"].nunique(), "Primer mes": x["tiempo"].min()})
    tp = pd.DataFrame(rows)
    st.caption("Se analizan juntas para tener más pacientes y datos que por separado; las demás IAFAS (ONCOSALUD) sirven de referencia para la credibilidad.")
    st.dataframe(fmt_df(tp, money=["Gasto"], pcts=["% del conjunto"], ints=["Pacientes"], dates=["Primer mes"]), hide_index=True, **STRETCH)
    st.caption(f"Pacientes únicos del conjunto: {d_p.loc[d_p.pid >= 0, 'pid'].nunique():,} · gasto {d_p['gasto'].sum():,.0f} soles. "
               "Un paciente puede estar en más de una IAFAS, por eso la suma de la tabla puede superar el total.")
    idx = month_index(d_p["tiempo"].min(), min(date_hi, MAX_M))
    R.panel_recomendaciones(d_p, d_o, "SALUDPOL y FF.AA.", idx, opts)

elif choice in IAFAS_TABS:
    d_i = flt[flt["iafas"] == choice]
    d_r = flt[flt["iafas"] != choice]
    if d_i.empty:
        st.warning(f"{choice} no tiene consumo con los filtros actuales.")
        st.stop()
    idx = month_index(d_i["tiempo"].min(), min(date_hi, MAX_M))
    st.subheader(choice)
    st.caption(f"Consumo desde {mes(d_i['tiempo'].min())} hasta {mes(d_i['tiempo'].max())} · "
               f"{d_i.loc[d_i.pid >= 0, 'pid'].nunique():,} pacientes · gasto {d_i['gasto'].sum():,.0f} soles")
    P.small_sample_note(d_i, idx, choice)
    tabs = st.tabs(SECTIONS)
    with tabs[0]:
        P.panel_siniestralidad(d_i, choice, idx, opts)
    with tabs[1]:
        P.panel_ambito(d_i, choice, idx, opts)
    with tabs[2]:
        P.panel_diagnosticos(d_i, choice, idx, opts, d_r)
    with tabs[3]:
        DS.panel_desenlace(d_i, choice, opts, d_r)
    with tabs[4]:
        P.panel_items(d_i, d_r, choice, "MEDICAMENTO", idx, opts)
    with tabs[5]:
        P.panel_items(d_i, d_r, choice, "INSUMO", idx, opts)
    with tabs[6]:
        P.panel_items(d_i, d_r, choice, "PROCEDIMIENTO", idx, opts)
    with tabs[7]:
        P.panel_gasto(d_i, choice, idx, opts)
    with tabs[8]:
        P.panel_tendencias(d_i, choice, idx, opts)
    with tabs[9]:
        P.panel_proyecciones(d_i, choice, idx, opts)
        PC.panel_componentes(d_i, choice, idx, opts)
    with tabs[10]:
        R.panel_recomendaciones(d_i, d_r, choice, idx, opts)

elif choice == "Tabla dinámica":
    st.subheader("Tabla dinámica y descarga")
    dims = {"IAFAS": "iafas", "Año": "anio", "Mes": "tiempo", "Tipo de consumo": "tipo", "Grupo de diagnóstico": "dx_grupo",
            "Diagnóstico (CIE-10)": "dx_etiqueta", "Ítem (descripción)": "descripcion", "Principio activo": "principio_activo",
            "Tipo de atención": "atencion", "Servicio del procedimiento (tarifario)": "ambito", "Estadío (agrupado)": "estadio_grupo",
            "Estadío (detalle)": "estadio", "Condición vital": "condicion_vital", "Grupo del tarifario": "grupo_tarifario", "Sexo": "sexo", "Grupo etario": "edad_grupo", "Departamento": "departamento"}
    c = st.columns([2, 2, 1.2, 1.2])
    rows = c[0].multiselect("Filas", list(dims), default=["IAFAS", "Año"])
    cols = c[1].multiselect("Columnas (opcional)", [x for x in dims if x not in rows], default=[])
    val = c[2].selectbox("Valor", ["Gasto (S/)", "Cantidad", "Líneas", "Pacientes únicos"])
    topk = c[3].number_input("Máx. filas", 10, 5000, 500, 50)
    if not rows:
        st.info("Elige al menos una dimensión en filas.")
    else:
        keys = [dims[x] for x in rows + cols]
        src = flt if val != "Pacientes únicos" else flt[flt["pid"] >= 0]
        if val == "Pacientes únicos":
            g = src.groupby(keys, observed=True)["pid"].nunique().rename("valor")
        else:
            g = src.groupby(keys, observed=True)[{"Gasto (S/)": "gasto", "Cantidad": "cantidad", "Líneas": "lineas"}[val]].sum().rename("valor")
        if cols:
            tb = g.unstack([dims[x] for x in cols]).fillna(0)
            tb["Total"] = tb.sum(axis=1)
        else:
            tb = g.to_frame(val)
        tb = tb.sort_values(tb.columns[-1], ascending=False).head(int(topk)).reset_index()
        if "tiempo" in tb.columns:
            tb["tiempo"] = pd.to_datetime(tb["tiempo"]).dt.strftime("%Y-%m")
        tb.columns = [" · ".join(map(str, c_)) if isinstance(c_, tuple) else c_ for c_ in tb.columns]
        _dec = 0 if val in ("Líneas", "Pacientes únicos") else 1
        _fm = {c_: (lambda v: num(v, _dec)) for c_ in tb.columns if pd.api.types.is_numeric_dtype(tb[c_])}
        st.dataframe(tb.style.format(_fm, na_rep="—"), hide_index=True, **STRETCH, height=520)
        st.download_button("Descargar CSV", tb.to_csv(index=False).encode("utf-8-sig"), file_name="tabla_dinamica_iafas.csv", mime="text/csv")
        if val == "Pacientes únicos":
            st.caption("Los pacientes únicos no se suman entre filas: un mismo paciente puede aparecer en varias.")

else:
    st.subheader("Metodología y notas")
    st.markdown(f"""
**Fuente.** Extracción `INEN.V_CONSUMO_CUENTA_PACIENTE_IAFAS` (ene-2019 a ago-2026): cada fila es una línea de consumo
(medicina, material médico o procedimiento) facturada a una IAFAS. El archivo se compacta con `build_data.py`; el DNI se
reemplaza por un identificador seudonimizado y no se publica.

**Liquidaciones.** Montos cancelados por cada IAFAS (SALUDPOL, ONCOSALUD, FOSPEME, FOSMAR, FOSFAP) con su fecha de cancelación, remitidos por la Oficina de Contabilidad y
Finanzas (Informe N.° 000271-2026-OCF-OGA/INEN). Se consolidan con `build_liquidaciones.py` en `data/liquidaciones.csv` y se contrastan con la producción en la pestaña
«Producción vs liquidaciones».

**Gasto.** `cantidad × precio` de cada línea. Es el valor facturado por el INEN a la IAFAS, es decir, el siniestro incurrido por la
IAFAS en este hospital; no incluye lo que la IAFAS paga en otros prestadores.

**Peso del INEN en el ingreso de la IAFAS.** Gasto facturado por el INEN dividido entre el ingreso anual de la IAFAS (repartido en doce partes
iguales). No es la siniestralidad total de la IAFAS, que incluiría lo pagado a otros prestadores. La extracción no trae ingresos: se cargan en la
barra lateral o en `data/primas.csv` (hay datos para ONCOSALUD 2022-2023 y SALUDPOL 2024; fuentes en `data/fuentes_primas.md`). Sin ingreso, la pestaña
muestra frecuencia (pacientes activos por mes), severidad (gasto por paciente activo), concentración del gasto y pacientes de alto costo.

**Paciente activo.** Paciente distinto (DNI seudonimizado) con al menos una línea facturada a la IAFAS en el mes. No mide tratamiento continuo y excluye líneas sin DNI.
Frecuencia absoluta = pacientes activos; severidad = gasto por paciente activo (el siniestro es el paciente-mes con consumo); gasto = frecuencia × severidad.
Con afiliados cargados, la frecuencia pasa a tasa de utilización (pacientes activos por 1,000 afiliados) y el costo por afiliado-mes (PMPM) equivale a la prima pura del INEN.
Afiliados: Registro de Afiliados SUSALUD 2019-2023 (`data/afiliados.csv`); los años posteriores repiten el último dato.

**Redondeo.** Porcentajes y montos se muestran con un decimal; se mantienen dos en indicadores de magnitud muy pequeña (tasa de utilización, costo por afiliado-mes) y en coeficientes estadísticos.

**Recomendaciones (transferencia prospectiva y cartas de garantía).** Las tarifas unitarias son las del tarifario INEN; se propone cambiar la modalidad de pago. Monto mensual por paciente en tratamiento = costo medio recortado en un percentil atípico, ponderado con
las demás IAFAS por credibilidad de límite de fluctuación y cargado con margen de riesgo z·σ/√N; lo que supera el umbral se liquida aparte por tarifario y los medicamentos de alto costo se pagan por adelantado. La pestaña «Recomendaciones SALUDPOL y FF.AA.» analiza esas cuatro IAFAS juntas para tener más datos. La carta de garantía mínima es el
percentil elegido del gasto acumulado por paciente durante la vigencia. Detalle y límites en la pestaña Recomendaciones.

**Costo por paciente según diagnóstico.** Para cada diagnóstico, suma de las líneas rotuladas con él por paciente, promediada entre pacientes (con mediana, P90 y costo
anualizado = gasto por mes de consumo × 12). Solo incluye lo facturado por el INEN con diagnóstico y DNI, y la observación está truncada por ambos lados (pacientes que
empezaron antes de ene-2019 o siguen en tratamiento); la opción «Solo pacientes con inicio observado» deja solo a quienes empiezan después de ene-2019 (su tratamiento se ve desde el comienzo) y reduce el primer sesgo.

**Medicamentos de alto costo.** Se marcan con la lista nominal no PNUME (`data/alto_costo.csv`, tomada del listado al 22.09.26): coincidencia por código SIGA y, opcionalmente,
otras presentaciones de la misma molécula. El precio referencial es el de adjudicación del listado (los marcadores de S/ 0.90 no se comparan).

**Tipo de atención.** Desde la versión V02 de la extracción cada línea trae el tipo de atención registrado por el INEN (`TIPO_ATENCION`: ambulatorio u hospitalizado),
que es el que se usa en todo el observatorio. Reemplaza la inferencia anterior, que asignaba a medicamentos e insumos el ámbito de los procedimientos del mismo paciente en el mes
(esa inferencia coincidía con el dato real en 86.7% del gasto en fármacos e insumos). Las líneas sin paciente identificado no traen el campo y figuran como «Sin dato».
Para procedimientos se conserva además el **servicio del tarifario**: cada código se asigna al grupo del Tarifario INEN 2026 al que pertenece (`data/tarifario_inen.csv`, 53 grupos)
y de ahí a consulta externa, apoyo al diagnóstico, quimioterapia y radioterapia, emergencia, hospitalización, centro quirúrgico o procedimientos y terapias ambulatorias
(los códigos sin grupo se clasifican por prefijo). El servicio describe qué se hizo; el tipo de atención, en qué condición estaba el paciente.

**Estadío y desenlace vital.** `DX_ESTADIO` se agrupa en 0, I, II, III y IV (IIIb cuenta como III; I-E como I); «desc» se muestra como «Desconocido». El estadío del paciente es el más
frecuente en las líneas de su diagnóstico de mayor gasto. `FEC_DECESO` se guarda solo como mes. La fecha de corte del estado vital es la fecha de deceso más reciente de la extracción;
quien no tiene fecha de deceso se considera vivo a esa fecha, lo que puede subestimar la mortalidad si hay fallecimientos no registrados. La supervivencia (Kaplan-Meier) se mide
desde el primer mes financiado por la IAFAS en la serie, no desde el diagnóstico: los pacientes que ya se trataban antes de 2019 entran tarde y la curva no equivale a la supervivencia por estadío
de un registro de cáncer.

**Líneas repetidas.** Las líneas de medicamentos e insumos nunca se repiten; en procedimientos, la mitad son filas idénticas porque la
extracción no trae un identificador de línea (p. ej. sesiones de radioterapia o análisis repetidos el mismo día). Se conservan,
porque coinciden con lo facturado; si fueran duplicados de extracción, el gasto de procedimientos estaría sobrestimado.

**Diagnósticos.** Agrupados por sitio tumoral según bloques CIE-10. «Sin diagnóstico registrado» (código SD) agrupa líneas sin código y puede pesar
mucho; usa el filtro de la barra lateral para excluirlo. Los códigos sin descripción en el origen aparecen como «Sin descripción».

**Medicamentos.** Por descripción de presentación o por principio activo (texto previo a la primera cifra de la descripción, una
aproximación). El perfil diferencial compara el peso de cada ítem en la IAFAS con su peso en las demás IAFAS juntas.

**Tendencias.** Mann-Kendall con corrección de autocorrelación (Hamed-Rao, respaldo Yue-Wang) y pendiente de Sen sobre la serie mensual; regresión
lineal sobre años completos; quiebre automático por mínimo de suma de cuadrados con test de Chow (el p-valor es optimista porque la fecha se
busca en los datos); test de Chow en una fecha elegida. La pandemia (mar–dic 2020) puede excluirse de tendencia y proyección.

**Proyecciones.** Holt con tendencia amortiguada (escenario recomendado) con banda por bootstrap de residuos, un escenario conservador (media de los
últimos tres meses), uno lineal con la pendiente de Sen y un escenario propio con la variación anual que definas. No hay estacionalidad explícita
y los picos por un solo medicamento de alto costo pueden mover la proyección: usa «Atenuar picos» para ver la sensibilidad.

**Proyección por componente.** Debajo de la proyección principal se proyectan, con el mismo horizonte, ventana y banda: medicamentos (por defecto los de la lista
nominal de alto costo no PNUME, por principio activo), insumos y procedimientos (por defecto los de mayor gasto) y pacientes activos por diagnóstico. Cada serie empieza en su
primer mes con consumo. Si hay consumo en la mayoría de los meses se usa Holt amortiguado; si más del 30% de los meses están en cero se usa Croston con corrección SBA
(promedio mensual esperado = probabilidad de consumo × monto típico); con menos de 12 meses desde el primer consumo se usa la tasa reciente. La banda se obtiene simulando
1 000 trayectorias futuras, y la banda de los 12 meses sale de esas trayectorias. Un medicamento de alto costo depende de pocos pacientes: la proyección es un valor esperado, no
anticipa el ingreso o la salida de un paciente concreto.

**Tendencia por componente.** Para cada componente (tipo de consumo, diagnóstico, principio activo, insumo, procedimiento o ámbito) se grafica su serie mensual con la recta
de Sen y se compara el APC con su IC 95%.

**Limitaciones.** Las IAFAS pequeñas (FOSFAP, FOSMAR, FOSPEME) tienen pocos pacientes y meses sin consumo; sus tendencias son descriptivas. 2026 llega hasta
agosto. Los meses sin consumo se cuentan como cero.

**Actualizar los datos.** `python build_data.py RUTA/DATA_IAFAS.xlsx` regenera `data/`. Para primas fijas, edita `data/primas.csv`.
""")

# ---------------------------------------------------------------------------
# Autoría y cómo citar
# ---------------------------------------------------------------------------
_YR = 2026
_URL = "https://mlobservatorio-iafas-pnp-ffaa-oncosalud.streamlit.app/"
_T_ES = "Observatorio de Consumo de IAFAS SaludPol, FF.AA. y Oncosalud"
_T_EN = "IAFAS Consumption Observatory: SaludPol, Armed Forces and Oncosalud"
st.divider()
st.markdown(
    """<p class="source-note" style="text-align:center; margin-top:0.6rem;">© Luis A. Orrego Ferreyros, Econ.,
MCE, MMD, PhD(c), CQRM · Epidemiólogo y Economista de la Salud — INEN</p>""",
    unsafe_allow_html=True,
)
st.markdown(
    f"""<div class="citation-box">
<h4>REFERENCIA WEB (formato AMA)</h4>
<p>Orrego-Ferreyros LA. {_T_ES}. Instituto Nacional de Enfermedades Neoplásicas. Published {_YR}. Accessed [Mes Día, Año]. {_URL}</p>
<h4>WEB REFERENCE (AMA format)</h4>
<p>Orrego-Ferreyros LA. {_T_EN}. National Institute of Neoplastic Diseases. Published {_YR}. Accessed [Month Day, Year]. {_URL}</p>
</div>""",
    unsafe_allow_html=True,
)
_ris = (f"TY  - ELEC\nAU  - Orrego-Ferreyros, Luis Alexander\nTI  - {_T_ES} / {_T_EN}\nPY  - {_YR}\n"
        f"PB  - Instituto Nacional de Enfermedades Neoplásicas\nUR  - {_URL}\nER  - \n")
_bib = (f"@misc{{orregoferreyros{_YR}iafas,\n  author = {{Orrego-Ferreyros, Luis Alexander}},\n  title = {{{_T_ES} / {_T_EN}}},\n"
        f"  year = {{{_YR}}},\n  publisher = {{Instituto Nacional de Enfermedades Neopl\\'asicas}},\n  url = {{{_URL}}}\n}}\n")
_xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n<xml><records><record>\n<ref-type name="Web Page">12</ref-type>\n'
        f"<contributors><authors><author>Orrego-Ferreyros, Luis Alexander</author></authors></contributors>\n"
        f"<titles><title>{_T_ES} / {_T_EN}</title></titles>\n<dates><year>{_YR}</year></dates>\n"
        f"<publisher>Instituto Nacional de Enfermedades Neoplásicas</publisher>\n<urls><related-urls><url>{_URL}</url></related-urls></urls>\n"
        f"</record></records></xml>\n")
_d1, _d2, _d3 = st.columns(3)
_d1.download_button("⬇️ BibTeX (*.bib)", _bib, file_name="observatorio_iafas_inen.bib", mime="text/plain", key="cite_bib")
_d2.download_button("⬇️ EndNote XML (*.xml)", _xml, file_name="observatorio_iafas_inen.xml", mime="application/xml", key="cite_xml")
_d3.download_button("⬇️ RIS (*.ris)", _ris, file_name="observatorio_iafas_inen.ris", mime="application/x-research-info-systems", key="cite_ris")
