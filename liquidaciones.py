"""Pestaña «Producción vs liquidaciones»: contrasta el consumo valorizado (producción) con los montos cancelados (liquidaciones)
remitidos por la Oficina de Contabilidad y Finanzas (Informe N.° 000271-2026-OCF-OGA/INEN)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import OK, PRIMARY, SECONDARY, STRETCH, WARN, fmt_df, interp, mes, num, pct, soles, soles_short, style_fig

LIQ_FILE = Path(__file__).parent / "data" / "liquidaciones.csv"
# Cifras del informe
INFORME = {"SALUDPOL": 25649352.86, "ONCOSALUD": 5961294.68, "FOSMAR (Marina de Guerra)": 395075.61,
           "FOSPEME (Ejército)": 247710.43, "FOSFAP (Fuerza Aérea)": 74863.06}
INFORME_TOTAL = 32328296.64
CON_DETALLE = ["SALUDPOL", "FOSPEME (Ejército)", "FOSMAR (Marina de Guerra)", "FOSFAP (Fuerza Aérea)", "ONCOSALUD"]
TODAS = "Las 5 IAFAS"


@st.cache_data
def load_liq() -> pd.DataFrame:
    d = pd.read_csv(LIQ_FILE, parse_dates=["fecha"])
    d["mes"] = d["fecha"].dt.to_period("M").dt.to_timestamp()
    d["persona"] = ~d["cliente"].fillna("").str.upper().str.contains("SALUDPOL|IAFAS|ONCOSALUD")
    return d


def _per(s: pd.Series, freq: str) -> pd.Series:
    return s.resample(freq).sum()


def _lbl(ix: pd.DatetimeIndex, freq: str) -> list[str]:
    if freq == "YS":
        return [str(t.year) for t in ix]
    if freq == "QS":
        return [f"T{(t.month - 1) // 3 + 1}-{t.year}" for t in ix]
    return [mes(t) for t in ix]


def panel_liquidaciones(fact: pd.DataFrame, opts, filtros_activos: bool) -> None:
    liq = load_liq()
    lo, hi = pd.Timestamp(opts.date_lo), pd.Timestamp(opts.date_hi)
    st.markdown(
        "Contrasta la **producción** (consumo del INEN valorizado a tarifa y atribuido al mes de atención) con las **liquidaciones** "
        "(montos cancelados por la IAFAS, registrados por Tesorería en su fecha de cancelación). Las liquidaciones llegan meses después de la atención, "
        "así que comparar un mismo año distorsiona: la lectura más fiel es la curva acumulada y el saldo por liquidar.")
    if filtros_activos:
        st.info("Esta pestaña usa solo el rango de meses. Los filtros de tipo de consumo, ámbito, diagnóstico y demografía no se aplican, "
                "porque las liquidaciones no vienen desagregadas.")

    c1, c2 = st.columns([1.4, 1])
    sel = c1.selectbox("IAFAS", [TODAS] + CON_DETALLE, key="liq_iafas")
    frq = c2.radio("Periodo del gráfico de barras", ["Anual", "Trimestral", "Mensual"], horizontal=True, key="liq_frq")
    freq = {"Anual": "YS", "Trimestral": "QS", "Mensual": "MS"}[frq]
    names = CON_DETALLE if sel == TODAS else [sel]

    prod = fact[fact["iafas"].astype(str).isin(names) & (fact["tiempo"] >= lo) & (fact["tiempo"] <= hi)]
    lq = liq[liq["iafas"].isin(names) & (liq["mes"] >= lo) & (liq["mes"] <= hi)]
    fuera = liq[liq["iafas"].isin(names) & (liq["mes"] > hi)]["monto"].sum()
    if prod.empty:
        st.warning("Sin producción en el rango elegido.")
        return
    idx = pd.date_range(lo, hi, freq="MS")
    pm = prod.groupby("tiempo")["gasto"].sum().reindex(idx, fill_value=0.0)
    lm = lq.groupby("mes")["monto"].sum().reindex(idx, fill_value=0.0)
    # la serie arranca cuando hay producción o liquidación
    first = min([x for x in (pm[pm > 0].index.min(), lm[lm > 0].index.min()) if pd.notna(x)])
    pm, lm = pm[pm.index >= first], lm[lm.index >= first]
    P_, L_ = float(pm.sum()), float(lm.sum())
    saldo = P_ - L_
    media12 = float(pm.iloc[-12:].mean()) if len(pm) else 0.0

    k = st.columns(5)
    k[0].metric("Producción acumulada", soles_short(P_), help="Consumo valorizado del INEN en el rango elegido.")
    k[1].metric("Liquidado acumulado", soles_short(L_), help="Monto cancelado con fecha de cancelación dentro del rango elegido.")
    k[2].metric("Saldo por liquidar", soles_short(saldo), help="Producción menos liquidado. Negativo si lo liquidado supera lo producido en el rango (pagos de atenciones anteriores).")
    k[3].metric("% liquidado", pct(L_ / P_ * 100 if P_ else None), help="Liquidado acumulado / producción acumulada.")
    k[4].metric("Saldo en meses de producción", num(saldo / media12, 1) if (media12 and saldo > 0) else "n/d",
                help="Saldo por liquidar dividido entre la producción mensual promedio de los últimos 12 meses.")

    desde = mes(first)
    txt = (f"Entre {desde} y {mes(hi)}, {sel if sel != TODAS else 'las 5 IAFAS'} generó {soles_short(P_)} de producción y se liquidaron {soles_short(L_)} "
           f"({pct(L_ / P_ * 100 if P_ else None)}). ")
    if saldo > 0:
        txt += f"Quedan {soles_short(saldo)} sin liquidar, equivalentes a {num(saldo / media12, 1) if media12 else 'n/d'} meses de producción reciente."
    else:
        txt += "Lo liquidado supera la producción del rango: incluye pagos de atenciones anteriores al inicio de la serie."
    if fuera > 0:
        txt += f" Se excluyen {soles_short(fuera)} liquidados después de {mes(hi)}."
    st.markdown(interp(txt), unsafe_allow_html=True)

    # --- barras por periodo ---------------------------------------------------
    a, b = st.columns(2)
    with a:
        pp, ll = _per(pm, freq), _per(lm, freq)
        x = _lbl(pp.index, freq)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=pp.values, name="Producción", marker_color=PRIMARY))
        fig.add_trace(go.Bar(x=x, y=ll.values, name="Liquidado", marker_color=SECONDARY))
        style_fig(fig, 360, f"Producción y liquidado ({frq.lower()})")
        fig.update_layout(barmode="group", hovermode="x unified")
        fig.update_xaxes(type="category")
        fig.update_yaxes(tickformat=",.0f", tickprefix="S/ ")
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("Barras azules: lo producido en el periodo. Barras naranjas: lo cancelado en el mismo periodo, sin importar a qué atención corresponda. "
                           "Pueden diferir mucho porque las liquidaciones se concentran en pocas fechas."), unsafe_allow_html=True)
    with b:
        cp, cl = pm.cumsum(), lm.cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cp.index, y=cp.values, name="Producción acumulada", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=cl.index, y=cl.values, name="Liquidado acumulado", line=dict(color=SECONDARY, width=3),
                                 fill="tonexty", fillcolor="rgba(217,84,47,0.12)"))
        style_fig(fig, 360, "Curvas acumuladas")
        fig.update_yaxes(tickformat=",.0f", tickprefix="S/ ")
        fig.update_xaxes(tickformat="%b-%Y")
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("La distancia entre las dos curvas es el saldo por liquidar. Si la curva naranja sube en escalones y se acerca a la azul, "
                           "la IAFAS está poniéndose al día; si la distancia crece, la deuda se acumula."), unsafe_allow_html=True)

    # --- tabla anual ----------------------------------------------------------
    ya = pd.DataFrame({"Producción": _per(pm, "YS"), "Liquidado": _per(lm, "YS")})
    ya.index = [t.year for t in ya.index]
    ya["% liquidado en el año"] = ya["Liquidado"] / ya["Producción"].replace(0, np.nan) * 100
    ya["Saldo acumulado"] = (ya["Producción"] - ya["Liquidado"]).cumsum()
    nl = lq.groupby(lq["mes"].dt.year).agg(n=("monto", "size"), mx=("monto", "max"), pers=("persona", "mean"))
    ya["N.° de liquidaciones"] = nl["n"].reindex(ya.index).fillna(0)
    ya["Mayor liquidación"] = nl["mx"].reindex(ya.index)
    ya["% a nombre de persona"] = nl["pers"].reindex(ya.index) * 100
    ya.index.name = "Año"
    ya = ya.reset_index()
    st.markdown("##### Detalle por año")
    st.dataframe(fmt_df(ya, money=["Producción", "Liquidado", "Saldo acumulado", "Mayor liquidación"],
                        pcts=["% liquidado en el año", "% a nombre de persona"], ints=["N.° de liquidaciones"]),
                 hide_index=True, **STRETCH)
    st.caption("El último año puede ser parcial. «A nombre de persona» son liquidaciones cuyo cliente figura como persona natural y no como la IAFAS; "
               "ocurre sobre todo en SALUDPOL entre 2019 y 2021.")
    from panels import csv_button
    csv_button(ya, "Descargar detalle anual (CSV)", f"produccion_vs_liquidaciones_{sel}.csv", "liq_csv")

    # --- por IAFAS (solo en la vista conjunta) ---------------------------------
    if sel == TODAS:
        rows = []
        for nm in CON_DETALLE:
            pr = fact[(fact["iafas"].astype(str) == nm) & (fact["tiempo"] >= lo) & (fact["tiempo"] <= hi)]["gasto"].sum()
            lr = lq[lq["iafas"] == nm]["monto"].sum()
            rows.append({"IAFAS": nm, "Producción": pr, "Liquidado": lr, "% liquidado": lr / pr * 100 if pr else np.nan, "Saldo": pr - lr})
        t = pd.DataFrame(rows)
        st.markdown("##### Por IAFAS")
        st.dataframe(fmt_df(t, money=["Producción", "Liquidado", "Saldo"], pcts=["% liquidado"]), hide_index=True, **STRETCH)
        fig = go.Figure(go.Bar(x=t["IAFAS"], y=t["% liquidado"], marker_color=[OK if v >= 100 else (WARN if v >= 50 else SECONDARY) for v in t["% liquidado"].fillna(0)],
                               text=[num(v, 1) + "%" if v == v else "" for v in t["% liquidado"]], textposition="outside"))
        style_fig(fig, 300, "% liquidado de la producción acumulada")
        fig.update_yaxes(ticksuffix="%", rangemode="tozero")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("Verde: lo liquidado alcanza o supera la producción. Ámbar: entre 50% y 100%. Naranja: menos de la mitad. "
                           "Los porcentajes sobre 100% reflejan pagos de atenciones previas al rango o a la serie."), unsafe_allow_html=True)

        # curvas acumuladas por IAFAS, una al lado de otra, con el total como primer panel
        from plotly.subplots import make_subplots
        paneles = [(nm, [nm]) for nm in CON_DETALLE]
        fig = make_subplots(rows=2, cols=3, subplot_titles=[p[0] for p in paneles], horizontal_spacing=0.09, vertical_spacing=0.2)
        for j, (ttl, nms) in enumerate(paneles):
            r_, c_ = j // 3 + 1, j % 3 + 1
            pr_m = fact[fact["iafas"].astype(str).isin(nms) & (fact["tiempo"] >= lo) & (fact["tiempo"] <= hi)].groupby("tiempo")["gasto"].sum().reindex(idx, fill_value=0.0)
            lq_m = lq[lq["iafas"].isin(nms)].groupby("mes")["monto"].sum().reindex(idx, fill_value=0.0)
            f0 = [x for x in (pr_m[pr_m > 0].index.min(), lq_m[lq_m > 0].index.min()) if pd.notna(x)]
            if not f0:
                continue
            pr_m, lq_m = pr_m[pr_m.index >= min(f0)], lq_m[lq_m.index >= min(f0)]
            fig.add_trace(go.Scatter(x=pr_m.index, y=pr_m.cumsum().values, name="Producción acumulada", line=dict(color=PRIMARY, width=2.5),
                                     legendgroup="p", showlegend=(j == 0)), row=r_, col=c_)
            fig.add_trace(go.Scatter(x=lq_m.index, y=lq_m.cumsum().values, name="Liquidado acumulado", line=dict(color=SECONDARY, width=2.5),
                                     fill="tonexty", fillcolor="rgba(217,84,47,0.12)", legendgroup="l", showlegend=(j == 0)), row=r_, col=c_)
        style_fig(fig, 560, "Curvas acumuladas por IAFAS")
        fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=70, b=70))
        fig.update_yaxes(tickformat=",.2~s", tickprefix="S/ ", rangemode="tozero", gridcolor="#eceff4")
        fig.update_xaxes(tickformat="%Y", nticks=5)
        for a in fig.layout.annotations:
            a.font = dict(size=13, color=PRIMARY)
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("Cada panel tiene su propia escala vertical: sirve para comparar la forma de las curvas, no los montos entre IAFAS. "
                           "Mientras más abierta está la zona sombreada, más producción sigue sin liquidar."), unsafe_allow_html=True)

    # --- conciliación con el informe -----------------------------------------
    with st.expander("Conciliación con el Informe N.° 000271-2026-OCF-OGA/INEN"):
        rows = []
        for nm, v in INFORME.items():
            det = float(liq.loc[liq["iafas"] == nm, "monto"].sum()) if nm in CON_DETALLE else np.nan
            pr = float(fact[(fact["iafas"].astype(str) == nm)]["gasto"].sum())
            rows.append({"IAFAS": nm, "Según el informe": v, "Suma del Excel": det, "Diferencia": det - v if det == det else np.nan,
                         "Producción total de la serie": pr})
        c = pd.DataFrame(rows)
        st.dataframe(fmt_df(c, money=["Según el informe", "Suma del Excel", "Diferencia", "Producción total de la serie"]), hide_index=True, **STRETCH)
        sf = liq[liq["fecha"].isna()]
        nota = (f" {len(sf)} liquidación(es) por {soles(sf['monto'].sum(), 0)} no trae fecha de cancelación: suma en el total pero no aparece en las curvas ni en los periodos. "
                "En el Excel de ONCOSALUD repite el N.° de documento de otra liquidación del mismo monto, así que conviene revisar si es un duplicado." if len(sf) else "")
        st.caption(f"Total del informe: {soles(INFORME_TOTAL, 2)}. MEDIC SER S.A.C. (Auna Clínica Delgado) no registra liquidaciones. "
                   f"Los totales por IAFAS del Excel coinciden con el informe.{nota}")

    st.markdown(interp("<b>Cómo leerlo.</b> Una brecha entre producción y liquidado puede deberse a deuda aún no pagada, glosas u observaciones, atenciones valorizadas que no llegaron a facturarse, "
                       "pagos hechos directamente por pacientes, o desfase de periodos. Esta pestaña mide la brecha; no explica su causa. "
                       "La producción de FF.AA. empieza cuando cada IAFAS aparece en la extracción, mientras que SALUDPOL y ONCOSALUD tienen liquidaciones desde enero de 2019."), unsafe_allow_html=True)
