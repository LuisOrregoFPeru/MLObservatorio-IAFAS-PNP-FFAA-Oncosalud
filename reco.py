"""reco.py: pestaña de recomendaciones basadas en los datos y en la teoría del seguro.

Contenido
  - Resumen de recomendaciones (reglas con evidencia numérica de la IAFAS).
  - Transferencia prospectiva por paciente-mes con tarifa truncada, margen de riesgo (principio de desviación
    estándar), corredor para valores atípicos y credibilidad de límite de fluctuación con las demás IAFAS.
  - Monto mínimo de carta de garantía por diagnóstico (principio de percentil sobre el costo acumulado de la ventana).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from data_processing import SD_GROUP
from panels import NOTA_INICIO, csv_button
from stats_analysis import mann_kendall_trend
from ui import (
    BAD, OK, PRIMARY, SECONDARY, STRETCH, WARN, Opts, fmt_df, interp, pct, soles, soles_short, style_fig,
)

MIN_PM = 30  # mínimo de paciente-mes por diagnóstico para tarificar


# ---------------------------------------------------------------------------
# cálculo
# ---------------------------------------------------------------------------

def _pm(d: pd.DataFrame) -> pd.DataFrame:
    """Costo por diagnóstico, paciente y mes (solo pacientes con DNI y diagnóstico registrado)."""
    dd = d[(d["pid"] >= 0) & (d["dx_grupo"] != SD_GROUP)]
    out = dd.groupby(["dx_grupo", "pid", "tiempo"], observed=True)["gasto"].sum().reset_index()
    out["dx_grupo"] = out["dx_grupo"].astype(str)
    return out


def tarifa_table(d: pd.DataFrame, d_rest: pd.DataFrame, pct_out: float, conf: float, r_cred: float) -> pd.DataFrame:
    """Tarifa prospectiva mensual por paciente en tratamiento, por grupo de diagnóstico."""
    pm = _pm(d)
    pr = _pm(d_rest) if d_rest is not None and not d_rest.empty else pm.iloc[0:0]
    meses = max(d["tiempo"].nunique(), 1)
    years = meses / 12.0
    z = float(norm.ppf(conf))
    rest_groups = {g: x["gasto"].values for g, x in pr.groupby("dx_grupo")}
    rows = []
    for g, x in pm.groupby("dx_grupo"):
        v = x["gasto"].values
        n = len(v)
        if n < MIN_PM:
            continue
        c = float(np.quantile(v, pct_out / 100))
        vt = np.minimum(v, c)
        mu, sd = float(v.mean()), float(v.std(ddof=1))
        mut, sdt = float(vt.mean()), float(vt.std(ddof=1))
        if mut <= 0:
            continue
        N = n / years                                   # pacientes-mes por año
        lam = z * sdt / np.sqrt(max(N, 1.0)) / mut      # margen relativo por riesgo de proceso
        cvt = sdt / mut
        nF = (z / r_cred) ** 2 * (1 + cvt ** 2)         # tamaño de credibilidad plena (límite de fluctuación)
        Z = float(min(1.0, np.sqrt(n / nF)))
        vr = rest_groups.get(g)
        mu_rest = float(np.minimum(vr, c).mean()) if vr is not None and len(vr) >= MIN_PM else np.nan
        base = Z * mut + (1 - Z) * mu_rest if np.isfinite(mu_rest) else mut
        tarifa = base * (1 + lam)
        exc = float((v - vt).mean())                    # exceso esperado sobre el umbral por paciente-mes
        if N >= 120 and cvt <= 1.2:
            clase = "Transferencia prospectiva por paciente-mes"
        elif N >= 36 and cvt <= 2.0:
            clase = "Transferencia con corredor y monto agrupado"
        else:
            clase = "Mantener pago por servicio o agrupar"
        rows.append({
            "Diagnóstico": g, "Pacientes-mes": n, "Pacientes-mes por año": N, "Costo medio observado": mu, "CV observado": sd / mu if mu else np.nan,
            "Umbral atípico": c, "CV truncado": cvt, "Monto base (truncado)": mut, "Margen de riesgo": lam * 100,
            "Credibilidad Z": Z, "Media resto IAFAS (truncada)": mu_rest, "Transferencia propuesta": tarifa,
            "Exceso esperado": exc, "% gasto en atípicos": exc / mu * 100 if mu else np.nan, "Recomendación": clase,
            "Pago anual esperado": tarifa * N + exc * N,
        })
    return pd.DataFrame(rows)


def carta_table(d: pd.DataFrame, h: int, conf: float, incident_only: bool) -> pd.DataFrame:
    """Monto mínimo de carta de garantía por diagnóstico: cuantil `conf` del gasto acumulado en los primeros `h` meses."""
    dd = d[(d["pid"] >= 0) & (d["dx_grupo"] != SD_GROUP)]
    if dd.empty:
        return pd.DataFrame()
    gm = dd.groupby(["dx_grupo", "pid", "tiempo"], observed=True)["gasto"].sum().reset_index()
    gm["dx_grupo"] = gm["dx_grupo"].astype(str)
    gm["mi"] = gm["tiempo"].dt.year * 12 + gm["tiempo"].dt.month
    k = ["dx_grupo", "pid"]
    start = gm.groupby(k)["mi"].transform("min")
    gm["rel"] = gm["mi"] - start
    last_mi = int(d["tiempo"].max().year * 12 + d["tiempo"].max().month)
    first_mi = int(d["tiempo"].min().year * 12 + d["tiempo"].min().month)
    st0 = gm.groupby(k)["mi"].min()
    win = gm[gm["rel"] < h].groupby(k)["gasto"].sum()
    ok = (st0 + h - 1) <= last_mi
    if incident_only:
        ok &= st0 > first_mi
    win = win[ok.reindex(win.index).fillna(False)]
    years = max(d["tiempo"].nunique(), 1) / 12.0
    cont = gm[gm["rel"] >= 1]
    rows = []
    for g, w in win.groupby(level=0):
        v = w.values
        if len(v) < 10:
            continue
        c_ = cont.loc[cont["dx_grupo"] == g, "gasto"].values
        carta = float(np.quantile(v, conf))
        rows.append({
            "Diagnóstico": g, "Pacientes con ventana": len(v), "Costo medio de la ventana": float(v.mean()), "Mediana": float(np.median(v)),
            "P75": float(np.quantile(v, .75)), "P90": float(np.quantile(v, .90)), "P95": float(np.quantile(v, .95)),
            "Carta mínima recomendada": carta,
            "% que agota una carta igual a la media": float((v > v.mean()).mean() * 100),
            "% que agota la carta recomendada": float((v > carta).mean() * 100),
            "Renovación mensual (continuidad)": float(np.quantile(c_, conf)) if len(c_) >= 10 else np.nan,
            "Cartas nuevas por año": len(v) / years,
            "Exposición anual garantizada": len(v) / years * carta,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# recomendaciones por reglas
# ---------------------------------------------------------------------------

def build_reco(d: pd.DataFrame, d_rest: pd.DataFrame, name: str, o: Opts) -> list[tuple[str, str, str, str]]:
    out: list[tuple[str, str, str, str]] = []
    tot = d["gasto"].sum()
    if tot <= 0:
        return out
    pt = d[d["pid"] >= 0].groupby("pid")["gasto"].sum().sort_values(ascending=False)
    n_pac = len(pt)
    if n_pac:
        k10 = max(1, int(np.ceil(n_pac * .10)))
        s10 = pt.head(k10).sum() / pt.sum() * 100
        ac = pt[pt >= o.umbral_alto_costo]
        if s10 >= 50:
            out.append(("Alta", "Transferencia prospectiva anual con corredor para casos de alto costo",
                        f"El 10% de pacientes de {name} ({k10:,}) concentra {s10:,.0f}% del gasto; {len(ac):,} superan {soles_short(o.umbral_alto_costo)}.",
                        "Las tarifas ya están fijadas en el tarifario del INEN, por lo que no se propone cambiarlas sino la modalidad de pago: con el gasto anualizado de la IAFAS, "
                        "pasar a una transferencia prospectiva en cuotas mensuales, calculada sobre el consumo habitual del paciente y truncada en un percentil, y liquidar por tarifario "
                        "(o cubrir con stop-loss de la IAFAS) lo que exceda ese tope. Así el INEN deja de depender de pocos casos extremos y la IAFAS acota su gasto."))
    sd_pct = d.loc[d["dx_grupo"] == SD_GROUP, "gasto"].sum() / tot * 100
    if sd_pct >= 8:
        out.append(("Alta", "Verificar el registro del diagnóstico",
                    f"{sd_pct:,.1f}% del gasto no tiene diagnóstico CIE-10. Estos datos provienen de lo facturado y no de la carta de garantía.",
                    "Que la Unidad Funcional de Seguros verifique el registro del diagnóstico CIE-10 en la carta de garantía y en la facturación. "
                    "Sin diagnóstico no se puede tarificar por patología ni dimensionar cartas."))
    nodni = d.loc[d["pid"] < 0, "gasto"].sum() / tot * 100
    if nodni >= 2:
        out.append(("Media", "Identificar al paciente en todas las líneas",
                    f"{nodni:,.1f}% del gasto no trae DNI.", "Sin DNI no se miden pacientes ni episodios; corregir en el origen antes de pasar a pago por paciente."))
    med = d[d["tipo"] == "MEDICAMENTO"]
    if len(med) and "alto_costo" in med.columns:
        ac_m = med[med["alto_costo"].astype(str) != ""]["gasto"].sum()
        sh = ac_m / med["gasto"].sum() * 100 if med["gasto"].sum() else 0
        if sh >= 10:
            top_m = (med[med["alto_costo"].astype(str) != ""].groupby("molecula", observed=True)["gasto"].sum().sort_values(ascending=False))
            out.append(("Alta", "Pago por adelantado de medicamentos de alto costo",
                        f"La lista de alto costo es {sh:,.0f}% del gasto en medicamentos de {name}; lidera {str(top_m.index[0]).title() if len(top_m) else '—'}.",
                        "La IAFAS debe pagar por adelantado el costo del medicamento de alto costo cuando el paciente lo requiera (al aprobar el esquema o por ciclo), fuera de la "
                        "transferencia anual y con liquidación posterior contra lo dispensado. Así el INEN no financia con su caja fármacos de alto precio ni asume un riesgo que no controla."))
    proc = d[d["tipo"] == "PROCEDIMIENTO"]
    if len(proc) and "ambito" in proc.columns:
        ga = proc.groupby(proc["ambito"].astype(str))["gasto"].sum()
        gp = ga.sum()
        if gp > 0:
            a_dx = ga.get("Apoyo al diagnóstico", 0) / gp * 100
            qt = ga.get("Quimioterapia y radioterapia", 0) / gp * 100
            hosp = ga.get("Hospitalización", 0) / gp * 100
            if a_dx >= 30:
                out.append(("Media", "Paquetes de diagnóstico y estadificación con tarifa fija",
                            f"El apoyo al diagnóstico es {a_dx:,.0f}% del gasto en procedimientos.",
                            "Es un gasto de alto volumen y bastante predecible: conviene contratarlo por paquete (por ejemplo estadificación por tipo de cáncer) con tarifa prospectiva."))
            if qt >= 25:
                out.append(("Media", "Pago por ciclo o por curso de tratamiento en quimio y radioterapia",
                            f"Quimioterapia y radioterapia suman {qt:,.0f}% del gasto en procedimientos.",
                            "El curso de radioterapia (fracciones) y los ciclos de quimioterapia son unidades de pago naturales: previsibles para la IAFAS y con ingreso estable para el prestador."))
            if hosp >= 15:
                out.append(("Media", "Pago por día-cama o por grupo de casos en hospitalización",
                            f"La hospitalización es {hosp:,.0f}% del gasto en procedimientos.", "Usar una tarifa por día con ajuste por nivel de cuidado (hospitalización, UCI, TAMO) en lugar de facturar línea a línea."))
    if n_pac < 100:
        out.append(("Alta", "Tarifa con credibilidad y agrupación de riesgo",
                    f"{name} tiene solo {n_pac:,} pacientes únicos: su experiencia propia es poco creíble.",
                    "Ponderar la experiencia propia con la de las demás IAFAS (credibilidad). La tabla de transferencia prospectiva ya aplica esa ponderación y reduce el riesgo de fijar montos sobre ruido."))
    mm = d.groupby("tiempo")["gasto"].sum()
    if len(mm) >= 24:
        mk = mann_kendall_trend(mm.iloc[:-1].values, o.alpha)
        if mk is not None and mk.trend != "sin tendencia":
            ch = mk.sen_slope * 12 / mm.mean() * 100
            out.append(("Media", "Cláusula de reajuste anual de la tarifa",
                        f"El gasto mensual tiene tendencia {mk.trend} (p = {mk.p_value:.3f}), unos {ch:+.0f}% anual sobre la media.",
                        "Pactar actualización anual de tarifas con la experiencia reciente y un corredor simétrico (si el costo real se aleja de la tarifa más de cierto porcentaje, se comparte la diferencia)."))
    pr = {"Alta": 0, "Media": 1, "Baja": 2}
    return sorted(out, key=lambda r: pr[r[0]])


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------

def panel_recomendaciones(d: pd.DataFrame, d_rest: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    st.markdown("#### Recomendaciones para la negociación con " + name)
    recs = build_reco(d, d_rest, name, o)
    col = {"Alta": BAD, "Media": WARN, "Baja": OK}
    if not recs:
        st.info("Con los filtros actuales no se activó ninguna regla de recomendación.")
    for pr_, tit, evid, acc in recs:
        st.markdown(
            f'<div class="reco-card" style="border-left:5px solid {col[pr_]}"><div class="reco-head"><b>{tit}</b>'
            f'<span class="reco-tag" style="background:{col[pr_]}22;color:{col[pr_]}">Prioridad {pr_.lower()}</span></div>'
            f'<div><b>Evidencia.</b> {evid}</div><div><b>Propuesta.</b> {acc}</div></div>', unsafe_allow_html=True)

    # ------------------------------------------------------------ pago prospectivo
    st.markdown("#### A. Transferencia prospectiva por paciente en tratamiento")
    with st.expander("¿Qué es una transferencia prospectiva truncada?"):
        st.markdown(
            "- **Prospectiva:** el monto se calcula por adelantado, con el consumo histórico anualizado de la IAFAS, y se transfiere en cuotas mensuales, en lugar de pagar después línea por línea.\n"
            "- **Truncada:** al calcular el monto base, el gasto de cada paciente-mes se recorta en un percentil (por ejemplo P95). Los casos por encima de ese tope no entran al monto base, "
            "de modo que este no depende de pocos pacientes extremos. Lo que excede el tope se liquida aparte, según el tarifario del INEN, o lo cubre un stop-loss de la IAFAS.\n"
            "- **Margen de riesgo:** recargo por la variabilidad del consumo; baja cuando hay más volumen de pacientes.\n"
            "- Las tarifas unitarias siguen siendo las del tarifario INEN; cambia solo la modalidad de pago.")
    c = st.columns(3)
    pct_out = c[0].slider("Umbral de valor atípico (percentil)", 80, 99, 95, key=f"{name}_rc_out",
                          help="Lo que cada paciente-mes cueste por encima de este percentil se reconoce aparte por servicio, fuera de la transferencia.")
    conf_pct = c[1].select_slider("Confianza del margen de riesgo", options=[80, 90, 95, 99], value=95, key=f"{name}_rc_conf",
                                  help="Probabilidad de que el costo real anual no supere el pago esperado más el margen.")
    r_cred = c[2].select_slider("Precisión de credibilidad", options=[0.05, 0.10, 0.15], value=0.10, key=f"{name}_rc_r",
                                format_func=lambda x: f"±{int(x * 100)}%", help="Error relativo tolerado al estimar el costo medio. Más estricto exige más datos propios.")
    sin_ac = st.checkbox("Excluir de la transferencia los medicamentos de alto costo (se pagan por adelantado)", value=True, key=f"{name}_rc_ac")
    if sin_ac and "alto_costo" in d.columns:
        d_t = d[d["alto_costo"].astype(str) == ""]
        d_rt = d_rest[d_rest["alto_costo"].astype(str) == ""] if d_rest is not None and not d_rest.empty else d_rest
    else:
        d_t, d_rt = d, d_rest
    tt = tarifa_table(d_t, d_rt, pct_out, conf_pct / 100, r_cred)
    if tt.empty:
        st.info(f"Ningún diagnóstico tiene al menos {MIN_PM} pacientes-mes con los filtros actuales.")
    else:
        tt = tt.sort_values("Pago anual esperado", ascending=False).reset_index(drop=True)
        top = tt.head(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top["Diagnóstico"][::-1], x=top["Transferencia propuesta"][::-1], orientation="h", name="Transferencia propuesta por paciente-mes",
                             marker_color=PRIMARY))
        xr = np.maximum.reduce([top["Transferencia propuesta"].values, top["Umbral atípico"].values, top["Costo medio observado"].values])[::-1]
        fig.add_trace(go.Scatter(y=top["Diagnóstico"][::-1], x=xr * 1.03, mode="text", text=[soles_short(v) for v in top["Transferencia propuesta"][::-1]],
                                 textposition="middle right", showlegend=False, hoverinfo="skip", cliponaxis=False))
        fig.add_trace(go.Scatter(y=top["Diagnóstico"][::-1], x=top["Costo medio observado"][::-1], mode="markers", name="Costo medio observado (sin truncar)",
                                 marker=dict(symbol="diamond", size=9, color=SECONDARY)))
        fig.add_trace(go.Scatter(y=top["Diagnóstico"][::-1], x=top["Umbral atípico"][::-1], mode="markers", name=f"Umbral atípico (P{pct_out})",
                                 marker=dict(symbol="line-ns", size=14, color="#555", line=dict(width=2, color="#555"))))
        style_fig(fig, max(340, 38 * len(top) + 100), None, None)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=90, t=18, b=80))
        fig.update_xaxes(range=[0, float(max(top["Transferencia propuesta"].max(), top["Umbral atípico"].max())) * 1.3], tickprefix="S/ ", tickformat=",.0f")
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, **STRETCH)
        w = tt.iloc[0]
        n_ok = int((tt["Recomendación"] == "Transferencia prospectiva por paciente-mes").sum())
        st.markdown(interp(
            f"La barra es la transferencia mensual propuesta por paciente en tratamiento: costo medio recortado en el umbral atípico, ponderado con la experiencia de las demás IAFAS según credibilidad, "
            f"más un margen por riesgo de proceso. El rombo es el costo medio real (más alto cuando hay casos extremos) y la marca gris el umbral desde el cual el INEN seguiría facturando por servicio. "
            f"Para <b>{w['Diagnóstico']}</b>, el diagnóstico de mayor pago anual esperado, la transferencia es <b>{soles_short(w['Transferencia propuesta'])}</b> por paciente-mes con un margen de <b>{w['Margen de riesgo']:,.1f}%</b> "
            f"y credibilidad {w['Credibilidad Z']:,.2f}. {n_ok} de {len(tt)} diagnósticos califican para transferencia prospectiva directa."), unsafe_allow_html=True)
        cols_show = ["Diagnóstico", "Recomendación", "Pacientes-mes por año", "Costo medio observado", "CV observado", "CV truncado", "Umbral atípico", "Monto base (truncado)",
                     "Margen de riesgo", "Credibilidad Z", "Transferencia propuesta", "% gasto en atípicos", "Pago anual esperado"]
        st.dataframe(fmt_df(tt[cols_show], money=["Costo medio observado", "Umbral atípico", "Monto base (truncado)", "Transferencia propuesta", "Pago anual esperado"],
                            ints=["Pacientes-mes por año"], decs=["CV observado", "CV truncado", "Credibilidad Z"], pcts=["Margen de riesgo", "% gasto en atípicos"]),
                     hide_index=True, height=360, **STRETCH)
        csv_button(tt, "Descargar tarifas (CSV)", f"tarifas_prospectivas_{name}.csv", f"{name}_rc_csv1")
        red = (1 - tt["CV truncado"] / tt["CV observado"]).median() * 100
        st.markdown(interp(
            f"<b>Cómo leer la tabla.</b> El CV (desviación sobre media) mide la incertidumbre: al truncar en el umbral atípico baja de forma mediana {red:,.0f}%, que es el riesgo que el prestador deja de asumir. "
            "<b>Pacientes-mes por año</b> es el volumen que respalda el monto; con más volumen el margen necesario cae con la raíz del tamaño (ley de los grandes números). "
            "<b>Pago anual esperado</b> suma la transferencia por el volumen más el exceso esperado de los casos atípicos, que se paga por servicio. "
            "<b>Recomendación:</b> transferencia prospectiva directa si hay al menos 120 pacientes-mes al año y CV truncado hasta 1.2; con corredor si hay 36 o más y CV hasta 2; si no, mantener pago por servicio o agrupar con otras IAFAS."),
            unsafe_allow_html=True)
        st.markdown(interp(
            "<b>Beneficio para el prestador.</b> (1) Ingreso previsible por paciente en tratamiento en vez de depender de cada línea facturada. (2) El margen de riesgo cubre la variabilidad esperada con la confianza elegida. "
            "(3) Los casos atípicos y los medicamentos de alto costo salen de la transferencia y se reconocen aparte, por lo que no asume pérdidas por eventos que no controla. (4) La credibilidad evita que una racha de pocos pacientes fije la tarifa de todo el año."),
            unsafe_allow_html=True)

        # ---- monto anual
        st.markdown("##### Transferencia anual propuesta")
        l12 = d[d["tiempo"] > d["tiempo"].max() - pd.DateOffset(months=12)]
        fact12 = l12["gasto"].sum()
        ac12 = l12.loc[l12["alto_costo"].astype(str) != "", "gasto"].sum() if "alto_costo" in l12.columns else 0.0
        sindni12 = l12.loc[l12["pid"] < 0, "gasto"].sum()
        base_anual = float((tt["Transferencia propuesta"] * tt["Pacientes-mes por año"]).sum())
        exc_anual = float((tt["Exceso esperado"] * tt["Pacientes-mes por año"]).sum())
        m = st.columns(5)
        m[0].metric("Facturado últimos 12 meses", soles_short(fact12))
        m[1].metric("Transferencia anual propuesta", soles_short(base_anual), help="Σ transferencia por paciente-mes × volumen anual de pacientes-mes, por diagnóstico.")
        m[2].metric("Cuota mensual", soles_short(base_anual / 12))
        m[3].metric("Excedentes a liquidar aparte", soles_short(exc_anual), help="Gasto esperado por encima del tope truncado.")
        m[4].metric("Alto costo, pago por adelantado", soles_short(ac12) if sin_ac else "Incluido", help="Gasto en medicamentos de alto costo de los últimos 12 meses.")
        st.markdown(interp(
            f"Con el volumen anual medio de pacientes-mes, la transferencia propuesta para {name} es <b>{soles_short(base_anual)}</b> al año (<b>{soles_short(base_anual / 12)}</b> por mes), "
            f"más <b>{soles_short(exc_anual)}</b> esperados de casos atípicos que se liquidan aparte"
            + (f" y <b>{soles_short(ac12)}</b> de medicamentos de alto costo pagados por adelantado" if sin_ac else "")
            + f". Frente a los {soles_short(fact12)} facturados en los últimos 12 meses, la suma de estos componentes indica si la modalidad prospectiva reproduce el ingreso actual del INEN. "
            f"Quedan fuera {soles_short(sindni12)} sin DNI y el gasto sin diagnóstico, que se mantienen por servicio hasta corregir el registro."),
            unsafe_allow_html=True)

    # ------------------------------------------------------------ cartas de garantía
    st.markdown("#### B. Monto mínimo de carta de garantía")
    c = st.columns(3)
    h = c[0].select_slider("Vigencia de la carta (meses)", options=[1, 2, 3, 6, 12], value=3, key=f"{name}_cg_h")
    conf_c = c[1].select_slider("Cobertura buscada", options=[75, 80, 90, 95], value=90, key=f"{name}_cg_c", format_func=lambda x: f"{x}% de pacientes",
                                help="Porcentaje de pacientes cuyo gasto de la vigencia queda cubierto sin ampliar la carta.")
    st.caption("Los montos se estiman con el gasto facturado de cada paciente; no provienen de las cartas de garantía emitidas.")
    inc = c[2].checkbox("Solo pacientes con inicio observado", value=False, key=f"{name}_cg_inc",
                        help="Ventana desde el primer mes real de tratamiento. Si se desmarca, se toma el primer mes visible en los datos aunque el paciente ya viniera en tratamiento.")
    with st.expander("¿Qué significa «Solo pacientes con inicio observado»?"):
        st.markdown(NOTA_INICIO.split("** ", 1)[1])
    ct = carta_table(d, h, conf_c / 100, inc)
    if ct.empty:
        st.info("No hay diagnósticos con al menos 10 pacientes con la ventana completa observada. Reduce la vigencia o quita filtros.")
    else:
        ct = ct.sort_values("Exposición anual garantizada", ascending=False).reset_index(drop=True)
        top = ct.head(12)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top["Diagnóstico"][::-1], x=top["Carta mínima recomendada"][::-1], orientation="h", name=f"Carta mínima (cubre {conf_c}%)", marker_color=PRIMARY,
                             text=[soles_short(v) for v in top["Carta mínima recomendada"][::-1]], textposition="outside", cliponaxis=False))
        fig.add_trace(go.Scatter(y=top["Diagnóstico"][::-1], x=top["Costo medio de la ventana"][::-1], mode="markers", name="Costo medio de la ventana",
                                 marker=dict(symbol="diamond", size=9, color=SECONDARY)))
        style_fig(fig, max(340, 38 * len(top) + 100), None, None)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=90, t=18, b=80))
        fig.update_xaxes(range=[0, float(top["Carta mínima recomendada"].max()) * 1.2], tickprefix="S/ ", tickformat=",.0f")
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, **STRETCH)
        w = ct.iloc[0]
        st.markdown(interp(
            f"Una carta de garantía de {h} mes(es) dimensionada con el <b>percentil {conf_c}</b> del gasto real de los pacientes cubre a {conf_c}% sin ampliaciones. "
            f"Para <b>{w['Diagnóstico']}</b> eso es <b>{soles_short(w['Carta mínima recomendada'])}</b> frente a un costo medio de {soles_short(w['Costo medio de la ventana'])}. "
            f"Una carta igual al promedio se agotaría en {w['% que agota una carta igual a la media']:,.0f}% de los pacientes, es decir, el INEN atendería sin respaldo y tendría que reprocesar ampliaciones."),
            unsafe_allow_html=True)
        cols_show = ["Diagnóstico", "Pacientes con ventana", "Costo medio de la ventana", "Mediana", "P90", "P95", "Carta mínima recomendada",
                     "% que agota una carta igual a la media", "Renovación mensual (continuidad)", "Cartas nuevas por año", "Exposición anual garantizada"]
        st.dataframe(fmt_df(ct[cols_show], money=["Costo medio de la ventana", "Mediana", "P90", "P95", "Carta mínima recomendada", "Renovación mensual (continuidad)", "Exposición anual garantizada"],
                            ints=["Pacientes con ventana"], pcts=["% que agota una carta igual a la media"], decs=["Cartas nuevas por año"]), hide_index=True, height=340, **STRETCH)
        csv_button(ct, "Descargar cartas de garantía (CSV)", f"cartas_garantia_{name}.csv", f"{name}_cg_csv")
        st.markdown(interp(
            "<b>Cómo leer la tabla.</b> La ventana es el gasto acumulado de cada paciente en los primeros meses de su tratamiento. <b>Carta mínima</b> es el percentil elegido de esa distribución (principio de percentil de la teoría del seguro). "
            "<b>Renovación mensual</b> es el percentil equivalente del gasto mensual de pacientes que continúan, útil para cartas de continuidad. <b>Exposición anual garantizada</b> es cartas nuevas por año por monto mínimo: el compromiso de pago que la IAFAS asume. "
            "<b>Beneficio para el prestador:</b> la carta cubre casi todo el costo esperado, de modo que el INEN no financia la atención con su caja ni queda expuesto a rechazos por exceso de monto; la IAFAS, a su vez, evita ampliaciones sucesivas porque el compromiso es calculado y predecible."),
            unsafe_allow_html=True)

    with st.expander("Fundamento actuarial y límites de este análisis"):
        st.markdown(
            """
**Principios de prima.** El monto base es el valor esperado del costo recortado en el umbral atípico. El margen de riesgo usa el principio de desviación estándar:
margen = z × σ / √N, donde N es el volumen anual de paciente-mes y z sale de la confianza elegida (distribución normal). A mayor volumen, menor margen.

**Corredor de riesgo.** Lo que supera el percentil elegido se paga aparte (exceso de pérdida por paciente-mes). Es el mismo mecanismo del reaseguro stop-loss y protege al prestador de los casos extremos.

**Credibilidad.** Se usa credibilidad de límite de fluctuación: Z = mín(1, √(n / nF)) con nF = (z / r)² (1 + CV²). El monto se pondera la experiencia propia con la de las demás IAFAS: Z × propia + (1 − Z) × resto.

**Carta de garantía.** Se calcula el gasto acumulado de cada paciente en la vigencia y se toma el percentil buscado por diagnóstico (principio de percentil). Con el percentil, la carta rara vez se agota; con el promedio se agotaría en cerca de 4 de cada 10 pacientes.

**Límites.** Solo incluye lo facturado por el INEN con DNI y diagnóstico registrados; los costos son los facturados, no los costos de producción del prestador; la observación tiene truncamiento al inicio y al final; se supone independencia entre pacientes-mes (la correlación entre meses de un mismo paciente subestima el riesgo real). Valida con costos propios y con la casuística esperada antes de contratar.
""")
