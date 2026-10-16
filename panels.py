"""panels.py: paneles del dashboard (una función por sección de cada pestaña de IAFAS)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_processing import (
    AMBITO_DIRECTO, MACRO_ORDER, EDAD_ORDER, SD_GROUP, TIPO_LABEL, annual_metrics, month_index, monthly_metrics, patient_table, top_table,
)
from stats_analysis import (
    smooth_series, abc_classification, cagr, chow_test_event, detect_breakpoint_auto, linear_trend, mann_kendall_trend,
    project_series, significance_label,
)
from ui import (
    STRETCH, BAD, MACRO_COLOR, OK, PALETTE, PRIMARY, SECONDARY, TIPO_COLOR, WARN, Opts, fmt_df, hex_to_rgba, interp, mes, num, pct,
    shades_of, soles, soles_short, style_fig, time_figure, warn,
)

METRICS = {
    "Gasto (S/)": "gasto",
    "Pacientes activos": "pacientes",
    "Líneas facturadas": "lineas",
    "Costo por paciente activo (S/)": "costo_paciente",
}
FREQ_LABEL = {"Mensual": "MS", "Trimestral": "QS", "Anual": "YS"}
_PER = {"MS": "M", "QS": "Q", "YS": "Y"}
MES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
COVID_LO, COVID_HI = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-01")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def series_metric(d: pd.DataFrame, metric: str, freq: str = "MS", idx: pd.DatetimeIndex | None = None) -> pd.Series:
    """Serie de la métrica elegida a frecuencia mensual / trimestral / anual (índice completo, ceros en periodos sin datos)."""
    if d.empty:
        return pd.Series(dtype=float, name=metric)
    key = d["tiempo"].dt.to_period(_PER[freq]).dt.to_timestamp()
    if metric == "gasto":
        s = d.groupby(key)["gasto"].sum()
    elif metric == "lineas":
        s = d.groupby(key)["lineas"].sum()
    else:
        dd = d[d["pid"] >= 0]
        pac = dd.groupby(dd["tiempo"].dt.to_period(_PER[freq]).dt.to_timestamp())["pid"].nunique()
        if metric == "pacientes":
            s = pac
        else:
            g = d.groupby(key)["gasto"].sum()
            s = (g / pac.reindex(g.index)).where(pac.reindex(g.index) > 0)
    lo = idx[0] if idx is not None else s.index.min()
    hi = idx[-1] if idx is not None else s.index.max()
    full = pd.date_range(pd.Timestamp(lo).to_period(_PER[freq]).to_timestamp(), hi, freq=freq)
    s = s.reindex(full)
    if metric != "costo_paciente":
        s = s.fillna(0.0)
    s.name = metric
    s.index.name = "tiempo"
    return s


def metric_money(metric: str) -> bool:
    return metric in ("gasto", "costo_paciente")


def breakdown_wide(d: pd.DataFrame, by: str, metric: str, freq: str, idx, top: int = 8) -> pd.DataFrame:
    order = d.groupby(by, observed=True)["gasto"].sum().sort_values(ascending=False).index.tolist()
    cats = order[:top]
    cols = {str(c): series_metric(d[d[by] == c], metric, freq, idx) for c in cats}
    wide = pd.DataFrame(cols)
    if len(order) > top and metric in ("gasto", "lineas"):
        rest = series_metric(d[~d[by].isin(cats)], metric, freq, idx)
        wide["Otros"] = rest
    return wide


def complete_years(idx: pd.DatetimeIndex) -> list[int]:
    out = []
    for y in sorted(set(idx.year)):
        if ((idx.year == y).sum() == 12):
            out.append(int(y))
    return out


def small_sample_note(d: pd.DataFrame, idx: pd.DatetimeIndex, name: str) -> None:
    n_pac = d.loc[d["pid"] >= 0, "pid"].nunique()
    n_m = d["tiempo"].nunique()
    if n_pac < 100 or n_m < 24:
        st.markdown(warn(
            f"<b>Muestra pequeña en {name}:</b> {n_pac:,} pacientes únicos y {n_m} meses con consumo en la selección. "
            "Los promedios y las tendencias son inestables y los tests estadísticos tienen poca potencia; "
            "interprétalos como descripción, no como inferencia."), unsafe_allow_html=True)


def csv_button(df: pd.DataFrame, label: str, fname: str, key: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=fname, mime="text/csv", key=key)


def drop_covid(s: pd.Series, flag: bool) -> pd.Series:
    if not flag:
        return s
    return s[~((s.index >= COVID_LO) & (s.index <= COVID_HI))]


# ---------------------------------------------------------------------------
# 1. Resumen y peso del INEN en el ingreso de la IAFAS
# ---------------------------------------------------------------------------

def prima_mensual(primas: pd.DataFrame, iafas: str, idx: pd.DatetimeIndex) -> pd.Series | None:
    p = primas[primas["iafas"] == iafas].dropna(subset=["prima_anual"])
    if p.empty:
        return None
    by_year = p.groupby("anio")["prima_anual"].sum() / 12.0
    s = pd.Series([by_year.get(t.year, np.nan) for t in idx], index=idx)
    return s if s.notna().any() else None


def afiliados_mensual(afil: pd.DataFrame, iafas: str, idx: pd.DatetimeIndex) -> pd.Series | None:
    """Afiliados por mes: valor anual repetido; los años posteriores al último dato usan el último disponible."""
    p = afil[afil["iafas"] == iafas].dropna(subset=["afiliados"]).groupby("anio")["afiliados"].sum()
    if p.empty:
        return None
    yrs = sorted(set(idx.year))
    full = p.reindex(range(int(min(yrs[0], p.index.min())), int(max(yrs[-1], p.index.max())) + 1)).ffill()
    s = pd.Series([full.get(t.year, np.nan) for t in idx], index=idx)
    return s if s.notna().any() else None


NOTA_COSTOS = (
    "<b>S/ promedio por paciente</b> es el gasto total que el INEN facturó por cada paciente con ese diagnóstico en todo el periodo observado, promediado entre pacientes. "
    "Depende de cuánto tiempo estuvo cada paciente en tratamiento: quien consumió 2 meses y quien consumió 20 entran con su gasto total. "
    "<b>S/ por paciente-año</b> es el gasto por mes de consumo multiplicado por 12: lo que costaría mantener a un paciente en tratamiento durante un año completo. "
    "Quita el efecto de la duración y por eso sirve para comparar diagnósticos e IAFAS y para proyectar presupuesto anual. "
    "Ejemplo: si un diagnóstico cuesta S/ 3 000 al mes de consumo y los pacientes consumen en promedio 4 meses, el promedio por paciente es S/ 12 000 y el costo por paciente-año es S/ 36 000.")

NOTA_INICIO = ("**¿Qué significa «Solo pacientes con inicio observado»?** Un paciente tiene «inicio observado» cuando su primer mes con consumo en la IAFAS "
               "es posterior al primer mes de la serie (con el rango de fechas elegido). Ejemplo: si la serie parte en ene-2019, un paciente cuyo primer consumo es "
               "may-2021 tiene inicio observado, porque vemos su tratamiento desde el comienzo; uno que ya consume en ene-2019 no lo tiene, porque probablemente empezó antes "
               "y no vemos esos meses previos, así que su costo sale subestimado. Al activar la opción se descartan los segundos y el promedio se acerca a un costo por episodio. "
               "Sigue siendo incompleto para quienes aún están en tratamiento al cierre de los datos, y quien reaparece tras una pausa larga se cuenta con su primer consumo visible.")


def nice_ticks(vmax: float, n: int = 4) -> list[float]:
    """Marcas redondas (1, 2, 2,5, 5 x 10^k) desde 0 hasta cerca de vmax."""
    if not vmax or vmax <= 0 or vmax != vmax:
        return [0.0]
    raw = vmax / n
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [float(x) for x in np.arange(0, vmax + step * .01, step)]


def split_axes(fig: go.Figure, vmax_bar: float, vmax_pt: float, money_fmt=soles, low: float = 1.0) -> None:
    """Barras (eje izquierdo) en la mitad inferior y puntos o línea (eje derecho) en la superior."""
    tb, tp = nice_ticks(vmax_bar), nice_ticks(vmax_pt)
    top_b = max(tb[-1], vmax_bar)
    top_p = max(tp[-1], vmax_pt)
    fig.update_yaxes(range=[0, top_b * 2.1], tickvals=tb, selector=dict(title_text="Pacientes activos"))
    fig.update_yaxes(range=[0, top_b * 2.1], tickvals=tb, selector=dict(title_text="Pacientes"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title_text="", range=[-top_p * low, top_p * 1.25],
                                  tickvals=tp, ticktext=[money_fmt(v, 0) for v in tp]))


def dx_structure_chart(d: pd.DataFrame, col: str, topn: int, key: str, titulo: str) -> None:
    """Barras 100% apiladas: cómo se distribuye por dentro el gasto de cada diagnóstico."""
    if d.empty:
        st.info("Sin datos en la selección.")
        return
    vista = st.radio("Distribuir el gasto de cada diagnóstico por", ["Tipo de consumo", "Ámbito de atención"], horizontal=True, key=f"{key}_vista")
    top = d.groupby(col, observed=True)["gasto"].sum().sort_values(ascending=False).head(topn).index.tolist()
    dd = d[d[col].isin(top)]
    if vista.startswith("Tipo"):
        cat, order, colors, lab = "tipo", ["MEDICAMENTO", "INSUMO", "PROCEDIMIENTO"], TIPO_COLOR, TIPO_LABEL
    else:
        cat, order, colors, lab = "ambito_macro", MACRO_ORDER, MACRO_COLOR, {}
    m = dd.groupby([col, cat], observed=True)["gasto"].sum().unstack(fill_value=0).reindex(top)
    m = m.reindex(columns=[c_ for c_ in order if c_ in m.columns]).fillna(0)
    pctm = m.div(m.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig = go.Figure()
    for c_ in pctm.columns:
        fig.add_trace(go.Bar(y=[str(x)[:55] for x in pctm.index][::-1], x=pctm[c_][::-1], name=lab.get(c_, c_), orientation="h",
                             marker_color=colors.get(c_, "#999"),
                             text=[f"{v:,.1f}%" if v >= 6 else "" for v in pctm[c_][::-1]], textposition="inside",
                             customdata=m[c_][::-1], hovertemplate="%{y}<br>" + lab.get(c_, c_) + ": %{x:,.1f}% (S/ %{customdata:,.0f})<extra></extra>"))
    fig.update_layout(barmode="stack")
    style_fig(fig, max(320, 34 * len(pctm) + 110), titulo)
    fig.update_xaxes(ticksuffix="%", range=[0, 100])
    fig.update_yaxes(automargin=True)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, **STRETCH)
    dom = pctm.idxmax(axis=1)
    ej = pctm.index[0]
    st.markdown(interp(
        f"Cada barra es el 100% del gasto de un diagnóstico y sus colores muestran en qué se gasta por dentro. Por ejemplo, en <b>{ej}</b> "
        f"{lab.get(dom.iloc[0], dom.iloc[0]).lower()} es el componente principal ({pct(pctm.loc[ej].max())}). "
        "Diagnósticos con mayoría de medicamentos dependen de terapias sistémicas; los de mayoría de procedimientos, de radioterapia, cirugía o apoyo al diagnóstico."),
        unsafe_allow_html=True)


def bump_chart(d: pd.DataFrame, col: str, key: str) -> None:
    """Posición (ranking) de los diagnósticos año a año."""
    st.markdown("#### ¿Cómo se han movido los diagnósticos en el ranking?")
    b1, b2, b3 = st.columns([1, 1, 1.4])
    met = b1.radio("Ranking por", ["Gasto", "Pacientes"], horizontal=True, key=f"{key}_bm")
    nb = b2.slider("Diagnósticos a seguir", 3, 12, 8, key=f"{key}_bn")
    inc = b3.checkbox("Incluir el último año aunque esté incompleto", value=False, key=f"{key}_binc")
    dx = d[d["dx_grupo"] != SD_GROUP] if col == "dx_grupo" else d
    if dx.empty:
        return
    last = dx["tiempo"].max()
    yrs = sorted(dx["anio"].unique())
    if not inc and last.month < 12 and len(yrs) > 1:
        yrs = [y for y in yrs if y != last.year]
    if len(yrs) < 2:
        st.info("Se necesitan al menos dos años completos para el ranking.")
        return
    dx = dx[dx["anio"].isin(yrs)]
    if met == "Gasto":
        v = dx.groupby(["anio", col], observed=True)["gasto"].sum().unstack(0)
    else:
        v = dx[dx.pid >= 0].groupby(["anio", col], observed=True)["pid"].nunique().unstack(0)
    v = v.reindex(columns=yrs).fillna(0)
    rk = v.rank(ascending=False, method="first")
    rk = rk.where(v > 0)
    keep = v.sum(axis=1).sort_values(ascending=False).head(nb).index
    rk = rk.loc[keep]
    fig = go.Figure()
    for i, nm in enumerate(rk.index):
        y = rk.loc[nm]
        fig.add_trace(go.Scatter(x=[str(a) for a in y.index], y=y.values, mode="lines+markers+text", name=str(nm)[:40],
                                 line=dict(color=PALETTE[i % len(PALETTE)], width=3), marker=dict(size=20, color=PALETTE[i % len(PALETTE)]),
                                 text=[f"{int(r)}" if r == r else "" for r in y.values], textfont=dict(color="white", size=11),
                                 textposition="middle center", connectgaps=False,
                                 customdata=v.loc[nm].values,
                                 hovertemplate="%{x}<br>" + str(nm)[:40] + "<br>Posición %{y}<br>"
                                 + ("S/ %{customdata:,.0f}" if met == "Gasto" else "%{customdata:,.0f} pacientes") + "<extra></extra>"))
    style_fig(fig, max(380, 44 * min(int(rk.max().max()), 14) + 120), None, "Posición en el ranking")
    fig.update_yaxes(autorange=False, dtick=1, range=[float(rk.max().max()) + .6, .4], automargin=True)
    fig.update_xaxes(type="category")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, **STRETCH)
    ult, pri = yrs[-1], yrs[0]
    mov = (rk[pri] - rk[ult]).dropna()
    txt = ""
    if len(mov):
        up, dn = mov.idxmax(), mov.idxmin()
        txt = (f" Entre {pri} y {ult}, <b>{up}</b> {'subió' if mov[up] > 0 else 'se mantuvo'} {abs(int(mov[up]))} posiciones"
               + (f" y <b>{dn}</b> {'bajó' if mov[dn] < 0 else 'se mantuvo'} {abs(int(mov[dn]))}." if dn != up else "."))
    st.markdown(interp(
        f"Cada línea sigue la posición, entre todos los diagnósticos, de uno de los principales del periodo por {met.lower()} (1 es el mayor). Una línea que sube indica que gana peso; una que baja, que lo pierde."
        + txt + " Se excluye «Sin diagnóstico registrado»; no es un diagnóstico."), unsafe_allow_html=True)


def panel_siniestralidad(d: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    """Resumen de gasto, frecuencia y severidad, y peso del INEN en el ingreso de la IAFAS."""
    mm = monthly_metrics(d, idx)
    pt = patient_table(d)
    n_pac = len(pt)
    gasto = d["gasto"].sum()
    pac_mes = mm["pacientes"].sum()

    c = st.columns(5)
    c[0].metric("Gasto total (INEN)", soles_short(gasto))
    c[1].metric("Pacientes únicos", f"{n_pac:,}")
    c[2].metric("Costo por paciente", soles_short(gasto / n_pac) if n_pac else "—",
                help="Gasto total / pacientes únicos del periodo.")
    c[3].metric("Frecuencia: pacientes-mes (media)", f"{mm['pacientes'].mean():,.1f}",
                help="Promedio mensual de pacientes con al menos una línea facturada. Es una frecuencia absoluta.")
    c[4].metric("Severidad: S/ por paciente-mes", soles_short(gasto / pac_mes) if pac_mes else "—",
                help="Gasto total / suma de pacientes activos mensuales.")
    txt = (f"{name} acumuló <b>{soles_short(gasto)}</b> de gasto en el INEN con <b>{n_pac:,}</b> pacientes distintos, es decir "
           f"<b>{soles_short(gasto / n_pac) if n_pac else '—'}</b> por paciente en todo el periodo. En promedio hay "
           f"<b>{mm['pacientes'].mean():,.0f}</b> pacientes activos por mes (frecuencia) y cada paciente-mes cuesta "
           f"<b>{soles_short(gasto / pac_mes) if pac_mes else '—'}</b> (severidad). El gasto sube cuando aumentan los pacientes, "
           "cuando sube la severidad o ambas cosas; el gráfico de descomposición de abajo separa los dos efectos.")
    st.markdown(interp(txt), unsafe_allow_html=True)

    af = afiliados_mensual(o.afiliados, name, idx)
    if af is not None:
        afm = af.reindex(mm.index)
        ok_ = afm.notna() & (afm > 0)
        tasa = (mm["pacientes"] / afm * 1000).where(ok_)
        pmpm = (mm["gasto"] / afm).where(ok_)
        ka = st.columns(4)
        ka[0].metric("Afiliados (último dato)", f"{afm.dropna().iloc[-1]:,.0f}")
        ka[1].metric("Tasa de utilización (pac. por 1,000 afiliados)", f"{tasa.mean():,.2f}",
                     help="Promedio mensual de pacientes activos en el INEN (atendidos) por cada 1,000 afiliados de la IAFAS (toda su población beneficiaria, atendida o no).")
        ka[2].metric("Costo INEN por afiliado-mes (PMPM)", soles(pmpm.mean(), 2),
                     help="Gasto facturado por el INEN dividido entre afiliados y meses. Equivale a la prima pura que el INEN representa para la IAFAS.")
        ka[3].metric("Costo INEN por afiliado-año", soles(pmpm.mean() * 12, 2))
        st.markdown(interp(
            f"De cada 1,000 afiliados de {name}, en promedio <b>{tasa.mean():,.2f}</b> están en atención activa en el INEN cada mes, y cada afiliado, tenga o no consumo, "
            f"representa <b>{soles(pmpm.mean(), 2)} al mes</b> de gasto en este hospital ({soles(pmpm.mean() * 12, 0)} al año). "
            "Esa es la lectura de riesgo: costo por afiliado = tasa de utilización × severidad. Los afiliados son los del registro SUSALUD (hasta 2023, luego se repite el último dato), "
            "y no todos los afiliados son población oncológica, por eso las cifras por afiliado son pequeñas."), unsafe_allow_html=True)

    if n_pac:
        top10 = pt.head(max(1, int(np.ceil(n_pac * 0.10))))["gasto"].sum() / pt["gasto"].sum() * 100
        ac = pt[pt["gasto"] >= o.umbral_alto_costo]
        pct_ac = ac["gasto"].sum() / pt["gasto"].sum() * 100 if len(pt) else 0
        c2 = st.columns(5)
        c2[0].metric("Gasto medio mensual", soles_short(mm["gasto"].mean()), help="Gasto total de la IAFAS promediado entre los meses de la serie. No es por paciente: para eso mira el costo por paciente activo.")
        pk = mm["gasto"].idxmax()
        c2[1].metric("Mes de mayor gasto", f"{MES_ES[pk.month - 1]}-{pk.year}", soles_short(mm["gasto"].max()), delta_color="off")
        c2[2].metric("Top 10% pacientes: % gasto", pct(top10), help="Porcentaje del gasto explicado por el 10% de pacientes con mayor gasto.")
        c2[3].metric(f"Pacientes ≥ {soles_short(o.umbral_alto_costo)}", f"{len(ac):,}",
                     help="Pacientes con gasto acumulado en el periodo igual o mayor al umbral de alto costo (editable en la barra lateral).")
        c2[4].metric("Gasto de esos pacientes", pct(pct_ac))
        conc = "muy concentrado" if top10 >= 60 else ("moderadamente concentrado" if top10 >= 40 else "repartido")
        ratio = mm["gasto"].max() / mm["gasto"].median() if mm["gasto"].median() > 0 else np.nan
        pico = (f" El mes de mayor gasto ({MES_ES[pk.month - 1]}-{pk.year}) fue {ratio:,.1f} veces la mediana mensual: revisa si lo explica un solo ítem o paciente."
                if ratio == ratio and ratio >= 2 else "")
        st.markdown(interp(
            f"El gasto está <b>{conc}</b>: el 10% de pacientes con mayor gasto concentra <b>{pct(top10)}</b> del total. "
            f"{len(ac):,} pacientes superan el umbral de alto costo ({soles_short(o.umbral_alto_costo)}) y explican <b>{pct(pct_ac)}</b> del gasto."
            + pico), unsafe_allow_html=True)

    st.caption("**Paciente activo:** paciente distinto (DNI seudonimizado) con al menos una línea facturada a la IAFAS en ese mes. "
               "No mide pacientes en tratamiento continuo: quien no consume un mes no cuenta ese mes. Las líneas sin DNI no entran en este conteo.")
    st.markdown("##### Gasto mensual y pacientes activos")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mm.index, y=mm["pacientes"], name="Pacientes activos", marker_color=hex_to_rgba(PRIMARY, .75)))
    fig.add_trace(go.Scatter(x=mm.index, y=mm["gasto"], name="Gasto (S/)", yaxis="y2",
                             line=dict(color="#C0392B", width=2.5), mode="lines+markers" if o.markers else "lines"))
    style_fig(fig, 380, None, "Pacientes activos")
    split_axes(fig, float(mm["pacientes"].max() or 1), float(mm["gasto"].max() or 1))
    fig.update_layout(yaxis2_title_text="Gasto (S/)")
    st.plotly_chart(fig, **STRETCH)
    corr = np.corrcoef(mm["gasto"], mm["pacientes"])[0, 1] if len(mm) > 3 and mm["pacientes"].std() > 0 and mm["gasto"].std() > 0 else np.nan
    st.markdown(interp(
        "Las barras azules (parte inferior, eje izquierdo) son los pacientes activos de cada mes y la línea roja (parte superior, eje derecho) el gasto. "
        + (f"Ambas series se mueven {'juntas' if corr >= 0.6 else ('de forma parcial' if corr >= 0.3 else 'casi independientes')} (correlación {corr:,.2f}): "
           f"{'el gasto sigue sobre todo al número de pacientes.' if corr >= 0.6 else 'el gasto depende también de cuánto cuesta cada paciente, no solo de cuántos hay.'}" if corr == corr else "")),
        unsafe_allow_html=True)

    # ---- descomposición frecuencia x severidad
    st.markdown("##### ¿Por qué cambia el gasto: más pacientes o pacientes más caros?")
    sv = mm["costo_paciente"]
    if af is not None and ok_.sum() >= 12:
        fq = tasa
        t1 = "Frecuencia: tasa de utilización (pacientes activos por 1,000 afiliados)"
    else:
        fq = mm["pacientes"]
        t1 = "Frecuencia absoluta: pacientes activos por mes (sin denominador de afiliados)"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.1,
                        subplot_titles=(t1, "Severidad: gasto por paciente activo (S/)"))
    fig.add_trace(go.Scatter(x=fq.index, y=fq.values, name="Frecuencia", line=dict(color=OK, width=2),
                             mode="lines+markers" if o.markers else "lines", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=sv.index, y=sv.values, name="S/ por paciente activo", line=dict(color=SECONDARY, width=2),
                             mode="lines+markers" if o.markers else "lines", showlegend=False), row=2, col=1)
    style_fig(fig, 440, None)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=20))
    fig.update_yaxes(rangemode="tozero", title_text="")
    fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f", row=2, col=1)
    st.plotly_chart(fig, **STRETCH)
    ult, prev = mm.iloc[-12:], mm.iloc[-24:-12]
    txt = ("Es el esquema actuarial del riesgo: el costo esperado es <b>frecuencia × severidad</b>. La frecuencia es cuántos pacientes consumen"
           + (" por cada 1,000 afiliados de la IAFAS, es decir, de toda su población beneficiaria y no solo los atendidos (tasa de utilización)" if fq is not mm["pacientes"] else " (conteo absoluto, porque no hay afiliados cargados)")
           + " y la severidad cuánto cuesta cada paciente activo en el mes. ")
    if len(prev) == 12 and prev["gasto"].sum() > 0 and prev["pacientes"].sum() > 0 and ult["pacientes"].sum() > 0 and ult["gasto"].sum() > 0:
        d_tot = np.log(ult["gasto"].sum() / prev["gasto"].sum())
        d_fr = np.log(ult["pacientes"].sum() / prev["pacientes"].sum())
        d_sv = d_tot - d_fr
        txt += f"En los últimos 12 meses el gasto varió <b>{(np.exp(d_tot) - 1) * 100:+.1f}%</b> frente a los 12 previos"
        if af is not None and af.iloc[-12:].mean() > 0 and af.iloc[-24:-12].mean() > 0:
            d_ex = np.log(af.iloc[-12:].mean() / af.iloc[-24:-12].mean())
            d_ut = d_fr - d_ex
            txt += (f": exposición (afiliados) {(np.exp(d_ex) - 1) * 100:+.1f}%, tasa de utilización {(np.exp(d_ut) - 1) * 100:+.1f}% y severidad {(np.exp(d_sv) - 1) * 100:+.1f}%. ")
            main = max([("la exposición", abs(d_ex)), ("la utilización", abs(d_ut)), ("la severidad", abs(d_sv))], key=lambda x: x[1])[0]
        else:
            txt += f": pacientes-mes {(np.exp(d_fr) - 1) * 100:+.1f}% y severidad {(np.exp(d_sv) - 1) * 100:+.1f}%. "
            main = "la frecuencia" if abs(d_fr) >= abs(d_sv) else "la severidad"
        txt += f"El mayor componente del cambio es {main}."
    st.markdown(interp(txt), unsafe_allow_html=True)

    a, b, cch = st.columns(3)
    with a:
        if n_pac >= 2:
            st.markdown("##### ¿Qué parte del gasto explican pocos pacientes?")
            tot_g = pt["gasto"].sum()
            rows = []
            for p in (1, 5, 10, 20, 50):
                k_ = max(1, int(np.ceil(n_pac * p / 100)))
                rows.append((f"El {p}% con más gasto", k_, pt.head(k_)["gasto"].sum() / tot_g * 100, p))
            rows = pd.DataFrame(rows, columns=["g", "n", "pct", "p"]).drop_duplicates("n")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=rows["g"], x=rows["pct"], orientation="h", marker_color=PRIMARY, name="% del gasto que explican",
                                 text=[f"{v:,.0f}% ({n:,} pac.)" for v, n in zip(rows["pct"], rows["n"])], textposition="outside", cliponaxis=False))
            fig.add_trace(go.Scatter(y=rows["g"], x=rows["p"], mode="markers", name="Si el gasto fuera parejo",
                                     marker=dict(symbol="line-ns", size=18, color="#999", line=dict(width=3, color="#999"))))
            style_fig(fig, 320, None)
            fig.update_layout(hovermode="closest", margin=dict(l=10, r=70, t=18, b=60))
            fig.update_xaxes(range=[0, 118], ticksuffix="%")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, **STRETCH)
            r1, r2 = rows.iloc[0], rows[rows["p"] == 10].iloc[0] if (rows["p"] == 10).any() else rows.iloc[-1]
            st.markdown(interp(
                f"Cada barra dice qué porcentaje del gasto explica ese grupo de pacientes; la marca gris es lo que explicaría si todos costaran lo mismo. "
                f"El {int(r2['p'])}% con más gasto ({int(r2['n']):,} pacientes) explica <b>{r2['pct']:,.0f}%</b>: "
                f"{'el gasto está en pocos pacientes de muy alto costo.' if r2['pct'] >= 60 else 'la concentración es moderada.'}"), unsafe_allow_html=True)
    with b:
        if n_pac >= 2:
            st.markdown("##### Distribución del gasto por paciente")
            fig = go.Figure(go.Histogram(x=np.log10(pt["gasto"].clip(lower=1)), nbinsx=30, marker_color=PRIMARY))
            ticks = [0, 1, 2, 3, 4, 5, 6]
            style_fig(fig, 320, None, "Pacientes")
            fig.update_xaxes(tickvals=ticks, ticktext=[f"{10 ** t:,.0f}" for t in ticks], title_text="Gasto por paciente (S/, escala log)")
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, **STRETCH)
            st.markdown(interp(
                f"Cada barra agrupa pacientes con un gasto similar (escala logarítmica). La mediana es <b>{soles_short(pt['gasto'].median())}</b> y la media "
                f"<b>{soles_short(pt['gasto'].mean())}</b>; cuando la media supera a la mediana hay una minoría de pacientes muy costosos que la arrastra."),
                unsafe_allow_html=True)
    with cch:
        st.markdown("##### Composición del gasto por tipo")
        t = d.groupby("tipo", observed=True)["gasto"].sum()
        fig = go.Figure(go.Pie(labels=[TIPO_LABEL.get(i, i) for i in t.index], values=t.values, hole=.55,
                               marker=dict(colors=[TIPO_COLOR.get(i, "#999") for i in t.index]), texttemplate="%{percent:.1%}", textposition="inside", insidetextorientation="horizontal", sort=False))
        style_fig(fig, 320, None)
        fig.update_layout(showlegend=True, legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.05), hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        tt = (t / t.sum() * 100).sort_values(ascending=False)
        st.markdown(interp(f"El mayor componente es <b>{TIPO_LABEL.get(tt.index[0], tt.index[0]).lower()}</b> ({pct(tt.iloc[0])} del gasto), "
                            f"seguido de {TIPO_LABEL.get(tt.index[1], tt.index[1]).lower()} ({pct(tt.iloc[1])})." if len(tt) > 1 else "Solo hay un tipo de consumo en la selección."),
                    unsafe_allow_html=True)

    s1, s2 = st.columns(2)
    with s1:
        st.markdown("##### Pacientes y costo por sexo")
        om_sx = st.checkbox("Omitir «Sin dato» y recalcular los %", value=False, key=f"{name}_sx_sd")
        d_sx = d[d["sexo"] != "Sin dato"] if om_sx else d
        g = d_sx.groupby("sexo", observed=True)["gasto"].sum()
        p = d_sx[d_sx.pid >= 0].groupby("sexo", observed=True)["pid"].nunique()
        t = pd.concat([g.rename("gasto"), p.rename("pacientes")], axis=1).fillna(0).reset_index()
        t["pct"] = t["pacientes"] / t["pacientes"].sum() * 100
        t["sxp"] = np.where(t["pacientes"] > 0, t["gasto"] / t["pacientes"].replace(0, np.nan), np.nan)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t["sexo"], y=t["pacientes"], name="Pacientes", marker_color=PRIMARY,
                             text=[f"{n:,.0f}<br>{q:,.1f}%" for n, q in zip(t["pacientes"], t["pct"])], textposition="outside", cliponaxis=False,
                             constraintext="none", textfont=dict(size=11)))
        fig.add_trace(go.Scatter(x=t["sexo"], y=t["sxp"], name="S/ por paciente", yaxis="y2", mode="markers+text", marker=dict(color=SECONDARY, size=11, symbol="diamond"),
                                 text=[soles_short(v).replace("S/ ", "") for v in t["sxp"]], textposition="top center", textfont=dict(size=10)))
        style_fig(fig, 320, None, "Pacientes")
        fig.update_layout(hovermode="closest")
        split_axes(fig, float(t["pacientes"].max()), float(np.nanmax(t["sxp"])) if t["sxp"].notna().any() else 1.0, low=2.0)
        st.plotly_chart(fig, **STRETCH)
        tc = t[t["sexo"] != "Sin dato"].sort_values("sxp", ascending=False)
        sd = t[t["sexo"] == "Sin dato"]
        st.markdown(interp(
            "Las barras son los pacientes (con su % del total) y los rombos rojos el gasto promedio por paciente de cada grupo. "
            + (f"{tc.iloc[0]['sexo']} tiene el mayor costo por paciente ({soles_short(tc.iloc[0]['sxp'])}). " if len(tc) else "")
            + (f"«Sin dato» ({sd.iloc[0]['pct']:,.1f}% de los pacientes) agrupa registros sin sexo en el origen." if len(sd) and sd.iloc[0]["pct"] >= 1 else "")),
            unsafe_allow_html=True)
    with s2:
        st.markdown("##### Pacientes y costo por grupo etario")
        om_ed = st.checkbox("Omitir «Sin dato» y recalcular los %", value=False, key=f"{name}_ed_sd")
        d_ed = d[d["edad_grupo"] != "Sin dato"] if om_ed else d
        dd = d_ed[d_ed.pid >= 0]
        p = dd.groupby("edad_grupo", observed=True)["pid"].nunique().reindex(EDAD_ORDER).dropna()
        gg = d_ed.groupby("edad_grupo", observed=True)["gasto"].sum().reindex(p.index)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=p.index, y=p.values, name="Pacientes", marker_color=PRIMARY,
                             text=[f"{n:,.0f}<br>{n / p.sum() * 100:,.1f}%" for n in p.values], textposition="outside", cliponaxis=False,
                             constraintext="none", textfont=dict(size=10)))
        fig.add_trace(go.Scatter(x=p.index, y=(gg / p).values, name="S/ por paciente", yaxis="y2", mode="markers+text", marker=dict(color=SECONDARY, size=11, symbol="diamond"),
                                 text=[soles_short(v).replace("S/ ", "") for v in (gg / p).values], textposition="top center", textfont=dict(size=10)))
        style_fig(fig, 320, None, "Pacientes")
        fig.update_layout(hovermode="closest")
        split_axes(fig, float(p.max()), float(np.nanmax((gg / p).values)) if len(p) else 1.0, low=2.0)
        st.plotly_chart(fig, **STRETCH)
        cpp = (gg / p).dropna()
        cpp2 = cpp.drop("Sin dato", errors="ignore")
        st.markdown(interp(
            "Las barras son los pacientes de cada grupo de edad (con su % del total) y los rombos rojos su gasto promedio. "
            + (f"El grupo con más pacientes es <b>{p.idxmax()}</b> años ({p.max() / p.sum() * 100:,.0f}%) y el de mayor costo por paciente es <b>{cpp2.idxmax()}</b> ({soles_short(cpp2.max())})." if len(cpp2) else "")),
            unsafe_allow_html=True)

    with st.expander(f"Pacientes de mayor gasto (identificador seudonimizado)"):
        top = pt.head(25).copy()
        top["Paciente"] = ["P-" + format(int(x) % 16 ** 6, "06x") for x in top["pid"]]
        top["% del gasto"] = top["gasto"] / pt["gasto"].sum() * 100
        show = top[["Paciente", "gasto", "% del gasto", "meses_activos", "lineas", "primer_mes", "ultimo_mes"]].rename(columns={
            "gasto": "Gasto", "meses_activos": "Meses activos", "lineas": "Líneas", "primer_mes": "Primer mes", "ultimo_mes": "Último mes"})
        st.dataframe(fmt_df(show, money=["Gasto"], pcts=["% del gasto"], ints=["Meses activos", "Líneas"], dates=["Primer mes", "Último mes"]),
                     hide_index=True, **STRETCH)
        st.markdown(interp(f"Los 25 pacientes de mayor gasto suman <b>{top['% del gasto'].sum():,.1f}%</b> del gasto total de {name}. "
                           "Sirve para revisar casos individuales de alto costo; el identificador es un seudónimo y no permite reconocer a la persona."),
                    unsafe_allow_html=True)

    # peso del INEN en el ingreso de la IAFAS
    pm = prima_mensual(o.primas, name, idx)
    st.markdown("#### Peso del INEN en el ingreso de la IAFAS")
    if pm is None:
        st.markdown(interp(
            f"<b>Falta el ingreso de {name}.</b> La extracción trae el gasto facturado por el INEN, pero no los ingresos de la IAFAS. "
            "Ingresa su prima o ingreso anual en la barra lateral (<i>Ingreso de la IAFAS</i>) y esta sección calcula qué porcentaje de ese "
            "ingreso se factura en el INEN, mes a mes y por año. Mientras tanto, abajo tienes los indicadores de frecuencia y severidad."),
            unsafe_allow_html=True)
    else:
        peso = (mm["gasto"] / pm * 100).where(pm > 0)
        peso_a = mm.groupby(mm.index.year)["gasto"].sum() / pm.groupby(pm.index.year).sum().replace(0, np.nan) * 100
        peso_a = peso_a.dropna()
        figs = go.Figure()
        figs.add_trace(go.Scatter(x=peso.index, y=peso.values, name="Peso mensual", mode="lines+markers" if o.markers else "lines",
                                  line=dict(color=PRIMARY, width=2)))
        roll = (mm["gasto"].rolling(12, min_periods=12).sum() / pm.rolling(12, min_periods=12).sum() * 100)
        figs.add_trace(go.Scatter(x=roll.index, y=roll.values, name="Móvil 12 meses", line=dict(color=SECONDARY, width=2.5)))
        style_fig(figs, 360, "Peso del INEN en el ingreso (gasto INEN / ingreso mensual prorrateado)", "% del ingreso")
        figs.update_yaxes(ticksuffix="%", rangemode="tozero")
        st.plotly_chart(figs, **STRETCH)
        cols = st.columns(len(peso_a) or 1)
        for cc, (y, v) in zip(cols, peso_a.items()):
            cc.metric(f"{int(y)}", f"{v:,.1f}%")
        if len(peso_a):
            y_last, v_last = int(peso_a.index[-1]), float(peso_a.iloc[-1])
            cmp_txt = ""
            if len(peso_a) >= 2:
                v_prev = float(peso_a.iloc[-2])
                cmp_txt = (f" Frente a {int(peso_a.index[-2])} ({v_prev:,.1f}%) el peso {'aumentó' if v_last > v_prev else 'disminuyó'} "
                           f"{abs(v_last - v_prev):,.1f} puntos porcentuales.")
            st.markdown(interp(
                f"En <b>{y_last}</b> el INEN representó <b>{v_last:,.1f}%</b> del ingreso de {name}: de cada S/ 100 que recibe la IAFAS, "
                f"S/ {v_last:,.1f} se facturan en este hospital.{cmp_txt} Un peso creciente indica que el INEN absorbe una parte cada vez "
                "mayor del ingreso. No es la siniestralidad total de la IAFAS, porque no incluye lo que paga en otros prestadores."),
                unsafe_allow_html=True)
        st.caption("El ingreso anual se reparte en 12 partes iguales; solo hay cálculo en los años con ingreso registrado.")


def dx_cost_table(d: pd.DataFrame, col: str, solo_nuevos: bool = False) -> pd.DataFrame:
    """Costo por paciente según diagnóstico: una fila por diagnóstico con media, mediana, P90 y costo anualizado.
    El gasto de un paciente en un diagnóstico es la suma de sus líneas rotuladas con ese diagnóstico."""
    dd = d[(d["pid"] >= 0) & (d["dx_grupo"] != "Sin diagnóstico registrado")]
    if solo_nuevos and not d.empty:
        first = d[d["pid"] >= 0].groupby("pid")["tiempo"].min()
        dd = dd[dd["pid"].isin(first[first > d["tiempo"].min()].index)]
    if dd.empty:
        return pd.DataFrame()
    g = dd.groupby([col, "pid"], observed=True).agg(gasto=("gasto", "sum"), meses=("tiempo", "nunique"))
    a = g.groupby(level=0, observed=True).agg(
        pacientes=("gasto", "size"), media=("gasto", "mean"), mediana=("gasto", "median"),
        p90=("gasto", lambda x: x.quantile(.9)), meses_medios=("meses", "mean"), gasto=("gasto", "sum"), meses_tot=("meses", "sum"))
    a["costo_mes"] = a["gasto"] / a["meses_tot"].replace(0, np.nan)
    a["costo_anual"] = a["costo_mes"] * 12
    return a.reset_index()



# ---------------------------------------------------------------------------
# 2. Diagnósticos
# ---------------------------------------------------------------------------

def panel_diagnosticos(d: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts, d_rest: pd.DataFrame | None = None) -> None:
    c = st.columns([1.1, 1.1, 1, 1])
    nivel = c[0].radio("Nivel", ["Grupo (sitio tumoral)", "Código CIE-10"], horizontal=True, key=f"{name}_dx_nivel")
    metr = c[1].selectbox("Ordenar por", ["Gasto", "Pacientes", "Líneas facturadas", "Gasto por paciente"], key=f"{name}_dx_metr")
    topn = c[2].slider("Mostrar top", 5, 30, 12, key=f"{name}_dx_top")
    solo_onco = c[3].checkbox("Omitir «Sin diagnóstico registrado»", value=False, key=f"{name}_dx_onco",
                              help="Quita ese grupo y recalcula los porcentajes sobre el gasto con diagnóstico.")
    col = "dx_grupo" if nivel.startswith("Grupo") else "dx_etiqueta"
    dd = d[d["dx_grupo"] != "Sin diagnóstico registrado"] if solo_onco else d
    t = top_table(dd, col)
    if t.empty:
        st.info("Sin datos.")
        return
    key = {"Gasto": "gasto", "Pacientes": "pacientes", "Líneas facturadas": "lineas", "Gasto por paciente": "gasto_paciente"}[metr]
    t = t.sort_values(key, ascending=False).reset_index(drop=True)
    top = t.head(topn)

    left, right = st.columns([3, 2])
    with left:
        lab = {"gasto": "Gasto (S/)", "pacientes": "Pacientes", "lineas": "Líneas", "gasto_paciente": "S/ por paciente"}[key]
        fig = go.Figure(go.Bar(
            y=top[col][::-1], x=top[key][::-1], orientation="h", marker_color=shades_of(PRIMARY, len(top))[::-1][::-1],
            text=[(soles_short(v) if key in ("gasto", "gasto_paciente") else f"{v:,.0f}") for v in top[key][::-1]], textposition="outside", cliponaxis=False,
        ))
        style_fig(fig, max(380, 28 * len(top) + 80), f"Diagnósticos de mayor {lab.lower()}")
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=90, t=48, b=10))
        fig.update_xaxes(range=[0, float(top[key].max()) * 1.18])
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, **STRETCH)
        t0 = top.iloc[0]
        st.markdown(interp(f"Ranking por {lab.lower().replace('(s/)', '(S/)')}: <b>{t0[col]}</b> lidera con {soles_short(t0['gasto'])} ({pct(t0['pct_gasto'])} del gasto). "
                           f"Los {len(top)} primeros suman <b>{pct(top['pct_gasto'].sum())}</b> del gasto de {name}."), unsafe_allow_html=True)
    with right:
        tot_g = t["gasto"].sum()
        fig = go.Figure(go.Treemap(
            labels=top[col], parents=[""] * len(top), values=top["gasto"].clip(lower=0.01),
            marker=dict(colors=shades_of(PRIMARY, len(top), .35, .8)), textinfo="label+percent root",
        ))
        style_fig(fig, max(380, 28 * len(top) + 80), "Peso en el gasto (treemap)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("El área de cada rectángulo es proporcional al gasto del diagnóstico: sirve para ver de un vistazo cuáles dominan y cuáles son marginales."),
                    unsafe_allow_html=True)

    show = t.rename(columns={col: "Diagnóstico", "gasto": "Gasto", "pacientes": "Pacientes", "lineas": "Líneas",
                             "gasto_paciente": "S/ por paciente", "pct_gasto": "% del gasto", "pct_acum": "% acumulado"})
    st.dataframe(fmt_df(show[["Diagnóstico", "Gasto", "% del gasto", "% acumulado", "Pacientes", "Líneas", "S/ por paciente"]],
                        money=["Gasto", "S/ por paciente"], pcts=["% del gasto", "% acumulado"], ints=["Pacientes", "Líneas"]),
                 hide_index=True, **STRETCH, height=300)
    csv_button(show, "Descargar tabla de diagnósticos (CSV)", f"dx_{name}.csv", f"{name}_dx_csv")
    sd_pct = d.loc[d["dx_grupo"] == "Sin diagnóstico registrado", "gasto"].sum() / d["gasto"].sum() * 100 if d["gasto"].sum() else 0
    st.markdown(interp(f"La tabla ordena todos los diagnósticos por la métrica elegida e incluye el gasto por paciente, que permite distinguir los diagnósticos frecuentes de los costosos. "
                       f"{pct(sd_pct)} del gasto de {name} no tiene diagnóstico registrado y no puede atribuirse a ningún cáncer."), unsafe_allow_html=True)

    st.markdown("#### Evolución de los diagnósticos principales")
    e1, e2, e3 = st.columns(3)
    freq = e1.radio("Periodo", ["Anual", "Trimestral", "Mensual"], horizontal=True, key=f"{name}_dx_freq")
    ntop = e2.slider("Diagnósticos a graficar", 2, 10, 5, key=f"{name}_dx_nev")
    rel = e3.radio("Valores", ["Absolutos", "% del total"], horizontal=True, key=f"{name}_dx_rel")
    metric = "gasto"
    wide = breakdown_wide(dd, col, metric, FREQ_LABEL[freq], idx, top=ntop)
    if freq == "Anual":
        yrs = complete_years(idx)
        wide.index = wide.index.year
        wide.index.name = "Año"
        wide = wide.copy()
    if rel == "% del total":
        wide = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0) * 100
    fig = go.Figure()
    for i, cc in enumerate(wide.columns):
        fig.add_trace(go.Bar(x=wide.index.astype(str) if freq == "Anual" else wide.index, y=wide[cc], name=str(cc),
                             marker_color=PALETTE[i % len(PALETTE)]))
    fig.update_layout(barmode="stack")
    style_fig(fig, 400, None, "% del gasto" if rel != "Absolutos" else "S/")
    if rel == "Absolutos":
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    if freq == "Anual":
        st.caption("2026 está incompleto: compara con cuidado los totales absolutos; el modo «% del total» no se ve afectado.")
    st.plotly_chart(fig, **STRETCH)
    st.markdown(interp("Muestra cómo cambia el peso de cada diagnóstico en el tiempo. Un diagnóstico cuya franja se ensancha está ganando participación en el gasto; "
                       "en «% del total» se aprecia el cambio de composición sin el efecto del volumen total."), unsafe_allow_html=True)

    st.markdown("#### Estructura interna del gasto de cada diagnóstico")
    dx_structure_chart(dd, col, topn, f"{name}_dxs", "Cómo se distribuye el gasto dentro de cada diagnóstico (100% = gasto del diagnóstico)")
    bump_chart(dd, col, f"{name}_bump")

    # ---- costo promedio por paciente según diagnóstico
    st.markdown("#### ¿Cuánto cuesta atender a un paciente según su diagnóstico?")
    q1, q2, q3 = st.columns([1, 1, 1.4])
    minpac = q1.slider("Mínimo de pacientes", 3, 50, 10, key=f"{name}_dxc_min", help="Con pocos pacientes el promedio es inestable.")
    orden = q2.radio("Ordenar por", ["Más pacientes", "Mayor costo medio"], horizontal=True, key=f"{name}_dxc_ord")
    nuevos = q3.checkbox("Solo pacientes con inicio observado", value=False, key=f"{name}_dxc_new",
                         help="Deja solo a pacientes cuyo primer consumo es posterior al primer mes de la serie. Ver la nota debajo.")
    with st.expander("¿Qué significa «Solo pacientes con inicio observado»?"):
        st.markdown(NOTA_INICIO.split("** ", 1)[1])
    ct = dx_cost_table(d, col, nuevos)
    if ct.empty or (ct["pacientes"] >= minpac).sum() == 0:
        st.info("Ningún diagnóstico alcanza el mínimo de pacientes con los filtros actuales.")
    else:
        ct = ct[ct["pacientes"] >= minpac]
        if d_rest is not None and not d_rest.empty:
            cr = dx_cost_table(d_rest, col, nuevos)
            ct = ct.merge(cr[[col, "pacientes", "media"]].rename(columns={"pacientes": "pac_resto", "media": "media_resto"}), on=col, how="left")
        ct = ct.sort_values("pacientes" if orden.startswith("Más") else "media", ascending=False).reset_index(drop=True)
        top_c = ct.head(topn)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[str(x)[:60] for x in top_c[col]][::-1], x=top_c["media"][::-1], orientation="h", name="Promedio por paciente",
                             marker_color=hex_to_rgba(PRIMARY, .8), text=[soles_short(v) for v in top_c["media"][::-1]], textposition="outside", cliponaxis=False,
                             customdata=np.c_[top_c["pacientes"][::-1], top_c["mediana"][::-1], top_c["p90"][::-1]],
                             hovertemplate="%{y}<br>Promedio S/ %{x:,.0f}<br>%{customdata[0]:,.0f} pacientes · mediana S/ %{customdata[1]:,.0f} · P90 S/ %{customdata[2]:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(y=[str(x)[:60] for x in top_c[col]][::-1], x=top_c["mediana"][::-1], mode="markers", name="Mediana",
                                 marker=dict(symbol="diamond", size=9, color=SECONDARY), hoverinfo="skip"))
        style_fig(fig, max(360, 28 * len(top_c) + 90), None, "S/ por paciente en el periodo")
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=90, t=18, b=70))
        fig.update_xaxes(range=[0, float(top_c["media"].max()) * 1.22], title_text="S/ por paciente en el periodo")
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, **STRETCH)
        w = top_c.iloc[0]
        st.markdown(interp(
            f"Cada barra es el <b>gasto promedio facturado por el INEN a {name} por paciente</b> con ese diagnóstico en el periodo, y el rombo la mediana. "
            f"Por ejemplo, un paciente con <i>{w[col]}</i> costó en promedio <b>{soles_short(w['media'])}</b> (mediana {soles_short(w['mediana'])}, 1 de cada 10 pacientes supera {soles_short(w['p90'])}), "
            f"con {w['meses_medios']:,.1f} meses de consumo en promedio, es decir unos <b>{soles_short(w['costo_anual'])} por paciente-año</b>. "
            "Cuando la media supera claramente a la mediana, unos pocos casos muy costosos elevan el promedio."), unsafe_allow_html=True)
        cols_t = {col: "Diagnóstico", "pacientes": "Pacientes", "media": "S/ promedio por paciente", "mediana": "Mediana", "p90": "P90",
                  "meses_medios": "Meses con consumo (media)", "costo_mes": "S/ por paciente-mes", "costo_anual": "S/ por paciente-año"}
        if "media_resto" in ct.columns:
            cols_t.update({"media_resto": "Promedio resto de IAFAS", "pac_resto": "Pacientes resto"})
        tabc = ct[list(cols_t)].rename(columns=cols_t)
        money = [c_ for c_ in ["S/ promedio por paciente", "Mediana", "P90", "S/ por paciente-mes", "S/ por paciente-año", "Promedio resto de IAFAS"] if c_ in tabc.columns]
        st.dataframe(fmt_df(tabc, money=money, ints=[c_ for c_ in ["Pacientes", "Pacientes resto"] if c_ in tabc.columns], decs=["Meses con consumo (media)"]),
                     hide_index=True, height=320, **STRETCH)
        csv_button(tabc, "Descargar costo por paciente según diagnóstico (CSV)", f"costo_dx_{name}.csv", f"{name}_dxc_csv")
        st.markdown(interp(
            NOTA_COSTOS + " La columna «resto de IAFAS» permite ver si el mismo diagnóstico cuesta más o menos en otras IAFAS. "
            "<b>Limitaciones:</b> solo cuenta lo facturado por el INEN con diagnóstico registrado y DNI, y la observación está truncada (hay pacientes que empezaron antes de 2019 "
            "o siguen en tratamiento), por lo que el costo de un tratamiento completo es mayor que el observado. Usa «Solo pacientes con inicio observado» para acercarte a un costo por episodio."),
            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3-5. Medicamentos / insumos / procedimientos
# ---------------------------------------------------------------------------

def panel_items(d: pd.DataFrame, d_rest: pd.DataFrame, name: str, tipo: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    dt = d[d["tipo"] == tipo]
    lab = TIPO_LABEL[tipo]
    if dt.empty:
        st.info(f"No hay {lab.lower()} en la selección actual.")
        return
    kp = f"{name}_{tipo}"
    c = st.columns([1.2, 1.2, 1, 1.2])
    if tipo == "MEDICAMENTO":
        nivel = c[0].radio("Nivel", ["Descripción (presentación)", "Principio activo"], horizontal=True, key=f"{kp}_nivel")
        col = "descripcion" if nivel.startswith("Desc") else "principio_activo"
        subs = sorted([s for s in dt["subtipo"].dropna().unique() if s])
        sub_sel = c[1].multiselect("Tipo de medicina", subs, default=subs, key=f"{kp}_sub")
        if sub_sel:
            dt = dt[dt["subtipo"].isin(sub_sel)]
    elif tipo == "PROCEDIMIENTO":
        nivel = c[0].radio("Nivel", ["Descripción", "Grupo del tarifario"], horizontal=True, key=f"{kp}_nivel")
        col = "descripcion" if nivel.startswith("Desc") else "grupo_tarifario"
        if col == "grupo_tarifario":
            dt = dt.assign(grupo_tarifario=dt["grupo_tarifario"].astype(str).replace("", "Sin grupo en el tarifario"))
    else:
        col = "descripcion"
        c[0].markdown(f"**Nivel:** descripción del ítem")
    topn = c[2].slider("Top", 5, 40, 15, key=f"{kp}_top")
    q = c[3].text_input("Buscar en la descripción", "", key=f"{kp}_q")
    if q:
        dt = dt[dt[col].astype(str).str.contains(q, case=False, na=False, regex=False)]
    if dt.empty:
        st.info("Sin coincidencias.")
        return

    t = top_table(dt, col)
    k1 = st.columns(4)
    k1[0].metric(f"Gasto en {lab.lower()}", soles_short(t["gasto"].sum()))
    k1[1].metric("Ítems distintos", f"{len(t):,}")
    zona = abc_classification(dt, item_col=col, value_col="gasto", qty_col="cantidad")
    if zona is not None:
        zs = zona.zone_summary.set_index("Zona_ABC")
        nA = int(zs.loc["A", "n_items"]) if "A" in zs.index else 0
        k1[2].metric("Ítems en zona A (80% del gasto)", f"{nA:,}", f"{nA / len(t) * 100:,.1f}% de los ítems", delta_color="off")
    k1[3].metric("Pacientes con consumo", f"{dt.loc[dt.pid >= 0, 'pid'].nunique():,}")
    tg = t["gasto"].sum()
    top1 = t.iloc[0]
    top5 = t["gasto"].head(5).sum() / tg * 100 if tg else 0
    ztxt = ""
    if zona is not None:
        ztxt = (f" Solo <b>{nA:,}</b> ítems ({nA / len(t) * 100:,.1f}% del catálogo consumido) explican el 80% del gasto (zona A de Pareto)"
                f"{': el gasto depende de pocos ítems, ahí está la oportunidad de negociación y control' if nA / len(t) < 0.1 else ''}.")
    st.markdown(interp(
        f"{name} gastó <b>{soles_short(tg)}</b> en {lab.lower()} con <b>{len(t):,}</b> ítems distintos. El de mayor peso es "
        f"<i>{str(top1[col])[:80]}</i> ({pct(top1['pct_gasto'])} del gasto) y los cinco primeros suman <b>{pct(top5)}</b>.{ztxt}"),
        unsafe_allow_html=True)

    m1, m2 = st.columns([1.2, 1])
    with m1:
        rk_by = st.selectbox("Ordenar ranking por", ["Gasto", "Cantidad", "Pacientes", "Precio medio unitario", "Gasto por paciente"],
                             key=f"{kp}_rk")
        kmap = {"Gasto": "gasto", "Cantidad": "cantidad", "Pacientes": "pacientes", "Precio medio unitario": "precio_medio",
                "Gasto por paciente": "gasto_paciente"}
        kk = kmap[rk_by]
        tt = t.sort_values(kk, ascending=False).head(topn)
        fig = go.Figure(go.Bar(
            y=[str(x)[:70] for x in tt[col]][::-1], x=tt[kk][::-1], orientation="h",
            marker_color=shades_of(TIPO_COLOR[tipo], len(tt), .3, .75),
            text=[(soles_short(v) if kk in ("gasto", "precio_medio", "gasto_paciente") else f"{v:,.0f}") for v in tt[kk][::-1]],
            textposition="outside", cliponaxis=False, customdata=np.c_[tt[col][::-1]],
            hovertemplate="%{customdata[0]}<br>%{x:,.1f}<extra></extra>",
        ))
        style_fig(fig, max(380, 26 * len(tt) + 80), f"Mayor consumo de {lab.lower()} por {rk_by.lower()}")
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=90, t=48, b=10))
        fig.update_xaxes(range=[0, float(tt[kk].max()) * 1.2])
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp(f"Ranking de {lab.lower()} por {rk_by.lower()}. Cambia el criterio para distinguir lo que pesa por volumen (cantidad, pacientes) de lo que pesa por precio "
                           "(precio medio unitario, gasto por paciente)."), unsafe_allow_html=True)
    with m2:
        if zona is not None:
            z = zona.table.head(40)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=z["ranking"], y=z["valor_total"], name="Gasto", marker_color=hex_to_rgba(TIPO_COLOR[tipo], .8),
                                 customdata=z[col], hovertemplate="%{customdata}<br>S/ %{y:,.0f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=z["ranking"], y=z["%_acumulado"], name="% acumulado", yaxis="y2", line=dict(color=SECONDARY, width=2)))
            fig.add_hline(y=80, line_dash="dot", line_color="#999", yref="y2")
            style_fig(fig, max(380, 26 * len(tt) + 80), "Curva de Pareto (primeros 40 ítems)")
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], showgrid=False, ticksuffix="%"), hovermode="closest")
            fig.update_xaxes(title_text="Ranking")
            st.plotly_chart(fig, **STRETCH)
            zs2 = zona.zone_summary.rename(columns={"Zona_ABC": "Zona", "n_items": "Ítems", "valor": "Gasto", "%_items": "% ítems", "%_valor": "% gasto"})
            st.dataframe(fmt_df(zs2, money=["Gasto"], pcts=["% ítems", "% gasto"], ints=["Ítems"]), hide_index=True, **STRETCH)
            st.markdown(interp("La curva de Pareto ordena los ítems de mayor a menor gasto y acumula su peso (línea naranja, eje derecho). Zona A: ítems que explican el primer 80% del gasto; "
                               "B: el siguiente 15%; C: el 5% restante, normalmente muchísimos ítems de poco monto. El control se concentra en la zona A."), unsafe_allow_html=True)

    if zona is not None:
        abc_map = zona.table.set_index(col)["Zona_ABC"]
        t["ABC"] = t[col].map(abc_map)
    show = t.rename(columns={col: "Ítem", "gasto": "Gasto", "cantidad": "Cantidad", "pacientes": "Pacientes", "lineas": "Líneas",
                             "precio_medio": "Precio medio", "gasto_paciente": "S/ por paciente", "pct_gasto": "% gasto",
                             "pct_acum": "% acum.", "primer_mes": "Primer mes", "ultimo_mes": "Último mes"})
    cols_show = ["Ítem", "ABC", "Gasto", "% gasto", "% acum.", "Cantidad", "Precio medio", "Pacientes", "S/ por paciente", "Primer mes", "Último mes"]
    with st.expander("Tabla completa de ítems", expanded=False):
        st.dataframe(fmt_df(show[[c_ for c_ in cols_show if c_ in show.columns]], money=["Gasto", "S/ por paciente"],
                            pcts=["% gasto", "% acum."], ints=["Cantidad", "Pacientes"], decs=["Precio medio"], dates=["Primer mes", "Último mes"]),
                     hide_index=True, **STRETCH, height=380)
        csv_button(show, f"Descargar tabla de {lab.lower()} (CSV)", f"{tipo.lower()}_{name}.csv", f"{kp}_csv")
        st.markdown(interp("Tabla completa de ítems con su zona ABC, peso en el gasto, cantidad, precio medio, pacientes y gasto por paciente. Úsala para buscar un ítem concreto con el cuadro de búsqueda superior."),
                    unsafe_allow_html=True)

    # Perfil diferencial vs resto de IAFAS
    st.markdown(f"#### Consumo diferenciado: ¿qué {lab.lower()} pesan más en {name} que en las demás IAFAS?")
    dr = d_rest[d_rest["tipo"] == tipo]
    if tipo == "MEDICAMENTO" and sub_sel:
        dr = dr[dr["subtipo"].isin(sub_sel)]
    if q:
        dr = dr[dr[col].astype(str).str.contains(q, case=False, na=False, regex=False)]
    if dr.empty:
        st.caption("Las demás IAFAS no registran consumo comparable en esta selección.")
    else:
        a = dt.groupby(col, observed=True)["gasto"].sum()
        b = dr.groupby(col, observed=True)["gasto"].sum()
        df = pd.concat([a.rename("g_i"), b.rename("g_r")], axis=1).fillna(0.0)
        df["pct_i"] = df["g_i"] / df["g_i"].sum() * 100
        df["pct_r"] = df["g_r"] / df["g_r"].sum() * 100
        df["dif_pp"] = df["pct_i"] - df["pct_r"]
        df["indice"] = np.where(df["pct_r"] > 0, df["pct_i"] / df["pct_r"].replace(0, np.nan), np.nan)
        df["solo_esta"] = df["g_r"] == 0
        minpct = st.slider("Peso mínimo en esta IAFAS para considerar el ítem (% de su gasto)", 0.0, 2.0, 0.2, 0.05, key=f"{kp}_minpct")
        cand = df[df["pct_i"] >= minpct].sort_values("dif_pp", ascending=False)
        up, dn = cand.head(10), cand.tail(10).sort_values("dif_pp")
        l, r = st.columns(2)
        for colx, data, title, colr in ((l, up, f"Sobrerrepresentados en {name}", SECONDARY), (r, dn, f"Subrepresentados en {name}", PRIMARY)):
            with colx:
                fig = go.Figure(go.Bar(y=[str(x)[:60] for x in data.index][::-1], x=data["dif_pp"][::-1], orientation="h", marker_color=colr,
                                       customdata=np.c_[data.index.astype(str)[::-1], data["pct_i"][::-1], data["pct_r"][::-1]],
                                       hovertemplate="%{customdata[0]}<br>%{customdata[1]:.1f}% vs %{customdata[2]:.1f}% del gasto<extra></extra>"))
                style_fig(fig, 360, title)
                fig.update_layout(hovermode="closest", margin=dict(l=10, r=20, t=48, b=10))
                fig.update_xaxes(title_text="Diferencia de peso (puntos porcentuales)")
                st.plotly_chart(fig, **STRETCH)
        st.markdown(interp(f"Izquierda: ítems que pesan más en {name} que en las demás IAFAS (diferencia en puntos porcentuales del gasto); derecha: los que pesan menos. "
                           "Señalan diferencias de perfil clínico o de práctica de prescripción entre poblaciones asegurables."), unsafe_allow_html=True)
        ex = cand[cand["solo_esta"]].sort_values("g_i", ascending=False).head(15)
        if len(ex):
            with st.expander(f"Ítems que solo consume {name} (ninguna otra IAFAS los registra)"):
                e = ex[["g_i", "pct_i"]].reset_index().rename(columns={col: "Ítem", "g_i": "Gasto", "pct_i": "% del gasto"})
                st.dataframe(fmt_df(e, money=["Gasto"], pcts=["% del gasto"]), hide_index=True, **STRETCH)
        st.caption("Compara el peso de cada ítem dentro del gasto de la IAFAS con su peso en el gasto de las demás IAFAS juntas (misma selección de filtros).")

    # Evolución
    st.markdown("#### Evolución del consumo")
    e1, e2, e3, e4 = st.columns(4)
    freq = e1.radio("Periodo", ["Anual", "Trimestral", "Mensual"], horizontal=True, key=f"{kp}_ev_freq")
    metric = e2.selectbox("Métrica", ["Gasto", "Cantidad", "Precio medio unitario", "Pacientes"], key=f"{kp}_ev_m")
    default_items = t.sort_values("gasto", ascending=False)[col].astype(str).head(5).tolist()
    opts = t[col].astype(str).tolist()
    sel = st.multiselect("Ítems a graficar (por defecto, los 5 de mayor gasto)", opts, default=default_items, key=f"{kp}_ev_items")
    if sel:
        per = _PER_KEY[freq]
        dsel = dt[dt[col].astype(str).isin(sel)]
        key = dsel["tiempo"].dt.to_period(per).dt.to_timestamp()
        if metric == "Pacientes":
            ser = dsel[dsel.pid >= 0].assign(_k=key[dsel.pid >= 0]).groupby(["_k", col], observed=True)["pid"].nunique().unstack(col)
        elif metric == "Precio medio unitario":
            gg = dsel.assign(_k=key).groupby(["_k", col], observed=True).agg(g=("gasto", "sum"), q=("cantidad", "sum"))
            ser = (gg["g"] / gg["q"].replace(0, np.nan)).unstack(col)
        else:
            vcol = "gasto" if metric == "Gasto" else "cantidad"
            ser = dsel.assign(_k=key).groupby(["_k", col], observed=True)[vcol].sum().unstack(col)
        ser.columns = [str(x) for x in ser.columns]
        full = pd.date_range(idx[0].to_period(per).to_timestamp(), idx[-1], freq={"M": "MS", "Q": "QS", "Y": "YS"}[per])
        ser = ser.reindex(full)
        if metric not in ("Precio medio unitario",):
            ser = ser.fillna(0.0)
        if freq == "Anual":
            ser.index = ser.index.year.astype(str)
        short = {c_: (c_[:55] + "…" if len(c_) > 56 else c_) for c_ in ser.columns}
        ser = ser.rename(columns=short)
        fig = time_figure(ser, o, title=f"{metric} por ítem · {freq.lower()}", money=metric in ("Gasto", "Precio medio unitario"),
                          allow_lowess=False, height=420)
        if metric == "Precio medio unitario":
            fig.update_traces(connectgaps=True)
        st.plotly_chart(fig, **STRETCH)
        if freq == "Anual":
            st.caption("Los años incompletos (2026 hasta agosto, y el primer año de cada IAFAS si empezó a mitad de año) no son comparables en niveles absolutos.")
        st.markdown(interp(f"Evolución de la métrica «{metric.lower()}» para los ítems elegidos. Si el gasto sube mientras el precio medio unitario se mantiene, el aumento es de volumen; "
                           "si el precio medio sube, es de precio o de cambio de presentación."), unsafe_allow_html=True)

    # Comparación de dos años
    st.markdown("#### Cambio entre dos años")
    years = sorted(dt["anio"].unique().tolist())
    if len(years) >= 2:
        y1, y2 = st.columns(2)
        ya = y1.selectbox("Año base", years, index=max(0, len(years) - 2), key=f"{kp}_ya")
        yb = y2.selectbox("Año de comparación", years, index=len(years) - 1, key=f"{kp}_yb")
        ga = dt[dt["anio"] == ya].groupby(col, observed=True)["gasto"].sum()
        gb = dt[dt["anio"] == yb].groupby(col, observed=True)["gasto"].sum()
        cmp_ = pd.concat([ga.rename(f"Gasto {ya}"), gb.rename(f"Gasto {yb}")], axis=1).fillna(0.0)
        cmp_["Δ S/"] = cmp_[f"Gasto {yb}"] - cmp_[f"Gasto {ya}"]
        cmp_["Δ %"] = np.where(cmp_[f"Gasto {ya}"] > 0, cmp_["Δ S/"] / cmp_[f"Gasto {ya}"].replace(0, np.nan) * 100, np.nan)
        cmp_ = cmp_.reset_index().rename(columns={col: "Ítem"})
        cmp_["Ítem"] = cmp_["Ítem"].astype(str)
        la, lb = st.columns(2)
        la.markdown("**Mayor aumento**")
        la.dataframe(fmt_df(cmp_.sort_values("Δ S/", ascending=False).head(10), money=[f"Gasto {ya}", f"Gasto {yb}", "Δ S/"], pcts=["Δ %"]),
                     hide_index=True, **STRETCH)
        lb.markdown("**Mayor reducción**")
        lb.dataframe(fmt_df(cmp_.sort_values("Δ S/").head(10), money=[f"Gasto {ya}", f"Gasto {yb}", "Δ S/"], pcts=["Δ %"]),
                     hide_index=True, **STRETCH)
        if ya == 2026 or yb == 2026:
            st.caption("2026 llega hasta agosto: parte de la diferencia es solo duración del periodo.")
        top_up = cmp_.sort_values("Δ S/", ascending=False).iloc[0]
        top_dn = cmp_.sort_values("Δ S/").iloc[0]
        st.markdown(interp(f"Entre {ya} y {yb}, el mayor aumento absoluto es <i>{str(top_up['Ítem'])[:70]}</i> ({soles_short(top_up['Δ S/'])}) y la mayor reducción "
                           f"<i>{str(top_dn['Ítem'])[:70]}</i> ({soles_short(top_dn['Δ S/'])}). Un aumento grande en pocos ítems sugiere un cambio de esquema o de volumen puntual."),
                    unsafe_allow_html=True)
    if tipo == "MEDICAMENTO":
        panel_alto_costo(d, d_rest, name, idx, o)


def panel_alto_costo(d: pd.DataFrame, d_rest: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    """Subanálisis de los medicamentos de la lista nominal no PNUME de alto costo."""
    import alto_costo as AC
    st.markdown("#### Medicamentos de alto costo (lista nominal no PNUME)")
    dm = d[d["tipo"] == "MEDICAMENTO"]
    if dm.empty:
        st.info("No hay medicamentos en la selección actual.")
        return
    incl_mol = st.checkbox("Incluir otras presentaciones de la misma molécula", value=True, key=f"{name}_ac_mol",
                           help="La lista nominal identifica ítems por código. Con esta opción también se cuentan presentaciones distintas de la misma molécula "
                                "(p. ej. rituximab 500 mg cuando la lista trae rituximab 1400 mg).")
    marks = [AC.COD] + ([AC.MOL] if incl_mol else [])
    ac = dm[dm["alto_costo"].isin(marks)]
    L = AC.lista()
    if ac.empty:
        st.info(f"{name} no registra consumo de medicamentos de la lista de alto costo en la selección.")
    else:
        g_ac, g_med = ac["gasto"].sum(), dm["gasto"].sum()
        pac_ac = ac.loc[ac.pid >= 0, "pid"].nunique()
        pac_med = dm.loc[dm.pid >= 0, "pid"].nunique()
        k = st.columns(5)
        k[0].metric("Gasto en alto costo", soles_short(g_ac))
        k[1].metric("% del gasto en medicamentos", pct(g_ac / g_med * 100))
        k[2].metric("% del gasto total", pct(g_ac / d["gasto"].sum() * 100))
        k[3].metric("Pacientes tratados", f"{pac_ac:,}", f"{pac_ac / pac_med * 100:,.1f}% de los que reciben medicamentos" if pac_med else None, delta_color="off")
        k[4].metric("S/ por paciente tratado", soles_short(g_ac / pac_ac) if pac_ac else "—")
        dist = ac["molecula"].astype(str).nunique()
        st.markdown(interp(
            f"{name} gastó <b>{soles_short(g_ac)}</b> en {dist} moléculas de la lista de alto costo, que son <b>{pct(g_ac / g_med * 100)}</b> de su gasto en medicamentos y "
            f"<b>{pct(g_ac / d['gasto'].sum() * 100)}</b> de todo su gasto en el INEN. Los {pac_ac:,} pacientes tratados "
            f"({pct(pac_ac / pac_med * 100) if pac_med else '—'} de quienes reciben medicamentos) costaron <b>{soles_short(g_ac / pac_ac) if pac_ac else '—'}</b> cada uno solo en estos fármacos."
            " Un porcentaje alto con pocos pacientes indica que unos pocos tratamientos determinan el gasto en fármacos."), unsafe_allow_html=True)

        # ranking por molécula
        tm = top_table(ac.assign(molecula=ac["molecula"].astype(str).str.title()), "molecula")
        tmx = tm.head(15)
        fig = go.Figure(go.Bar(y=tmx["molecula"][::-1], x=tmx["gasto"][::-1], orientation="h", marker_color=shades_of(SECONDARY, len(tmx), .3, .75),
                               text=[soles_short(v) for v in tmx["gasto"][::-1]], textposition="outside", cliponaxis=False,
                               customdata=np.c_[tmx["pacientes"][::-1], tmx["gasto_paciente"][::-1]],
                               hovertemplate="%{y}<br>S/ %{x:,.0f}<br>%{customdata[0]:,.0f} pacientes · S/ %{customdata[1]:,.0f} por paciente<extra></extra>"))
        style_fig(fig, max(320, 26 * len(tmx) + 80), None, None)
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=90, t=18, b=40))
        fig.update_xaxes(range=[0, float(tmx["gasto"].max()) * 1.2])
        st.plotly_chart(fig, **STRETCH)
        t1 = tm.iloc[0]
        st.markdown(interp(
            f"Ranking por molécula de la lista. <b>{t1['molecula']}</b> es la de mayor gasto ({soles_short(t1['gasto'])}, {pct(t1['gasto'] / g_ac * 100)} del alto costo) "
            f"y las tres primeras suman <b>{pct(tm['gasto'].head(3).sum() / g_ac * 100)}</b>. El gasto por paciente (al pasar el cursor) muestra qué moléculas son caras por tratamiento y cuáles por volumen."),
            unsafe_allow_html=True)

        # detalle por presentación con precio de referencia
        tp = top_table(ac, "descripcion")
        cod_map = dm[["descripcion", "alto_costo"]].drop_duplicates().astype(str).set_index("descripcion")["alto_costo"]
        tp["Fuente"] = tp["descripcion"].astype(str).map(cod_map)
        pref = {}
        for desc in tp["descripcion"].astype(str):
            m = L[L["descripcion"].map(AC._norm) == AC._norm(desc)]
            pref[desc] = float(m["precio_ref"].iloc[0]) if len(m) else np.nan
        tp["precio_ref"] = tp["descripcion"].astype(str).map(pref)
        tp["precio_ref"] = tp["precio_ref"].where(tp["precio_ref"] > 1)
        tp["ratio"] = tp["precio_medio"] / tp["precio_ref"]
        show = tp.rename(columns={"descripcion": "Presentación", "gasto": "Gasto", "pct_gasto": "% del alto costo", "cantidad": "Cantidad", "pacientes": "Pacientes",
                                  "gasto_paciente": "S/ por paciente", "precio_medio": "Precio medio pagado", "precio_ref": "Precio referencial (lista)", "ratio": "Pagado / referencial"})
        show["% del alto costo"] = show["Gasto"] / g_ac * 100
        cols_s = ["Presentación", "Fuente", "Gasto", "% del alto costo", "Cantidad", "Pacientes", "S/ por paciente", "Precio medio pagado", "Precio referencial (lista)", "Pagado / referencial"]
        with st.expander("Detalle por presentación y comparación con el precio referencial", expanded=True):
            st.dataframe(fmt_df(show[cols_s], money=["Gasto", "S/ por paciente"], pcts=["% del alto costo"], ints=["Cantidad", "Pacientes"],
                                decs=["Precio medio pagado", "Precio referencial (lista)", "Pagado / referencial"]), hide_index=True, height=320, **STRETCH)
            csv_button(show[cols_s], "Descargar detalle de alto costo (CSV)", f"alto_costo_{name}.csv", f"{name}_ac_csv")
            st.markdown(interp(
                "Cada fila es una presentación de la lista de alto costo. <b>Precio medio pagado</b> es gasto dividido entre cantidad facturada; se compara con el <b>precio referencial de adjudicación</b> "
                "del listado (las filas con referencial de S/ 0.90 en el listado son marcadores sin precio histórico y se omiten). Una razón mayor a 1 indica que se factura por encima del precio adjudicado."
                " Es una referencia, porque la unidad de facturación puede no coincidir con la de adjudicación."), unsafe_allow_html=True)

        # evolución anual por molécula
        st.markdown("##### Evolución del gasto en alto costo")
        fq = st.radio("Periodo", ["Anual", "Trimestral", "Mensual"], horizontal=True, key=f"{name}_ac_f")
        top_m = tm["molecula"].head(6).tolist()
        acm = ac.assign(molecula=ac["molecula"].astype(str).str.title())
        acm["molecula"] = acm["molecula"].where(acm["molecula"].isin(top_m), "Otras de la lista")
        wide = breakdown_wide(acm, "molecula", "gasto", FREQ_LABEL[fq], idx, top=7)
        if fq == "Anual":
            wide.index = wide.index.year.astype(str)
        fig = go.Figure()
        for i, cc in enumerate(wide.columns):
            fig.add_trace(go.Bar(x=wide.index, y=wide[cc], name=str(cc), marker_color=PALETTE[i % len(PALETTE)]))
        fig.update_layout(barmode="stack")
        style_fig(fig, 360, None, "S/")
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
        st.plotly_chart(fig, **STRETCH)
        yr = acm.groupby("anio")["gasto"].sum()
        tot_yr = dm.groupby("anio")["gasto"].sum()
        sh = (yr / tot_yr * 100).dropna()
        st.markdown(interp(
            "Gasto en alto costo en el tiempo, por molécula. "
            + (f"Su peso dentro del gasto en medicamentos pasó de <b>{pct(sh.iloc[0])}</b> en {int(sh.index[0])} a <b>{pct(sh.iloc[-1])}</b> en {int(sh.index[-1])}"
               f"{' (2026 incompleto)' if int(sh.index[-1]) == 2026 else ''}. " if len(sh) >= 2 else "")
            + "La aparición de franjas nuevas marca la incorporación de una molécula; los picos aislados suelen ser uno o pocos pacientes."), unsafe_allow_html=True)

    # lista sin consumo
    cons = set(dm.loc[dm["alto_costo"] == AC.COD, "descripcion"].astype(str).map(AC._norm))
    sin = L[~L["descripcion"].map(AC._norm).isin(cons)]
    with st.expander(f"Ítems de la lista sin consumo en {name} en la selección ({len(sin)} de {len(L)})"):
        sh_ = sin[["descripcion", "precio_ref", "obs"]].rename(columns={"descripcion": "Descripción (lista)", "precio_ref": "Precio referencial", "obs": "Observación del listado"})
        st.dataframe(fmt_df(sh_, decs=["Precio referencial"]), hide_index=True, **STRETCH)
        st.markdown(interp("Son ítems de la lista nominal que {0} no consumió en el periodo y filtros elegidos (o que no figuran en la extracción con ese código). "
                           "Puede deberse a que son incorporaciones recientes, a presentaciones sin código coincidente o a que la IAFAS no los usa.".format(name)), unsafe_allow_html=True)


_PER_KEY = {"Mensual": "M", "Trimestral": "Q", "Anual": "Y"}


# ---------------------------------------------------------------------------
# Ámbito de atención
# ---------------------------------------------------------------------------

def panel_ambito(d: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    vista = st.radio("Vista", ["Todo el consumo (agrupado)", "Solo procedimientos (clasificación directa)"], horizontal=True, key=f"{name}_amb_v")
    directo = vista.startswith("Solo")
    dd = d[d["tipo"] == "PROCEDIMIENTO"] if directo else d
    if dd.empty:
        st.info("Sin datos en la selección.")
        return
    col = "ambito" if directo else "ambito_macro"
    order = [a for a in (AMBITO_DIRECTO if directo else MACRO_ORDER) if a in set(dd[col].astype(str))]
    colors = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(order)}
    if not directo:
        colors.update(MACRO_COLOR)
    g = dd.groupby(col, observed=True).agg(gasto=("gasto", "sum"), lineas=("lineas", "sum")).reindex(order).dropna(how="all")
    pac = dd[dd.pid >= 0].groupby(col, observed=True)["pid"].nunique().reindex(g.index)
    tot = g["gasto"].sum()
    g["pct"] = g["gasto"] / tot * 100
    g["pacientes"] = pac.fillna(0)

    k = st.columns(min(5, len(g)))
    for cc, (a, r) in zip(k, g.sort_values("gasto", ascending=False).head(5).iterrows()):
        cc.metric(f"{a} · {pct(r['pct'])}", soles_short(r["gasto"]), help=f"{pct(r['pct'])} del gasto de la selección")
    lead = g["gasto"].idxmax()
    if directo:
        apoyo = g["pct"].get("Apoyo al diagnóstico", 0.0)
        hosp = g["pct"].get("Hospitalización", 0.0)
        cons = g["pct"].get("Consulta externa", 0.0)
        emer = g["pct"].get("Emergencia", 0.0)
        txt = (f"En procedimientos, <b>{lead}</b> es el ámbito de mayor gasto ({pct(g.loc[lead, 'pct'])}). El apoyo al diagnóstico "
               f"(laboratorio, imágenes, patología) pesa {pct(apoyo)}, la hospitalización {pct(hosp)}, la consulta externa {pct(cons)} y la emergencia {pct(emer)}. "
               "La clasificación sale del grupo del tarifario INEN al que pertenece el código del procedimiento facturado, por lo que es directa y no depende de supuestos.")
    else:
        inf = dd[dd["tipo"] != "PROCEDIMIENTO"]["gasto"].sum() / tot * 100 if tot else 0
        txt = (f"<b>{lead}</b> concentra {pct(g.loc[lead, 'pct'])} del gasto de {name}. Los procedimientos se clasifican por el grupo del tarifario INEN de su código; "
               f"los medicamentos e insumos ({pct(inf)} del gasto) no traen ámbito en la extracción y se asignan según los procedimientos "
               "del mismo paciente en el mismo mes (hospitalización si hubo días de hospitalización o UCI, si no emergencia, si no quirúrgico, si no ambulatorio). "
               "Es una inferencia a nivel paciente-mes: un paciente hospitalizado unos días y con quimioterapia ambulatoria el mismo mes pasa todo su consumo de fármacos a hospitalización.")
    st.markdown(interp(txt), unsafe_allow_html=True)

    l, r = st.columns([2, 3])
    with l:
        fig = go.Figure(go.Pie(labels=list(g.index.astype(str)), values=g["gasto"].values, hole=.55,
                               marker=dict(colors=[colors.get(str(a), "#999") for a in g.index]), texttemplate="%{percent:.1%}", textposition="inside", insidetextorientation="horizontal", sort=False))
        style_fig(fig, 360, "Distribución del gasto por ámbito")
        fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5, yanchor="middle"))
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("Cada porción es la parte del gasto de la IAFAS que corresponde a ese ámbito de atención."), unsafe_allow_html=True)
    with r:
        frq = st.radio("Periodo", ["Anual", "Trimestral", "Mensual"], horizontal=True, key=f"{name}_amb_f")
        rel = st.radio("Valores", ["Gasto (S/)", "% del total"], horizontal=True, key=f"{name}_amb_r")
        cols = {str(a): series_metric(dd[dd[col] == a], "gasto", FREQ_LABEL[frq], idx) for a in g.index}
        wide = pd.DataFrame(cols)
        if frq == "Anual":
            wide.index = wide.index.year.astype(str)
        if rel.startswith("%"):
            wide = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0) * 100
        fig = go.Figure()
        for a in wide.columns:
            fig.add_trace(go.Bar(x=wide.index, y=wide[a], name=a, marker_color=colors.get(a, "#999")))
        fig.update_layout(barmode="stack")
        style_fig(fig, 360, "Evolución del gasto por ámbito", "%" if rel.startswith("%") else "S/")
        if not rel.startswith("%"):
            fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
        st.plotly_chart(fig, **STRETCH)
        if frq == "Anual":
            st.caption("2026 llega hasta agosto; usa «% del total» para comparar años incompletos.")
        st.markdown(interp("Cómo se reparte el gasto entre ámbitos en cada periodo. Si una franja crece en «% del total», esa forma de atención gana peso en el gasto de la IAFAS."), unsafe_allow_html=True)

    show = g.reset_index().rename(columns={col: "Ámbito", "gasto": "Gasto", "pct": "% del gasto", "lineas": "Líneas", "pacientes": "Pacientes"})
    show["S/ por paciente"] = np.where(show["Pacientes"] > 0, show["Gasto"] / show["Pacientes"].replace(0, np.nan), np.nan)
    st.dataframe(fmt_df(show[["Ámbito", "Gasto", "% del gasto", "Pacientes", "S/ por paciente", "Líneas"]],
                        money=["Gasto", "S/ por paciente"], pcts=["% del gasto"], ints=["Pacientes", "Líneas"]), hide_index=True, **STRETCH)
    csv_button(show, "Descargar ámbitos (CSV)", f"ambito_{name}.csv", f"{name}_amb_csv")
    mx = show.sort_values("S/ por paciente", ascending=False).iloc[0]
    st.markdown(interp(f"La tabla resume cada ámbito. El de mayor gasto por paciente es <b>{mx['Ámbito']}</b> ({soles_short(mx['S/ por paciente'])}). "
                       "Un paciente puede aparecer en varios ámbitos, por lo que los pacientes de las filas no se suman."), unsafe_allow_html=True)

    # grupos del tarifario
    pr = d[d["tipo"] == "PROCEDIMIENTO"]
    if not pr.empty:
        st.markdown("#### Procedimientos según los grupos del tarifario INEN")
        gt = pr.groupby("grupo_tarifario", observed=True).agg(gasto=("gasto", "sum"), lineas=("lineas", "sum"))
        gt["pacientes"] = pr[pr.pid >= 0].groupby("grupo_tarifario", observed=True)["pid"].nunique()
        amb = pr.groupby(["grupo_tarifario", "ambito"], observed=True)["gasto"].sum().reset_index().sort_values("gasto", ascending=False).drop_duplicates("grupo_tarifario").set_index("grupo_tarifario")["ambito"]
        gt["ambito"] = amb
        gt = gt.sort_values("gasto", ascending=False)
        gt["pct"] = gt["gasto"] / gt["gasto"].sum() * 100
        gt = gt.reset_index()
        gt["grupo_tarifario"] = gt["grupo_tarifario"].astype(str).replace("", "Sin grupo en el tarifario")
        ng = st.slider("Grupos a mostrar", 5, 30, 12, key=f"{name}_gt_n")
        tg_ = gt.head(ng)
        amb_col = {a: PALETTE[i % len(PALETTE)] for i, a in enumerate(AMBITO_DIRECTO)}
        fig = go.Figure(go.Bar(y=[str(x)[:60] for x in tg_["grupo_tarifario"]][::-1], x=tg_["gasto"][::-1], orientation="h",
                               marker_color=[amb_col.get(str(a), "#999") for a in tg_["ambito"]][::-1],
                               text=[f"{soles_short(v)} ({q:,.1f}%)" for v, q in zip(tg_["gasto"][::-1], tg_["pct"][::-1])],
                               textposition="outside", cliponaxis=False,
                               customdata=tg_["ambito"][::-1], hovertemplate="%{y}<br>S/ %{x:,.0f}<br>Ámbito: %{customdata}<extra></extra>"))
        for a_ in dict.fromkeys(str(x) for x in tg_["ambito"]):
            fig.add_trace(go.Bar(x=[None], y=[None], name=a_, marker_color=amb_col.get(a_, "#999"), showlegend=True))
        style_fig(fig, max(360, 28 * len(tg_) + 130), "Gasto en procedimientos por grupo del tarifario (color = ámbito asignado)")
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=130, t=48, b=10))
        fig.update_xaxes(range=[0, float(tg_["gasto"].max()) * 1.3])
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, **STRETCH)
        l1 = gt.iloc[0]
        st.markdown(interp(
            f"Agrupa los procedimientos con las filas azules del tarifario INEN. <b>{l1['grupo_tarifario']}</b> es el grupo de mayor gasto ({pct(l1['pct'])} del gasto en procedimientos). "
            "El color indica el ámbito de atención que se asignó a cada grupo."), unsafe_allow_html=True)
        sh = gt.rename(columns={"grupo_tarifario": "Grupo del tarifario", "gasto": "Gasto", "pct": "% del gasto en procedimientos", "pacientes": "Pacientes",
                                "lineas": "Líneas", "ambito": "Ámbito asignado"})
        st.dataframe(fmt_df(sh[["Grupo del tarifario", "Ámbito asignado", "Gasto", "% del gasto en procedimientos", "Pacientes", "Líneas"]],
                            money=["Gasto"], pcts=["% del gasto en procedimientos"], ints=["Pacientes", "Líneas"]), hide_index=True, height=300, **STRETCH)
        csv_button(sh, "Descargar grupos del tarifario (CSV)", f"grupos_tarifario_{name}.csv", f"{name}_gt_csv")

    # qué se consume en cada ámbito
    st.markdown("#### Qué se consume en cada ámbito")
    amb_sel = st.selectbox("Ámbito", [str(a) for a in g.index], key=f"{name}_amb_sel")
    cat = st.radio("Mostrar", ["Medicamentos", "Insumos", "Procedimientos"], horizontal=True, key=f"{name}_amb_cat") if not directo else "Procedimientos"
    tp = {"Medicamentos": "MEDICAMENTO", "Insumos": "INSUMO", "Procedimientos": "PROCEDIMIENTO"}[cat]
    sub = dd[(dd[col] == amb_sel) & (dd["tipo"] == tp)]
    if sub.empty:
        st.caption("No hay consumo de este tipo en el ámbito elegido.")
    else:
        tt = top_table(sub, "descripcion").head(15)
        fig = go.Figure(go.Bar(y=[str(x)[:70] for x in tt["descripcion"]][::-1], x=tt["gasto"][::-1], orientation="h",
                               marker_color=shades_of(TIPO_COLOR[tp], len(tt), .3, .75),
                               text=[soles_short(v) for v in tt["gasto"][::-1]], textposition="outside", cliponaxis=False))
        style_fig(fig, max(360, 26 * len(tt) + 80), f"{cat} de mayor gasto en {amb_sel}")
        fig.update_layout(hovermode="closest", margin=dict(l=10, r=90, t=48, b=10))
        fig.update_xaxes(range=[0, float(tt["gasto"].max()) * 1.2])
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp(f"Los {cat.lower()} que más gasto generan en «{amb_sel}». Muestra qué concreta el costo de cada forma de atención."), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 6. Gasto y evolución
# ---------------------------------------------------------------------------

def panel_gasto(d: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    c = st.columns(4)
    freq = c[0].radio("Periodo", list(FREQ_LABEL), horizontal=True, key=f"{name}_g_freq")
    metr_lbl = c[1].selectbox("Métrica", list(METRICS), key=f"{name}_g_m")
    desg = c[2].selectbox("Desglose", ["Total", "Tipo de consumo", "Diagnóstico (grupo)", "Sexo", "Grupo etario", "Departamento"], key=f"{name}_g_d")
    roll = c[3].checkbox("Añadir media móvil 12 meses", value=False, key=f"{name}_g_roll", disabled=freq != "Mensual")
    metric = METRICS[metr_lbl]
    fq = FREQ_LABEL[freq]
    if desg == "Total":
        wide = pd.DataFrame({metr_lbl: series_metric(d, metric, fq, idx)})
        stacked = False
    else:
        by = {"Tipo de consumo": "tipo", "Diagnóstico (grupo)": "dx_grupo", "Sexo": "sexo", "Grupo etario": "edad_grupo", "Departamento": "departamento"}[desg]
        db = d
        if by in ("dx_grupo", "sexo", "edad_grupo", "departamento"):
            sd_lbl = {"dx_grupo": SD_GROUP, "sexo": "Sin dato", "edad_grupo": "Sin dato", "departamento": "Sin dato"}[by]
            if st.checkbox(f"Omitir «{sd_lbl}» y recalcular los %", value=False, key=f"{name}_g_sd"):
                db = d[d[by].astype(str) != sd_lbl]
        wide = breakdown_wide(db, by, metric, fq, idx, top=7)
        if by == "tipo":
            wide = wide.rename(columns=TIPO_LABEL)
        stacked = metric in ("gasto", "lineas")
    cm = {TIPO_LABEL[k_]: v for k_, v in TIPO_COLOR.items()}
    fig = time_figure(wide, o, title=f"{metr_lbl} · {freq.lower()}" + ("" if desg == "Total" else f" · por {desg.lower()}"),
                      money=metric_money(metric), stacked=stacked, colors=cm, height=440)
    if roll and desg == "Total":
        r = wide.iloc[:, 0].rolling(12, min_periods=12).mean() if metric != "pacientes" else wide.iloc[:, 0].rolling(12, min_periods=12).mean()
        fig.add_trace(go.Scatter(x=r.index, y=r.values, name="Media móvil 12m", line=dict(color=SECONDARY, width=3)))
    st.plotly_chart(fig, **STRETCH)
    ser0 = wide.sum(axis=1) if (desg != "Total" and stacked) else wide.iloc[:, 0]
    st.markdown(interp(f"Serie de «{metr_lbl.lower()}» a frecuencia {freq.lower()}. El máximo ocurrió en <b>{mes(ser0.idxmax())}</b> "
                       f"({soles_short(ser0.max()) if metric_money(metric) else f'{ser0.max():,.0f}'}) y el promedio del periodo es "
                       f"{soles_short(ser0.mean()) if metric_money(metric) else f'{ser0.mean():,.0f}'}. "
                       "Con desglose, cada franja muestra qué componente explica el nivel; la media móvil de 12 meses quita la estacionalidad y deja ver la dirección de fondo."), unsafe_allow_html=True)

    ma = annual_metrics(d)
    # Comparación misma ventana
    last = d["tiempo"].max()
    m_last = last.month
    left, right = st.columns(2)
    with left:
        ytd = d[d["tiempo"].dt.month <= m_last].groupby("anio", observed=True)["gasto"].sum()
        fig = go.Figure(go.Bar(x=ytd.index.astype(str), y=ytd.values, marker_color=[SECONDARY if int(y) == last.year else PRIMARY for y in ytd.index],
                               text=[soles_short(v) for v in ytd.values], textposition="outside"))
        style_fig(fig, 340, f"Gasto acumulado ene–{MES_ES[m_last - 1].lower()} de cada año (misma ventana)")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        if len(ytd) >= 2:
            st.markdown(interp(f"Compara todos los años con la misma ventana de meses (ene a {MES_ES[m_last - 1].lower()}), por lo que el año en curso no queda en desventaja. "
                               f"{int(ytd.index[-1])} acumula {soles_short(ytd.iloc[-1])}, {(ytd.iloc[-1] / ytd.iloc[-2] - 1) * 100:+.1f}% frente a {int(ytd.index[-2])}." if ytd.iloc[-2] else ""),
                        unsafe_allow_html=True)
    with right:
        mm = monthly_metrics(d, idx)
        piv = mm["gasto"].groupby([mm.index.year, mm.index.month]).sum().unstack()
        piv = piv.reindex(columns=range(1, 13))
        # meses fuera del índice => NaN
        mask = pd.DataFrame(False, index=piv.index, columns=piv.columns)
        for t in idx:
            mask.loc[t.year, t.month] = True
        piv = piv.where(mask)
        fig = go.Figure(go.Heatmap(z=piv.values, x=MES_ES, y=piv.index.astype(str), colorscale="Blues", hoverongaps=False,
                                   hovertemplate="%{y} %{x}: S/ %{z:,.0f}<extra></extra>"))
        style_fig(fig, 340, "Mapa de calor: gasto por año y mes")
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("Cada celda es el gasto de un mes; más oscuro significa más gasto. Las columnas oscuras repetidas revelan meses estacionalmente altos y las celdas aisladas oscuras, picos puntuales."),
                    unsafe_allow_html=True)

    st.markdown("#### Variación interanual")
    mm = monthly_metrics(d, idx)
    g = mm["gasto"]
    yoy = (g / g.shift(12).replace(0, np.nan) - 1) * 100
    r12 = g.rolling(12, min_periods=12).sum()
    yoy12 = (r12 / r12.shift(12).replace(0, np.nan) - 1) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=yoy.index, y=yoy.values, name="Mes vs mismo mes del año previo", marker_color=hex_to_rgba(PRIMARY, .55)))
    fig.add_trace(go.Scatter(x=yoy12.index, y=yoy12.values, name="12 meses móviles vs 12 previos", line=dict(color=SECONDARY, width=2.5)))
    fig.add_hline(y=0, line_color="#999")
    style_fig(fig, 340, None, "Variación %")
    yl = float(np.nanpercentile(np.abs(yoy.dropna()), 90)) if yoy.notna().sum() > 3 else 100
    fig.update_yaxes(range=[-min(300, max(100, yl * 1.3)), min(500, max(100, yl * 1.3))])
    st.plotly_chart(fig, **STRETCH)
    st.caption("El eje se recorta para que los meses extremos no aplasten el resto; pasa el cursor por las barras para ver el valor real.")
    last_y = yoy12.dropna()
    st.markdown(interp("Las barras comparan cada mes con el mismo mes del año anterior y la línea compara los últimos 12 meses con los 12 previos, que es la lectura más estable. "
                       + (f"Hoy la variación de 12 meses es <b>{last_y.iloc[-1]:+.1f}%</b>." if len(last_y) else "Aún no hay 24 meses para calcularla.")), unsafe_allow_html=True)

    st.markdown("#### Resumen anual")
    ma2 = ma.copy()
    ma2["delta"] = ma2["gasto"].pct_change() * 100
    base_y = ma2.index[0]
    ma2["indice"] = ma2["gasto"] / ma2["gasto"].iloc[0] * 100
    ma2["estado"] = ["Completo" if m_ == 12 and y in complete_years(idx) else f"Parcial ({m_} meses)" for y, m_ in zip(ma2.index, ma2["meses"])]
    show = ma2.reset_index().rename(columns={"anio": "Año", "gasto": "Gasto", "lineas": "Líneas", "pacientes": "Pacientes", "costo_paciente": "S/ por paciente",
                                             "delta": "Δ % vs año previo", "indice": f"Índice (base {base_y}=100)", "estado": "Cobertura"})
    st.dataframe(fmt_df(show[["Año", "Cobertura", "Gasto", "Δ % vs año previo", f"Índice (base {base_y}=100)", "Pacientes", "S/ por paciente", "Líneas"]],
                        money=["Gasto", "S/ por paciente"], pcts=["Δ % vs año previo"], ints=["Líneas", "Pacientes"], decs=[f"Índice (base {base_y}=100)"]),
                 hide_index=True, **STRETCH)
    full = [y for y in complete_years(idx) if y in ma.index]
    if len(full) >= 2:
        cg = cagr(float(ma.loc[full[0], "gasto"]), float(ma.loc[full[-1], "gasto"]), full[-1] - full[0])
        st.markdown(interp(f"Entre {full[0]} y {full[-1]} (años completos) el gasto de {name} varió a una tasa anual compuesta de <b>{pct(cg * 100 if cg is not None else None)}</b>."),
                    unsafe_allow_html=True)
    st.markdown(interp("La tabla resume cada año: gasto, variación frente al año previo, índice con base en el primer año (=100), pacientes y costo por paciente. "
                       "Los años parciales se marcan en «Cobertura» y no deben compararse en niveles."), unsafe_allow_html=True)
    mm_out = mm.reset_index()
    csv_button(mm_out, "Descargar serie mensual (CSV)", f"serie_mensual_{name}.csv", f"{name}_g_csv")


# ---------------------------------------------------------------------------
# 7. Tendencias
# ---------------------------------------------------------------------------

def _mk_row(label: str, s: pd.Series, alpha: float) -> dict | None:
    s = s.dropna()
    mk = mann_kendall_trend(s.values, alpha)
    if mk is None:
        return None
    media = s.mean()
    return {"Serie": label, "Meses": len(s), "Tendencia": mk.trend, "p": mk.p_value, "Sen (S//mes)": mk.sen_slope,
            "Sen anual (% de la media)": mk.sen_slope * 12 / media * 100 if media else np.nan, "Variante": mk.variant_used}


def panel_tendencias(d: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    c = st.columns([1.3, 1, 1])
    metr_lbl = c[0].selectbox("Métrica a evaluar", list(METRICS), key=f"{name}_t_m")
    excl_covid = c[1].checkbox("Excluir mar–dic 2020 (pandemia)", value=False, key=f"{name}_t_covid")
    excl_last = c[2].checkbox("Excluir último mes", value=False, key=f"{name}_t_last",
                              help="El último mes puede estar incompleto por rezago de facturación.")
    metric = METRICS[metr_lbl]
    s = series_metric(d, metric, "MS", idx)
    if excl_last and len(s):
        s = s.iloc[:-1]
    s = drop_covid(s, excl_covid)
    if metric == "costo_paciente":
        s = s.dropna()
    if len(s) < 8:
        st.info(f"Se necesitan al menos 8 meses para evaluar tendencia; hay {len(s)}.")
        return
    mk = mann_kendall_trend(s.values, o.alpha)
    first_nz = s[s > 0].index.min() if (s > 0).any() else s.index[0]

    k = st.columns(5)
    _ya = series_metric(d, metric, "YS", idx)
    _yrs = [y for y in complete_years(idx) if not (excl_covid and y == 2020)]
    _ya = _ya[_ya.index.year.isin(_yrs)]
    _cg = cagr(float(_ya.iloc[0]), float(_ya.iloc[-1]), int(_ya.index.year[-1] - _ya.index.year[0])) if len(_ya) >= 2 else None
    k[4].metric("CAGR (años completos)", pct(_cg * 100) if _cg is not None else "n/d",
                help="Crecimiento anual compuesto entre el primer y el último año completo: (último / primero)^(1/años) − 1. Solo usa los dos extremos, a diferencia del cambio anual implícito, que usa toda la serie mensual.")
    if mk:
        icon = {"creciente": "📈", "decreciente": "📉"}.get(mk.trend, "➖")
        k[0].metric("Mann-Kendall", f"{icon} {mk.trend}")
        k[1].metric("p-valor", f"{mk.p_value:.3f}", significance_label(mk.p_value, o.alpha), delta_color="off")
        k[2].metric("Pendiente de Sen", (soles_short(mk.sen_slope) if metric_money(metric) else f"{mk.sen_slope:,.2f}") + " /mes")
        mean_ = s.mean()
        k[3].metric("Cambio anual implícito", pct(mk.sen_slope * 12 / mean_ * 100 if mean_ else None), help="Pendiente de Sen × 12 sobre la media de la serie. Es un cambio lineal sobre el nivel medio; no equivale al crecimiento anual compuesto (CAGR) de la pestaña Gasto, que compara solo el primer y el último año completos.")

    if mk:
        txt = (f"La serie de <b>{metr_lbl.lower()}</b> de {name} muestra tendencia <b>{mk.trend}</b> "
               f"(Mann-Kendall {mk.variant_used}, p = {mk.p_value:.3f}; {significance_label(mk.p_value, o.alpha).lower()} con α = {o.alpha}). ")
        if (s == 0).mean() > 0.2 and metric in ("gasto", "lineas", "pacientes"):
            txt += "Ojo: más del 20% de los meses son cero, lo que debilita el test. "
        txt += ("El p-valor es la probabilidad de ver una tendencia así si la serie no tuviera ninguna; por debajo de α se considera tendencia real. "
                "La pendiente de Sen es el cambio mensual típico, poco sensible a meses extremos.")
        st.markdown(interp(txt), unsafe_allow_html=True)

    # gráfico con tendencia
    x = np.arange(len(s))
    sen = None
    if mk:
        med_int = np.median(s.values - mk.sen_slope * x)
        sen = med_int + mk.sen_slope * x
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name=metr_lbl, mode="lines+markers" if o.markers else "lines", line=dict(color=PRIMARY, width=2)))
    if sen is not None:
        fig.add_trace(go.Scatter(x=s.index, y=sen, name="Tendencia de Sen", line=dict(color=SECONDARY, width=3, dash="dash")))
    sm = smooth_series(np.nan_to_num(s.values.astype(float)))
    if sm is not None and o.lowess:
        fig.add_trace(go.Scatter(x=s.index, y=sm, name="LOWESS", line=dict(color=OK, width=2.5)))
    style_fig(fig, 400, f"{metr_lbl}: serie mensual y tendencia")
    if metric_money(metric):
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    if o.log:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, **STRETCH)
    st.markdown(interp("La línea azul es la serie mensual y la discontinua la tendencia estimada con la pendiente de Sen (la mediana de todas las pendientes entre pares de meses, "
                       "poco sensible a meses atípicos). Si los puntos se agrupan alrededor de la línea, la tendencia describe bien la serie; si oscilan mucho, el nivel mensual es muy variable."),
                unsafe_allow_html=True)

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("##### Tendencia anual (años completos)")
        ya = series_metric(d, metric, "YS", idx)
        yrs = [y for y in complete_years(idx) if not (excl_covid and y == 2020)]
        ya = ya[ya.index.year.isin(yrs)]
        if len(ya) >= 3:
            lt = linear_trend(ya.index.year.values, ya.values)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ya.index.year.astype(str), y=ya.values, marker_color=hex_to_rgba(PRIMARY, .7), name="Observado"))
            fig.add_trace(go.Scatter(x=ya.index.year.astype(str), y=lt.y_pred, name="Regresión lineal", line=dict(color=SECONDARY, width=3, dash="dash")))
            style_fig(fig, 320)
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, **STRETCH)
            st.caption(f"Pendiente: {lt.slope:,.1f} por año · R² = {lt.r2:.2f} · p = {lt.p_value:.3f} ({significance_label(lt.p_value, o.alpha).lower()}). "
                       f"Con {len(ya)} puntos, es una guía y no una prueba.")
            st.markdown(interp(f"Ajusta una recta a los totales anuales de años completos: el gasto {'sube' if lt.slope > 0 else 'baja'} en promedio <b>{soles_short(abs(lt.slope)) if metric_money(metric) else f'{abs(lt.slope):,.1f}'}</b> por año. "
                               "El R² indica qué parte de la variación anual explica la recta (cerca de 1, casi toda)."), unsafe_allow_html=True)
        else:
            st.info("Se necesitan al menos 3 años completos para la regresión anual.")
    with a2:
        st.markdown("##### Estacionalidad (índice por mes del año)")
        if len(s) >= 24:
            ratio = s / s.rolling(12, center=True, min_periods=6).mean().replace(0, np.nan)
            seas = ratio.groupby(ratio.index.month).mean()
            fig = go.Figure(go.Bar(x=[MES_ES[m - 1] for m in seas.index], y=seas.values,
                                   marker_color=[SECONDARY if v > 1 else PRIMARY for v in seas.values]))
            fig.add_hline(y=1, line_dash="dot", line_color="#888")
            style_fig(fig, 320)
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig, **STRETCH)
            st.caption("Valores > 1: meses típicamente por encima del promedio móvil anual.")
            st.markdown(interp(f"Cada barra compara un mes con el promedio móvil de su año. El mes típicamente más alto es <b>{MES_ES[int(seas.idxmax()) - 1]}</b> ({(seas.max() - 1) * 100:+.0f}%) y el más bajo "
                               f"<b>{MES_ES[int(seas.idxmin()) - 1]}</b> ({(seas.min() - 1) * 100:+.0f}%). Si todas las barras rondan 1, no hay estacionalidad relevante."), unsafe_allow_html=True)
        else:
            st.info("Se necesitan al menos 24 meses para estimar estacionalidad.")

    st.markdown("##### Cambios de nivel o de pendiente")
    b1, b2 = st.columns(2)
    with b1:
        auto = detect_breakpoint_auto(s, min_segment=6)
        if auto is None:
            st.info("Serie demasiado corta para buscar un quiebre automático (mínimo 12 meses).")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="markers", marker=dict(color=PRIMARY, size=5), name="Observado"))
            fig.add_trace(go.Scatter(x=auto.x_before, y=auto.y_pred_before, name="Ajuste antes", line=dict(color=SECONDARY, width=2)))
            fig.add_trace(go.Scatter(x=auto.x_after, y=auto.y_pred_after, name="Ajuste después", line=dict(color=OK, width=2)))
            style_fig(fig, 320, f"Quiebre automático: {MES_ES[auto.date.month - 1]}-{auto.date.year}")
            st.plotly_chart(fig, **STRETCH)
            st.caption(f"Pendiente antes {auto.slope_before:,.1f}/mes, después {auto.slope_after:,.1f}/mes · Chow F = {auto.f_stat:.2f}, p = {auto.p_value:.3f} "
                       f"({'significativo' if auto.significant else 'no significativo'}). La fecha se busca sobre los datos, así que el p-valor es optimista.")
            st.markdown(interp(f"El algoritmo busca la fecha en que la serie cambia más de nivel o pendiente: <b>{MES_ES[auto.date.month - 1]}-{auto.date.year}</b>. "
                               f"Antes la pendiente era {auto.slope_before:,.1f}/mes y después {auto.slope_after:,.1f}/mes. "
                               f"{'El cambio es estadísticamente significativo, aunque la fecha se eligió mirando los datos.' if auto.significant else 'El cambio no es estadísticamente significativo.'}"), unsafe_allow_html=True)
    with b2:
        ev_date = st.date_input("Fecha de un evento a evaluar", value=pd.Timestamp("2022-01-01").date(),
                                min_value=s.index.min().date(), max_value=s.index.max().date(), key=f"{name}_t_ev")
        lag = st.slider("Meses de transición a excluir tras el evento", 0, 12, 0, key=f"{name}_t_lag")
        ev = chow_test_event(s, pd.Timestamp(ev_date).replace(day=1), lag)
        if ev is None:
            st.info("Muy pocos meses antes o después de esa fecha (mínimo 4 en cada lado).")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="markers", marker=dict(color=PRIMARY, size=5), name="Observado"))
            fig.add_trace(go.Scatter(x=ev.x_before, y=ev.y_pred_before, name="Ajuste antes", line=dict(color=SECONDARY, width=2)))
            fig.add_trace(go.Scatter(x=ev.x_after, y=ev.y_pred_after, name="Ajuste después", line=dict(color=OK, width=2)))
            fig.add_vline(x=pd.Timestamp(ev_date).replace(day=1).timestamp() * 1000, line_dash="dot", line_color="#888")
            style_fig(fig, 320, "Test de Chow en la fecha elegida")
            st.plotly_chart(fig, **STRETCH)
            st.caption(f"Pendiente antes {ev.slope_before:,.1f}/mes, después {ev.slope_after:,.1f}/mes · F = {ev.f_stat:.2f}, p = {ev.p_value:.3f} "
                       f"({'significativo' if ev.significant else 'no significativo'}).")
            st.markdown(interp(f"Compara las dos rectas ajustadas antes y después de la fecha que elijas. "
                               f"{'Hay evidencia de que el comportamiento de la serie cambió' if ev.significant else 'No hay evidencia suficiente de que el comportamiento cambiara'} "
                               f"en esa fecha (p = {ev.p_value:.3f}). Es asociación temporal, no prueba que el evento sea la causa."), unsafe_allow_html=True)

    st.markdown("##### Tendencia por componente")
    comp = st.radio("Componente", ["Tipo de consumo", "Diagnóstico (grupo)", "Medicamentos (principio activo)"], horizontal=True, key=f"{name}_t_comp")
    by = {"Tipo de consumo": "tipo", "Diagnóstico (grupo)": "dx_grupo", "Medicamentos (principio activo)": "principio_activo"}[comp]
    base = d[d["tipo"] == "MEDICAMENTO"] if by == "principio_activo" else d
    if by == "dx_grupo" and st.checkbox(f"Omitir «{SD_GROUP}»", value=False, key=f"{name}_t_sd"):
        base = base[base["dx_grupo"] != SD_GROUP]
    cats = base.groupby(by, observed=True)["gasto"].sum().sort_values(ascending=False).head(12).index.tolist()
    rows = []
    for cc in cats:
        sc = series_metric(base[base[by] == cc], "gasto", "MS", idx)
        if excl_last and len(sc):
            sc = sc.iloc[:-1]
        sc = drop_covid(sc, excl_covid)
        r = _mk_row(str(TIPO_LABEL.get(cc, cc)), sc, o.alpha)
        if r:
            r["Gasto total"] = base.loc[base[by] == cc, "gasto"].sum()
            rows.append(r)
    if rows:
        tb = pd.DataFrame(rows)
        tb["Lectura"] = [significance_label(p_, o.alpha) for p_ in tb["p"]]
        tb["p"] = [num(v, 3) for v in tb["p"]]

        def _c(v):
            return {"creciente": f"background-color:{BAD}33", "decreciente": f"background-color:{OK}33"}.get(v, "")
        st.dataframe(fmt_df(tb[["Serie", "Gasto total", "Meses", "Tendencia", "Lectura", "p", "Sen (S//mes)", "Sen anual (% de la media)"]],
                            money=["Gasto total", "Sen (S//mes)"], pcts=["Sen anual (% de la media)"], ints=["Meses"]).map(_c, subset=["Tendencia"]),
                     hide_index=True, **STRETCH)
        st.caption("Gasto mensual de cada componente. Con muchas filas probadas a la vez, algún p-valor bajo saldrá por azar: úsalo para priorizar, no para concluir.")
        n_up = int((tb["Tendencia"] == "creciente").sum())
        n_dn = int((tb["Tendencia"] == "decreciente").sum())
        st.markdown(interp(f"De {len(tb)} componentes evaluados, {n_up} tienen tendencia creciente y {n_dn} decreciente (en rojo y verde). "
                           "La columna «Sen anual» expresa el cambio anual típico como porcentaje de la media, útil para comparar componentes de distinto tamaño."), unsafe_allow_html=True)
    else:
        st.info("Ningún componente tiene suficientes meses para el test.")


# ---------------------------------------------------------------------------
# 8. Proyecciones
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Calculando proyección...", max_entries=64)
def run_projection(vals: tuple, start: str, horizon: int, n_boot: int, ci: float):
    s = pd.Series(list(vals), index=pd.date_range(start, periods=len(vals), freq="MS"))
    return project_series(s, horizon=horizon, n_boot=n_boot, ci=ci)


def panel_proyecciones(d: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    c = st.columns(4)
    metr_lbl = c[0].selectbox("Métrica a proyectar", list(METRICS), key=f"{name}_p_m")
    horizon = c[1].slider("Horizonte (meses)", 1, 24, 12, key=f"{name}_p_h")
    window = c[2].selectbox("Entrenar con", ["Toda la serie", "Últimos 36 meses", "Últimos 24 meses", "Últimos 12 meses"], key=f"{name}_p_w")
    ci = c[3].select_slider("Banda de incertidumbre", options=[0.80, 0.90, 0.95], value=0.90, key=f"{name}_p_ci")
    c2 = st.columns(4)
    excl_covid = c2[0].checkbox("Excluir mar–dic 2020", value=False, key=f"{name}_p_covid")
    winsor = c2[1].checkbox("Atenuar picos (tope en P95)", value=True, key=f"{name}_p_wins",
                            help="Reemplaza los meses por encima del percentil 95 por ese valor antes de ajustar. Activado por defecto: un solo medicamento caro (p. ej. jul-2026 en SALUDPOL) puede inflar toda la proyección. Desactívalo para ver la serie cruda.")
    growth = c2[2].number_input("Escenario propio: variación anual (%)", -50.0, 200.0, 0.0, 1.0, key=f"{name}_p_g",
                                help="Nivel base = media de los últimos 12 meses, creciendo a esta tasa anual.")
    nboot = c2[3].select_slider("Réplicas bootstrap", options=[100, 300, 600], value=300, key=f"{name}_p_nb")

    metric = METRICS[metr_lbl]
    s = series_metric(d, metric, "MS", idx)
    if metric == "costo_paciente":
        s = s.dropna()
    s = drop_covid(s, excl_covid)
    n = {"Toda la serie": None, "Últimos 36 meses": 36, "Últimos 24 meses": 24, "Últimos 12 meses": 12}[window]
    if n:
        s = s.iloc[-n:]
    if winsor and len(s) >= 12:
        cap = np.nanpercentile(s.values, 95)
        s = s.clip(upper=cap)
    if len(s) < 12:
        st.info(f"Se necesitan al menos 12 meses para proyectar; hay {len(s)} en la selección.")
        return
    small_sample_note(d, idx, name)
    proj = run_projection(tuple(float(v) for v in s.values), str(s.index[0].date()), horizon, nboot, ci)
    if proj is None:
        st.warning("No se pudo ajustar el modelo con esta serie.")
        return
    # los índices de futuro deben continuar el calendario real aunque se haya excluido 2020
    fut_idx = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Histórico", mode="lines+markers" if o.markers else "lines", line=dict(color=PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=fut_idx, y=proj.recommended, name="Recomendado (Holt amortiguado)", line=dict(color=SECONDARY, width=3)))
    fig.add_trace(go.Scatter(x=list(fut_idx) + list(fut_idx[::-1]), y=list(proj.recommended_upper) + list(proj.recommended_lower[::-1]),
                             fill="toself", fillcolor=hex_to_rgba(SECONDARY, .13), line=dict(color="rgba(0,0,0,0)"), name=f"Banda {int(ci * 100)}%", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fut_idx, y=proj.conservative, name="Conservador (media 3 meses)", line=dict(color=WARN, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=fut_idx, y=proj.optimistic, name="Lineal (Sen)", line=dict(color=OK, width=2, dash="dash")))
    user = None
    if growth != 0.0:
        base = s.iloc[-12:].mean()
        user = base * (1 + growth / 100) ** (np.arange(1, horizon + 1) / 12)
        fig.add_trace(go.Scatter(x=fut_idx, y=user, name=f"Escenario propio ({growth:+.0f}% anual)", line=dict(color="#8850C4", width=2, dash="dashdot")))
    style_fig(fig, 440, f"Proyección de {metr_lbl.lower()} · {horizon} meses")
    if metric_money(metric):
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    st.plotly_chart(fig, **STRETCH)

    if metric == "gasto":
        last12 = s.iloc[-12:].sum()
        p12 = proj.recommended[:12].sum() if horizon >= 12 else None
        k = st.columns(4)
        k[0].metric("Últimos 12 meses observados", soles_short(last12))
        if p12 is not None:
            k[1].metric("Próximos 12 meses (recomendado)", soles_short(p12), f"{(p12 / last12 - 1) * 100:+.1f}%" if last12 else None)
            k[2].metric("Rango de la banda", f"{soles_short(proj.recommended_lower[:12].sum())} – {soles_short(proj.recommended_upper[:12].sum())}")
        # cierre del año en curso
        yr = idx[-1].year
        obs_ytd = d.loc[d["anio"] == yr, "gasto"].sum()
        mo_obs = d.loc[d["anio"] == yr, "tiempo"].dt.month.max() if (d["anio"] == yr).any() else 0
        rest_n = 12 - int(idx[-1].month)
        if rest_n > 0 and horizon >= rest_n and not excl_covid:
            cierre = obs_ytd + proj.recommended[:rest_n].sum()
            prev = d.loc[d["anio"] == yr - 1, "gasto"].sum()
            k[3].metric(f"Cierre estimado {yr}", soles_short(cierre), f"{(cierre / prev - 1) * 100:+.1f}% vs {yr - 1}" if prev else None, delta_color="off")
        st.caption("Cierre estimado = gasto observado del año en curso + meses restantes del escenario recomendado.")
        if p12 is not None and last12:
            var = (p12 / last12 - 1) * 100
            lo12, hi12 = proj.recommended_lower[:12].sum(), proj.recommended_upper[:12].sum()
            ancho = (hi12 - lo12) / p12 * 100 if p12 else np.nan
            st.markdown(interp(
                f"Con el escenario recomendado, el gasto de {name} en los próximos 12 meses sería <b>{soles_short(p12)}</b>, "
                f"<b>{var:+.1f}%</b> frente a los últimos 12 observados ({soles_short(last12)}). La banda va de {soles_short(lo12)} a {soles_short(hi12)}"
                f"{f', un rango de ±{ancho / 2:,.0f}% alrededor del central, así que la cifra es orientativa' if ancho == ancho and ancho > 60 else ''}."),
                unsafe_allow_html=True)

    tabla = pd.DataFrame({"Mes": [f"{MES_ES[t.month - 1]}-{t.year}" for t in fut_idx], "Conservador": proj.conservative, "Recomendado": proj.recommended,
                          "Límite inferior": proj.recommended_lower, "Límite superior": proj.recommended_upper, "Lineal (Sen)": proj.optimistic})
    if user is not None:
        tabla["Escenario propio"] = user
    with st.expander("Tabla de proyección"):
        money = metric_money(metric)
        st.dataframe(fmt_df(tabla, money=[c_ for c_ in tabla.columns if c_ != "Mes"] if money else [], decs=[] if money else [c_ for c_ in tabla.columns if c_ != "Mes"]),
                     hide_index=True, **STRETCH)
        csv_button(tabla, "Descargar proyección (CSV)", f"proyeccion_{name}.csv", f"{name}_p_csv")
    st.markdown(interp(proj.method_note + " La banda proviene de remuestrear los residuos del ajuste; no incluye cambios de política, nuevos tratamientos ni "
                       "cambios en la cartera de asegurados, que son justamente lo que más mueve el gasto oncológico."), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Comparativo entre IAFAS
# ---------------------------------------------------------------------------

def panel_comparativo(d: pd.DataFrame, iafas_list: list[str], o: Opts) -> None:
    if d.empty:
        st.info("Sin datos en la selección.")
        return
    inc_onco = st.checkbox("Incluir ONCOSALUD en los indicadores", value=True, key="cmp_onco",
                           help="Desmárcalo para ver solo las IAFAS públicas (convenio) y separar el contrato con ONCOSALUD.")
    if not inc_onco:
        d = d[d["iafas"] != "ONCOSALUD"]
        iafas_list = [i for i in iafas_list if i != "ONCOSALUD"]
        if d.empty:
            st.info("Sin datos de IAFAS públicas en la selección.")
            return
    idx_all = month_index(d["tiempo"].min(), d["tiempo"].max())
    rows = []
    for nm in iafas_list:
        di = d[d["iafas"] == nm]
        if di.empty:
            continue
        pt = patient_table(di)
        mm = monthly_metrics(di, month_index(di["tiempo"].min(), d["tiempo"].max()))
        rows.append({"IAFAS": nm, "Gasto": di["gasto"].sum(), "% del total": di["gasto"].sum() / d["gasto"].sum() * 100,
                     "Pacientes": len(pt), "S/ por paciente": di["gasto"].sum() / len(pt) if len(pt) else np.nan,
                     "Pacientes activos/mes": mm["pacientes"].mean(), "Gasto medio mensual (total IAFAS)": mm["gasto"].mean(),
                     "Primer mes": di["tiempo"].min(), "Meses con consumo": di["tiempo"].nunique(),
                     "% medicamentos": di.loc[di.tipo == "MEDICAMENTO", "gasto"].sum() / di["gasto"].sum() * 100,
                     "% insumos": di.loc[di.tipo == "INSUMO", "gasto"].sum() / di["gasto"].sum() * 100,
                     "% procedimientos": di.loc[di.tipo == "PROCEDIMIENTO", "gasto"].sum() / di["gasto"].sum() * 100})
    t = pd.DataFrame(rows)
    k = st.columns(4)
    k[0].metric("Gasto total", soles_short(d["gasto"].sum()))
    k[1].metric("Pacientes únicos", f"{d.loc[d.pid >= 0, 'pid'].nunique():,}")
    k[2].metric("IAFAS con consumo", f"{len(t)}")
    k[3].metric("Concentración de la mayor IAFAS", pct(t["% del total"].max()), t.sort_values("Gasto", ascending=False)["IAFAS"].iloc[0], delta_color="off")
    tops = t.sort_values("Gasto", ascending=False)
    resto = tops.iloc[1:]
    st.markdown(interp(
        f"El INEN facturó <b>{soles_short(d['gasto'].sum())}</b> a {len(t)} IAFAS en el periodo. <b>{tops['IAFAS'].iloc[0]}</b> concentra "
        f"<b>{pct(tops['% del total'].iloc[0])}</b> del gasto"
        + (f" y las otras {len(resto)} suman {pct(resto['% del total'].sum())}, por lo que los resultados globales reflejan sobre todo a la primera." if len(resto) else ".")
        + " El costo por paciente permite comparar IAFAS de distinto tamaño."), unsafe_allow_html=True)
    st.dataframe(fmt_df(t, money=["Gasto", "S/ por paciente", "Gasto medio mensual (total IAFAS)"], pcts=["% del total", "% medicamentos", "% insumos", "% procedimientos"],
                        ints=["Pacientes", "Meses con consumo"], decs=["Pacientes activos/mes"], dates=["Primer mes"]), hide_index=True, **STRETCH)
    c = st.columns(2)
    with c[0]:
        modo = st.radio("Vista", ["Gasto (S/)", "Participación (%)"], horizontal=True, key="cmp_modo")
        frq = st.radio("Periodo", ["Anual", "Trimestral", "Mensual"], horizontal=True, key="cmp_frq", index=0)
        cols = {nm: series_metric(d[d["iafas"] == nm], "gasto", FREQ_LABEL[frq], idx_all) for nm in iafas_list if (d["iafas"] == nm).any()}
        wide = pd.DataFrame(cols)
        if frq == "Anual":
            wide.index = wide.index.year.astype(str)
        if modo.startswith("Part"):
            wide = wide.div(wide.sum(axis=1).replace(0, np.nan), axis=0) * 100
        fig = go.Figure()
        for i, cc in enumerate(wide.columns):
            fig.add_trace(go.Bar(x=wide.index, y=wide[cc], name=cc, marker_color=PALETTE[i % len(PALETTE)]))
        fig.update_layout(barmode="stack")
        style_fig(fig, 400, "Gasto por IAFAS")
        fig.update_yaxes(ticksuffix="%" if modo.startswith("Part") else "", tickprefix="" if modo.startswith("Part") else "S/ ", tickformat=",.0f")
        st.plotly_chart(fig, **STRETCH)
        st.markdown(interp("Gasto de cada IAFAS en el tiempo, apilado. En «Participación» se ve cómo cambia el peso relativo de cada una sin el efecto del volumen total."), unsafe_allow_html=True)
    with c[1]:
        fig = go.Figure()
        for i, tp in enumerate(["MEDICAMENTO", "INSUMO", "PROCEDIMIENTO"]):
            fig.add_trace(go.Bar(y=t["IAFAS"], x=t[{"MEDICAMENTO": "% medicamentos", "INSUMO": "% insumos", "PROCEDIMIENTO": "% procedimientos"}[tp]],
                                 name=TIPO_LABEL[tp], orientation="h", marker_color=TIPO_COLOR[tp]))
        fig.update_layout(barmode="stack")
        style_fig(fig, 400, "Estructura del gasto por tipo de consumo")
        fig.update_xaxes(ticksuffix="%", range=[0, 100])
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        mxp = t.sort_values("% medicamentos", ascending=False).iloc[0]
        st.markdown(interp(f"Qué parte del gasto de cada IAFAS es medicamentos, insumos o procedimientos. <b>{mxp['IAFAS']}</b> es la que más gasta en medicamentos ({pct(mxp['% medicamentos'])}); "
                           "diferencias grandes de estructura reflejan distintos tipos de tratamiento (más quimioterapia oral o biológicos frente a más radioterapia o cirugía)."), unsafe_allow_html=True)

    st.markdown("#### Diagnósticos principales por IAFAS")
    h1, h2, h3 = st.columns([1.3, 1.2, 1])
    vista = h1.radio("Vista", ["Peso de cada diagnóstico en la IAFAS", "Estructura interna del gasto de cada diagnóstico"], key="cmp_hvista")
    omit = h2.checkbox(f"Omitir «{SD_GROUP}» (recalcula el 100%)", value=False, key="cmp_hsd")
    nd = h3.slider("Diagnósticos a mostrar", 5, 20, 12, key="cmp_hn")
    dh = d[d["dx_grupo"] != SD_GROUP] if omit else d
    if vista.startswith("Peso"):
        top_dx = dh.groupby("dx_grupo", observed=True)["gasto"].sum().sort_values(ascending=False).head(nd).index.tolist()
        mat = pd.DataFrame({nm: dh[dh["iafas"] == nm].groupby("dx_grupo", observed=True)["gasto"].sum().reindex(top_dx).fillna(0)
                            / max(dh[dh["iafas"] == nm]["gasto"].sum(), 1e-9) * 100 for nm in t["IAFAS"]})
        mat = mat.round(1)
        fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=mat.index, colorscale="Blues", text=mat.values, texttemplate="%{text:.1f}%",
                                   hovertemplate="%{y} · %{x}: %{z:.1f}%<extra></extra>"))
        style_fig(fig, max(420, 34 * nd + 80))
        fig.update_layout(hovermode="closest")
        fig.update_yaxes(autorange="reversed", automargin=True)
        st.plotly_chart(fig, **STRETCH)
        st.caption(("Cada columna suma 100% sobre el gasto con diagnóstico registrado" if omit else "Cada columna suma 100% sobre todo el gasto de la IAFAS, incluido «Sin diagnóstico registrado»")
                   + f"; se muestran los {nd} diagnósticos de mayor gasto global.")
        st.markdown(interp("Cada celda es el porcentaje del gasto de una IAFAS que corresponde a ese diagnóstico. Compara perfiles clínicos: un color intenso en una sola columna indica un diagnóstico que pesa mucho más en esa IAFAS que en las demás. "
                           + ("" if omit else "Si una IAFAS tiene mucho gasto sin diagnóstico, todos sus porcentajes quedan comprimidos; omite ese grupo para comparar perfiles clínicos en igualdad de condiciones.")),
                    unsafe_allow_html=True)
    else:
        sel = st.selectbox("IAFAS", ["Todas"] + list(t["IAFAS"]), key="cmp_hsel")
        ds = dh if sel == "Todas" else dh[dh["iafas"] == sel]
        dx_structure_chart(ds, "dx_grupo", nd, "cmp_dxs", f"Estructura interna del gasto por diagnóstico · {sel}")
    st.markdown("#### Costo promedio por paciente con un mismo diagnóstico, según IAFAS")
    dxs = d[d["dx_grupo"] != "Sin diagnóstico registrado"].groupby("dx_grupo", observed=True)["gasto"].sum().sort_values(ascending=False).index.tolist()
    if dxs:
        cc1, cc2 = st.columns([2, 1])
        dsel = cc1.selectbox("Diagnóstico (grupo)", dxs, key="cmp_dxsel")
        nuevos = cc2.checkbox("Solo pacientes con inicio observado", value=False, key="cmp_dxnew")
        with st.expander("¿Qué significa «Solo pacientes con inicio observado»?"):
            st.markdown(NOTA_INICIO.split("** ", 1)[1])
        rows2 = []
        for nm in t["IAFAS"]:
            ct = dx_cost_table(d[(d["iafas"] == nm) & (d["dx_grupo"] == dsel)], "dx_grupo", nuevos)
            if len(ct):
                r = ct.iloc[0]
                rows2.append({"IAFAS": nm, "Pacientes": int(r["pacientes"]), "S/ promedio por paciente": r["media"], "Mediana": r["mediana"], "P90": r["p90"],
                              "Meses con consumo (media)": r["meses_medios"], "S/ por paciente-año": r["costo_anual"]})
        if rows2:
            t2 = pd.DataFrame(rows2).sort_values("S/ promedio por paciente", ascending=False)
            fig = go.Figure(go.Bar(y=t2["IAFAS"][::-1], x=t2["S/ promedio por paciente"][::-1], orientation="h", marker_color=PRIMARY,
                                   text=[f"{soles_short(v)} ({n:,} pac.)" for v, n in zip(t2["S/ promedio por paciente"][::-1], t2["Pacientes"][::-1])],
                                   textposition="outside", cliponaxis=False))
            style_fig(fig, max(240, 60 * len(t2) + 80), None, None)
            fig.update_layout(hovermode="closest", margin=dict(l=10, r=110, t=18, b=40))
            fig.update_xaxes(range=[0, float(t2["S/ promedio por paciente"].max()) * 1.3])
            st.plotly_chart(fig, **STRETCH)
            st.dataframe(fmt_df(t2, money=["S/ promedio por paciente", "Mediana", "P90", "S/ por paciente-año"], ints=["Pacientes"], decs=["Meses con consumo (media)"]), hide_index=True, **STRETCH)
            st.markdown(interp(NOTA_COSTOS), unsafe_allow_html=True)
            lo, hi = t2.iloc[-1], t2.iloc[0]
            st.markdown(interp(
                f"Gasto facturado por el INEN por paciente con <b>{dsel}</b>. {hi['IAFAS']} tiene el promedio más alto ({soles_short(hi['S/ promedio por paciente'])}, {int(hi['Pacientes']):,} pacientes)"
                + (f" y {lo['IAFAS']} el más bajo ({soles_short(lo['S/ promedio por paciente'])}, {int(lo['Pacientes']):,} pacientes)." if len(t2) > 1 else ".")
                + " Las diferencias pueden deberse a estadio, esquemas terapéuticos, tiempo de seguimiento observado o a pocos pacientes; mira la columna de pacientes antes de comparar."), unsafe_allow_html=True)
        else:
            st.info("Ninguna IAFAS tiene pacientes con ese diagnóstico en la selección.")
    csv_button(t, "Descargar comparativo (CSV)", "comparativo_iafas.csv", "cmp_csv")
