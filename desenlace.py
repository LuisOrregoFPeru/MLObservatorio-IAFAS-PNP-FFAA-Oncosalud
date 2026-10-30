"""
desenlace.py
------------
Pestaña «Estadío y desenlace»: estadío clínico registrado (DX_ESTADIO) y desenlace vital de los pacientes
financiados (FEC_DECESO, guardado solo como mes).

  1. Indicadores: pacientes, fallecidos, gasto en pacientes que fallecieron, gasto en los últimos meses de vida
     y consumo facturado después del mes de deceso (alerta de auditoría).
  2. Estadío: pacientes, gasto por paciente y mortalidad observada por estadío.
  3. Supervivencia de Kaplan-Meier desde el primer mes financiado, con prueba log-rank.
  4. Gasto al final de la vida: gasto por mes antes del deceso y quimioterapia en el mes del deceso.
  5. Consumo después del deceso: líneas facturadas en meses posteriores al mes de fallecimiento.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_processing import ESTADIO_ORDER, SD_GROUP, TIPO_LABEL, load_meta
from ui import (STRETCH, BAD, ESTADIO_COLOR, PALETTE, PRIMARY, TIPO_COLOR, VITAL_COLOR, fmt_df, interp, mes, num, pct,
                soles_short, style_fig, warn)

CONOCIDOS = ["0", "I", "II", "III", "IV"]


def _corte(d: pd.DataFrame) -> pd.Timestamp:
    c = load_meta().get("corte_vital")
    t = pd.Timestamp(c) if c else d["tiempo"].max()
    return t.to_period("M").to_timestamp()


def _es_quimio(d: pd.DataFrame) -> pd.Series:
    """Medicamentos oncológicos o procedimientos de administración de quimioterapia."""
    med = (d["tipo"] == "MEDICAMENTO") & (d["subtipo"].astype(str) == "Oncológicas")
    proc = (d["tipo"] == "PROCEDIMIENTO") & d["descripcion"].astype(str).str.upper().str.contains("QUIMIOTERAPIA", regex=False)
    return med | proc


@st.cache_data(show_spinner=False)
def tabla_pacientes(d: pd.DataFrame, corte: pd.Timestamp) -> pd.DataFrame:
    """Una fila por paciente: gasto, primer y último mes, diagnóstico principal, estadío y desenlace."""
    p = d[d["pid"] >= 0]
    if p.empty:
        return pd.DataFrame()
    base = p.groupby("pid").agg(gasto=("gasto", "sum"), primer_mes=("tiempo", "min"), ultimo_mes=("tiempo", "max"),
                                mes_deceso=("mes_deceso", "first"))
    # diagnóstico principal = el de mayor gasto, sin contar «Sin diagnóstico registrado»
    conx = p[p["dx_grupo"].astype(str) != SD_GROUP]
    gdx = conx.groupby(["pid", "dx_grupo"], observed=True)["gasto"].sum().reset_index().sort_values("gasto", ascending=False)
    principal = gdx.drop_duplicates("pid").set_index("pid")["dx_grupo"].astype(str)
    base["dx_principal"] = principal.reindex(base.index).fillna(SD_GROUP)
    # estadío: el más frecuente (en líneas) dentro del diagnóstico principal; se prefiere un estadío conocido
    sub = p.merge(base[["dx_principal"]], left_on="pid", right_index=True)
    sub = sub[sub["dx_grupo"].astype(str) == sub["dx_principal"]]
    cnt = sub.groupby(["pid", "estadio_grupo"], observed=True)["lineas"].sum().reset_index()
    cnt["conocido"] = cnt["estadio_grupo"].astype(str).isin(CONOCIDOS)
    cnt = cnt.sort_values(["pid", "conocido", "lineas"], ascending=[True, False, False])
    est = cnt.drop_duplicates("pid").set_index("pid")["estadio_grupo"].astype(str)
    base["estadio"] = est.reindex(base.index).fillna("Sin dato")
    dem = p.groupby("pid").agg(sexo=("sexo", "first"), edad_grupo=("edad_grupo", "first"))
    base = base.join(dem)
    base["fallecido"] = base["mes_deceso"].notna()
    fin = base["mes_deceso"].where(base["fallecido"], corte)
    base["meses_seg"] = ((fin.dt.year - base["primer_mes"].dt.year) * 12 + (fin.dt.month - base["primer_mes"].dt.month)).clip(lower=0)
    return base.reset_index()


def _km(t: np.ndarray, e: np.ndarray) -> pd.DataFrame:
    from statsmodels.duration.survfunc import SurvfuncRight
    if len(t) == 0:
        return pd.DataFrame(columns=["t", "s", "lo", "hi"])
    if e.sum() == 0:
        return pd.DataFrame({"t": [0, t.max()], "s": [1.0, 1.0], "lo": [1.0, 1.0], "hi": [1.0, 1.0]})
    sf = SurvfuncRight(t, e)
    st_, sp = np.asarray(sf.surv_times), np.asarray(sf.surv_prob)
    se = np.asarray(sf.surv_prob_se)
    # IC 95% log(-log)
    with np.errstate(divide="ignore", invalid="ignore"):
        ll = np.log(-np.log(sp))
        sell = se / (sp * np.abs(np.log(sp)))
        lo = np.exp(-np.exp(ll + 1.96 * sell))
        hi = np.exp(-np.exp(ll - 1.96 * sell))
    lo, hi = np.where(np.isfinite(lo), lo, sp), np.where(np.isfinite(hi), hi, sp)
    out = pd.DataFrame({"t": np.r_[0, st_, t.max()], "s": np.r_[1, sp, sp[-1]], "lo": np.r_[1, lo, lo[-1]], "hi": np.r_[1, hi, hi[-1]]})
    return out


def _s_at(km: pd.DataFrame, m: int) -> float:
    k = km[km["t"] <= m]
    return float(k["s"].iloc[-1]) * 100 if len(k) and km["t"].max() >= m else np.nan


def _mediana(km: pd.DataFrame) -> float:
    k = km[km["s"] <= 0.5]
    return float(k["t"].iloc[0]) if len(k) else np.nan


def panel_desenlace(d: pd.DataFrame, name: str, o, d_rest: pd.DataFrame | None = None) -> None:
    if "mes_deceso" not in d.columns or d["estadio_grupo"].astype(str).eq("Sin dato").all() and d["mes_deceso"].isna().all():
        st.info("La extracción cargada no trae estadío ni fecha de deceso. Regenera los datos con build_data.py a partir de la versión V02.")
        return
    corte = _corte(d)
    pac = tabla_pacientes(d, corte)
    if pac.empty:
        st.info("Sin pacientes identificados en la selección.")
        return
    n, nf = len(pac), int(pac["fallecido"].sum())
    gasto_id = d.loc[d["pid"] >= 0, "gasto"].sum()
    g_fall = pac.loc[pac["fallecido"], "gasto"].sum()
    dec = d[d["mes_deceso"].notna()]
    m6 = dec[(dec["meses_al_deceso"] >= 0) & (dec["meses_al_deceso"] <= 5)]["gasto"].sum()
    post = dec[dec["meses_al_deceso"] < 0]

    st.caption(f"Estado vital al {mes(corte)} (fecha de deceso más reciente de la extracción). Solo pacientes con DNI; "
               "el estadío y el deceso se leen tal como vienen registrados.")
    k = st.columns(5)
    k[0].metric("Pacientes financiados", num(n))
    tasa_rest = None
    if d_rest is not None and len(d_rest):
        pr = d_rest[d_rest["pid"] >= 0].groupby("pid")["mes_deceso"].first()
        tasa_rest = pr.notna().mean() * 100 if len(pr) else None
    k[1].metric("Fallecidos", f"{num(nf)} · {pct(nf / n * 100)}",
                help="Pacientes con fecha de deceso registrada." + (f" Resto de IAFAS: {pct(tasa_rest)}." if tasa_rest is not None else ""))
    k[2].metric("Gasto en pacientes que fallecieron", soles_short(g_fall), help=f"{pct(g_fall / gasto_id * 100 if gasto_id else np.nan)} del gasto con paciente identificado.")
    k[3].metric("Gasto en los últimos 6 meses de vida", soles_short(m6),
                help=f"Del mes del deceso a 5 meses antes. {pct(m6 / g_fall * 100 if g_fall else np.nan)} del gasto de los pacientes fallecidos.")
    k[4].metric("Facturado después del mes de deceso", soles_short(post["gasto"].sum()),
                help=f"{num(post['pid'].nunique())} pacientes con líneas en meses posteriores a su fallecimiento. Ver el detalle al final.")
    st.markdown(interp(
        f"De los {num(n)} pacientes financiados por {name} en la selección, <b>{num(nf)} ({pct(nf / n * 100)})</b> tienen fecha de deceso registrada. "
        f"Su atención representa {pct(g_fall / gasto_id * 100 if gasto_id else np.nan)} del gasto con paciente identificado y, de ese gasto, "
        f"{pct(m6 / g_fall * 100 if g_fall else np.nan)} ocurrió en sus últimos seis meses de vida."), unsafe_allow_html=True)
    if len(post):
        st.markdown(warn(f"Hay <b>{soles_short(post['gasto'].sum())}</b> facturados a {num(post['pid'].nunique())} pacientes en meses posteriores a su mes de deceso. "
                         "Puede ser rezago en el registro de la cuenta o un error en la fecha de deceso, pero conviene revisarlo (detalle al final de la pestaña)."),
                    unsafe_allow_html=True)

    # ------------------------------------------------------------------ estadío
    st.markdown("#### Estadío clínico")
    n_sd = int((pac["dx_principal"] == SD_GROUP).sum())
    c1, c2 = st.columns([2, 1.3])
    inc_sd = c2.checkbox(f"Incluir pacientes sin diagnóstico registrado ({num(n_sd)})", value=False, key=f"{name}_ds_sd",
                         help="Pacientes cuyas líneas no traen código CIE-10 (suelen ser evaluaciones con poco gasto). No tienen estadío y bajan la mortalidad observada.")
    pac_e = pac if inc_sd else pac[pac["dx_principal"] != SD_GROUP]
    dx_opts = ["Todos los diagnósticos"] + pac_e.groupby("dx_principal")["gasto"].sum().sort_values(ascending=False).index.tolist()
    dx_f = c1.selectbox("Diagnóstico principal", dx_opts, key=f"{name}_ds_dx",
                        help="El estadío solo es comparable dentro de un mismo tipo de cáncer. Elige uno para afinar estadío y supervivencia.")
    pp = pac_e if dx_f == dx_opts[0] else pac_e[pac_e["dx_principal"] == dx_f]
    if pp.empty:
        st.info("Sin pacientes con ese diagnóstico.")
        return
    t = pp.groupby("estadio").agg(pacientes=("pid", "size"), fallecidos=("fallecido", "sum"), gasto=("gasto", "sum"),
                                  meses=("meses_seg", "median"))
    t = t.reindex([e for e in ESTADIO_ORDER if e in t.index])
    t["pct_pac"] = t["pacientes"] / t["pacientes"].sum() * 100
    t["mortalidad"] = t["fallecidos"] / t["pacientes"] * 100
    t["gasto_pac"] = t["gasto"] / t["pacientes"]
    l, r = st.columns(2)
    with l:
        fig = go.Figure(go.Bar(x=t.index.astype(str), y=t["pacientes"], marker_color=[ESTADIO_COLOR.get(e, "#999") for e in t.index],
                               text=[f"{num(v)} ({pct(q)})" for v, q in zip(t["pacientes"], t["pct_pac"])], textposition="outside", cliponaxis=False,
                               hovertemplate="Estadío %{x}<br>%{y:,.0f} pacientes<extra></extra>"))
        style_fig(fig, 340, "Pacientes por estadío registrado")
        fig.update_yaxes(range=[0, t["pacientes"].max() * 1.2])
        st.plotly_chart(fig, **STRETCH)
    with r:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=t.index.astype(str), y=t["gasto_pac"], name="Gasto por paciente", marker_color=[ESTADIO_COLOR.get(e, "#999") for e in t.index],
                             hovertemplate="Estadío %{x}<br>S/ %{y:,.0f} por paciente<extra></extra>"))
        fig.add_trace(go.Scatter(x=t.index.astype(str), y=t["mortalidad"], name="Fallecidos (%)", yaxis="y2", mode="markers+lines+text",
                                 marker=dict(size=9, color="#4A4A4A"), line=dict(color="#4A4A4A", width=1.5, dash="dot"),
                                 text=[pct(v, 0) for v in t["mortalidad"]], textposition="top center",
                                 hovertemplate="Estadío %{x}<br>%{y:,.1f}% fallecidos<extra></extra>"))
        style_fig(fig, 340, "Gasto por paciente y % de fallecidos por estadío")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, ticksuffix="%", range=[0, 105], dtick=25),
                          hovermode="closest")
        fig.update_xaxes(tickangle=0)
        fig.update_layout(yaxis=dict(tickprefix="S/ ", tickformat=",.0f"))
        st.plotly_chart(fig, **STRETCH)
    sin_est = t.loc[[e for e in ("Desconocido", "Sin dato") if e in t.index], "pct_pac"].sum()
    kn = t[t.index.isin(CONOCIDOS)]
    txt = (f"{pct(sin_est)} de los pacientes no tiene un estadío útil registrado (desconocido o sin dato), lo que limita la comparación. ")
    if len(kn) >= 2:
        avz = kn.loc[kn.index.isin(["III", "IV"]), "pacientes"].sum() / kn["pacientes"].sum() * 100
        txt += (f"Entre quienes sí lo tienen, {pct(avz)} llega en estadío III o IV. ")
        if "IV" in kn.index and "I" in kn.index:
            txt += (f"La mortalidad observada va de {pct(kn.loc['I', 'mortalidad'])} en estadío I a {pct(kn.loc['IV', 'mortalidad'])} en estadío IV, "
                    f"y el gasto por paciente de {soles_short(kn.loc['I', 'gasto_pac'])} a {soles_short(kn.loc['IV', 'gasto_pac'])}. ")
    if dx_f == dx_opts[0]:
        txt += "Al mezclar todos los cánceres el estadío es solo orientativo; elige un diagnóstico para compararlo bien."
    st.markdown(interp(txt), unsafe_allow_html=True)
    sh = t.reset_index().rename(columns={"estadio": "Estadío", "pacientes": "Pacientes", "pct_pac": "% de pacientes", "fallecidos": "Fallecidos",
                                         "mortalidad": "% fallecidos", "gasto": "Gasto", "gasto_pac": "Gasto por paciente",
                                         "meses": "Meses de seguimiento (mediana)"})
    st.dataframe(fmt_df(sh, money=["Gasto", "Gasto por paciente"], ints=["Pacientes", "Fallecidos", "Meses de seguimiento (mediana)"],
                        pcts=["% de pacientes", "% fallecidos"]), hide_index=True, **STRETCH)

    # ------------------------------------------------------------------ supervivencia
    st.markdown("#### Supervivencia desde el primer mes financiado")
    c1, c2 = st.columns([2, 1])
    por = c1.radio("Comparar por", ["Estadío", "Diagnóstico principal", "Sexo", "Grupo etario", "Total"], horizontal=True, key=f"{name}_ds_km")
    excl = c2.checkbox("Omitir estadío desconocido o sin dato", value=True, key=f"{name}_ds_kmx", disabled=por != "Estadío")
    col = {"Estadío": "estadio", "Diagnóstico principal": "dx_principal", "Sexo": "sexo", "Grupo etario": "edad_grupo", "Total": None}[por]
    base = pp if por != "Diagnóstico principal" else pac_e
    if col == "estadio" and excl:
        base = base[base["estadio"].isin(CONOCIDOS)]
    if col is None:
        grupos = {"Todos": base}
    else:
        orden = [e for e in ESTADIO_ORDER if e in set(base[col])] if col == "estadio" else \
            base[col].value_counts().index.tolist()[:6]
        grupos = {str(g_): base[base[col] == g_] for g_ in orden if (base[col] == g_).sum() >= 5}
    if not grupos:
        st.info("Muy pocos pacientes por grupo (se requieren al menos 5).")
    else:
        fig = go.Figure()
        rows = []
        for i, (g_, x) in enumerate(grupos.items()):
            km = _km(x["meses_seg"].to_numpy(float), x["fallecido"].to_numpy(int))
            colr = ESTADIO_COLOR.get(g_, PALETTE[i % len(PALETTE)]) if col == "estadio" else PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(x=km["t"], y=km["s"] * 100, mode="lines", line=dict(shape="hv", width=2.4, color=colr),
                                     name=f"{g_} (n={len(x)})", hovertemplate=f"{g_}<br>Mes %{{x}}: %{{y:,.1f}}%<extra></extra>"))
            if len(grupos) <= 2:
                fig.add_trace(go.Scatter(x=np.r_[km["t"], km["t"][::-1]], y=np.r_[km["hi"], km["lo"][::-1]] * 100, fill="toself", line=dict(width=0, shape="hv"),
                                         fillcolor=colr, opacity=.15, showlegend=False, hoverinfo="skip"))
            md_ = _mediana(km)
            rows.append({"Grupo": g_, "Pacientes": len(x), "Fallecidos": int(x["fallecido"].sum()), "Sobrevida 12 meses": _s_at(km, 12),
                         "Sobrevida 24 meses": _s_at(km, 24), "Sobrevida 60 meses": _s_at(km, 60),
                         "Mediana de supervivencia": "No alcanzada" if np.isnan(md_) else f"{num(md_)} meses"})
        style_fig(fig, 420, f"Curva de Kaplan-Meier por {por.lower()}", "% vivos")
        fig.update_yaxes(range=[0, 101], ticksuffix="%")
        fig.update_xaxes(title_text="Meses desde el primer mes financiado")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, **STRETCH)
        kt = pd.DataFrame(rows)
        p_txt = ""
        if len(grupos) >= 2:
            try:
                from statsmodels.duration.survfunc import survdiff
                bb = pd.concat([x.assign(_g=g_) for g_, x in grupos.items()])
                _, pv = survdiff(bb["meses_seg"].to_numpy(float), bb["fallecido"].to_numpy(int), bb["_g"].to_numpy())
                p_txt = (f"La prueba log-rank {'muestra' if pv < o.alpha else 'no muestra'} diferencias entre las curvas "
                         f"(p {'< 0,001' if pv < 0.001 else '= ' + num(pv, 3)}). ")
            except Exception:  # noqa: BLE001
                p_txt = ""
        st.dataframe(fmt_df(kt, ints=["Pacientes", "Fallecidos"], pcts=["Sobrevida 12 meses", "Sobrevida 24 meses", "Sobrevida 60 meses"],
                            ), hide_index=True, **STRETCH)
        st.markdown(interp(
            "Cada curva muestra qué porcentaje de pacientes sigue vivo a medida que pasan los meses desde que la IAFAS empezó a financiar su atención en el INEN "
            f"(dentro de la serie). {p_txt}Quien no tiene fecha de deceso cuenta como vivo hasta {mes(corte)}. "
            "No es la supervivencia desde el diagnóstico: muchos pacientes ya se trataban antes de 2019 y otros pueden haber fallecido sin registro, por lo que conviene "
            "leerla como comparación entre grupos y no como cifra poblacional. La sobrevida a 60 meses solo aparece si hay seguimiento suficiente."), unsafe_allow_html=True)

    # ------------------------------------------------------------------ final de la vida
    st.markdown("#### Gasto al final de la vida")
    if dec.empty:
        st.info("No hay pacientes fallecidos en la selección.")
    else:
        ult = dec[(dec["meses_al_deceso"] >= 0) & (dec["meses_al_deceso"] <= 11)].copy()
        ult["m"] = ult["meses_al_deceso"].astype(int)
        w = ult.groupby(["m", "tipo"], observed=True)["gasto"].sum().unstack(fill_value=0).reindex(range(12), fill_value=0)
        npac = ult[ult["pid"] >= 0].groupby("m")["pid"].nunique().reindex(range(12), fill_value=0)
        lbl = ["Deceso" if m_ == 0 else f"−{m_}" for m_ in range(12)][::-1]
        fig = go.Figure()
        for tp in [c_ for c_ in ("MEDICAMENTO", "INSUMO", "PROCEDIMIENTO") if c_ in w.columns]:
            fig.add_trace(go.Bar(x=lbl, y=w[tp].values[::-1], name=TIPO_LABEL[tp], marker_color=TIPO_COLOR[tp],
                                 hovertemplate="%{x}<br>" + TIPO_LABEL[tp] + ": S/ %{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=lbl, y=npac.values[::-1], name="Pacientes con consumo", yaxis="y2", mode="lines+markers",
                                 line=dict(color="#4A4A4A", width=1.8), hovertemplate="%{x}<br>%{y} pacientes<extra></extra>"))
        style_fig(fig, 400, "Gasto de los pacientes fallecidos en sus últimos 12 meses", "S/")
        fig.update_layout(barmode="stack", hovermode="closest",
                          yaxis2=dict(overlaying="y", side="right", showgrid=False, rangemode="tozero", tickformat=",.0f", title_text="Pacientes"))
        fig.update_xaxes(title_text="Meses antes del deceso (0 = mes del deceso)", tickangle=0)
        fig.update_layout(yaxis=dict(tickprefix="S/ ", tickformat=",.0f"))
        st.plotly_chart(fig, **STRETCH)
        q = _es_quimio(dec)
        qm0 = dec[q & (dec["meses_al_deceso"] == 0)]["pid"].nunique()
        qm1 = dec[q & (dec["meses_al_deceso"] == 1)]["pid"].nunique()
        nf_obs = dec[dec["meses_al_deceso"] >= 0]["pid"].nunique()
        tot12 = w.values.sum()
        ult3 = w.loc[[0, 1, 2]].values.sum()
        st.markdown(interp(
            f"Muestra cómo se concentra el gasto en el último año de vida. Los últimos tres meses suman {pct(ult3 / tot12 * 100 if tot12 else np.nan)} "
            f"del gasto de ese año. <b>{num(qm0)} de {num(nf_obs)} pacientes fallecidos ({pct(qm0 / nf_obs * 100 if nf_obs else np.nan)})</b> "
            f"tuvieron quimioterapia facturada en el mes de su deceso y {num(qm1)} ({pct(qm1 / nf_obs * 100 if nf_obs else np.nan)}) en el mes anterior. "
            "La quimioterapia al final de la vida es un indicador clásico de atención agresiva; como la data solo tiene mes, el mes del deceso puede incluir "
            "hasta 30 días previos y no equivale exactamente al indicador de los últimos 14 o 30 días."), unsafe_allow_html=True)
        dd0 = dec[dec["meses_al_deceso"].isin([0, 1])]
        top = (dd0.groupby(["tipo", "descripcion"], observed=True).agg(gasto=("gasto", "sum"), pacientes=("pid", "nunique"))
               .reset_index().sort_values("gasto", ascending=False).head(15))
        top["tipo"] = top["tipo"].astype(str).map(TIPO_LABEL)
        with st.expander("Qué se facturó en el mes del deceso y el anterior (15 ítems de mayor gasto)"):
            st.dataframe(fmt_df(top.rename(columns={"tipo": "Tipo", "descripcion": "Ítem", "gasto": "Gasto", "pacientes": "Pacientes"}),
                                money=["Gasto"], ints=["Pacientes"]), hide_index=True, **STRETCH)

    # ------------------------------------------------------------------ después del deceso
    st.markdown("#### Consumo facturado después del mes de deceso")
    if post.empty:
        st.markdown(interp("No hay líneas facturadas en meses posteriores al fallecimiento del paciente."), unsafe_allow_html=True)
        return
    pm = post.assign(meses_despues=-post["meses_al_deceso"].astype(int))
    pm["rango"] = pd.cut(pm["meses_despues"], [0, 1, 3, 12, 1000], labels=["1 mes", "2 a 3 meses", "4 a 12 meses", "Más de 12 meses"])
    rr = pm.groupby("rango", observed=False).agg(gasto=("gasto", "sum"), pacientes=("pid", "nunique"), lineas=("lineas", "sum")).reset_index()
    tt = (pm.groupby(["tipo", "descripcion"], observed=True).agg(gasto=("gasto", "sum"), pacientes=("pid", "nunique"))
          .reset_index().sort_values("gasto", ascending=False).head(15))
    tt["tipo"] = tt["tipo"].astype(str).map(TIPO_LABEL)
    l, r = st.columns([1, 1.8])
    l.dataframe(fmt_df(rr.rename(columns={"rango": "Tiempo tras el deceso", "gasto": "Gasto", "pacientes": "Pacientes", "lineas": "Líneas"}),
                       money=["Gasto"], ints=["Pacientes", "Líneas"]), hide_index=True, **STRETCH)
    r.dataframe(fmt_df(tt.rename(columns={"tipo": "Tipo", "descripcion": "Ítem", "gasto": "Gasto", "pacientes": "Pacientes"}),
                       money=["Gasto"], ints=["Pacientes"]), hide_index=True, **STRETCH, height=260)
    uno = rr.loc[rr["rango"] == "1 mes", "gasto"].sum()
    st.markdown(interp(
        f"Son {soles_short(post['gasto'].sum())} en {num(post['pid'].nunique())} pacientes. Lo facturado un mes después ({soles_short(uno)}) suele explicarse "
        "por cuentas cerradas o registradas después del alta o por diferencias de fecha; lo facturado varios meses después merece revisión de la cuenta y de la "
        "fecha de deceso. Para identificar los casos se debe volver a la extracción original, porque el observatorio no guarda el DNI."), unsafe_allow_html=True)
