"""
proyecciones_comp.py
--------------------
Métricas especiales de la pestaña Proyecciones: proyección por componente.

  - Medicamentos: por defecto los de la lista nominal de alto costo (no PNUME), por principio activo.
  - Insumos y procedimientos: por defecto los de mayor gasto.
  - Pacientes por diagnóstico: pacientes activos mensuales de cada grupo de diagnóstico.

Método por serie (se elige solo y se informa en la tabla):
  - Holt con tendencia amortiguada si la serie tiene consumo en la mayoría de los meses. La banda se obtiene
    simulando 1 000 trayectorias futuras con residuos remuestreados (statsmodels `simulate`), de modo que la banda
    de la suma de 12 meses sale de las trayectorias y no de sumar percentiles mes a mes.
  - Cada serie empieza en su primer mes con consumo, para que un medicamento o diagnóstico nuevo no arrastre años en cero.
    Si desde entonces hay menos de 12 meses se usa la tasa reciente (probabilidad de consumo × monto medio).
  - Croston corregido (SBA, Syntetos-Boylan) si más del 30% de los meses están en cero (demanda intermitente,
    típica de un medicamento de alto costo que usan pocos pacientes). La banda simula meses con o sin consumo
    (probabilidad estimada) y montos remuestreados de los meses con consumo.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data_processing import SD_GROUP
from ui import BAD, OK, PRIMARY, SECONDARY, STRETCH, WARN, Opts, fmt_df, hex_to_rgba, interp, mes, num, pct, soles_short, warn

COVID_LO, COVID_HI = pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-01")
INTERMITENTE = 0.30          # proporción de meses en cero a partir de la cual se usa SBA
N_SIM = 1000
MAX_PANELS = 12
MET_LABEL = {"gasto": "Gasto (S/)", "cantidad": "Cantidad (unidades)", "pacientes": "Pacientes activos"}


# ---------------------------------------------------------------------------
# Series y métodos
# ---------------------------------------------------------------------------
def wide_series(d: pd.DataFrame, by: str, cats: list, metric: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Una columna por categoría con la serie mensual completa (ceros en meses sin consumo)."""
    dd = d[d[by].isin(cats)]
    if metric == "pacientes":
        dd = dd[dd["pid"] >= 0]
        g = dd.groupby(["tiempo", by], observed=True)["pid"].nunique()
    else:
        g = dd.groupby(["tiempo", by], observed=True)[metric].sum()
    w = g.unstack(by).reindex(idx).fillna(0.0) if len(g) else pd.DataFrame(index=idx)
    w = w.reindex(columns=cats, fill_value=0.0)
    w.columns = [str(c) for c in w.columns]
    return w.astype(float)


def total_series(d: pd.DataFrame, by: str, cats: list, metric: str, idx: pd.DatetimeIndex) -> pd.Series:
    """Serie de la selección completa (para pacientes cuenta personas distintas, no suma columnas)."""
    dd = d[d[by].isin(cats)]
    if metric == "pacientes":
        dd = dd[dd["pid"] >= 0]
        s = dd.groupby("tiempo")["pid"].nunique()
    else:
        s = dd.groupby("tiempo")[metric].sum()
    return s.reindex(idx).fillna(0.0).astype(float)


def prep(s: pd.Series, n_train: int | None, excl_covid: bool) -> pd.Series:
    if excl_covid:
        s = s[~((s.index >= COVID_LO) & (s.index <= COVID_HI))]
    if n_train:
        s = s.iloc[-n_train:]
    return s


def _sba(y: np.ndarray, alpha: float = 0.15) -> tuple[float, float, float]:
    """Croston con corrección SBA sobre una serie que empieza en su primer consumo.
    Devuelve (pronóstico mensual, tamaño medio suavizado, probabilidad de consumo en un mes)."""
    z, p, q = y[0], 1.0, 1
    for t in range(1, len(y)):
        if y[t] > 0:
            z = z + alpha * (y[t] - z)
            p = p + alpha * (q - p)
            q = 1
        else:
            q += 1
    p = max(p, 1.0, q * 0.5) if q > p else max(p, 1.0)   # una racha final larga sin consumo reduce la probabilidad
    return (1 - alpha / 2) * z / p, z, 1 / p


@st.cache_data(show_spinner=False, max_entries=512)
def forecast_one(vals: tuple, horizon: int, ci: float, winsor: bool, seed: int = 42) -> dict | None:
    y0 = np.asarray(vals, dtype=float)
    if (y0 > 0).sum() < 2:
        return None
    y = y0[int(np.flatnonzero(y0 > 0)[0]):]      # la serie empieza en el primer consumo (ítem o diagnóstico nuevo)
    n = len(y)
    nz = int((y > 0).sum())
    if n < 6:
        return None
    zero_share = 1 - nz / n
    a = (1 - ci) / 2
    rng = np.random.default_rng(seed)
    H = min(12, horizon)
    out = {"zero_share": zero_share, "n": n}
    if n < 12:
        prob, size = nz / n, float(y[y > 0].mean())
        sims = (rng.random((N_SIM, horizon)) < prob) * rng.choice(y[y > 0], size=(N_SIM, horizon), replace=True)
        out.update(metodo="Tasa reciente (serie corta)", fc=np.full(horizon, prob * size), prob=prob, size=size)
    elif zero_share > INTERMITENTE:
        fc, size, prob = _sba(y)
        sizes = y[y > 0]
        sims = (rng.random((N_SIM, horizon)) < prob) * rng.choice(sizes, size=(N_SIM, horizon), replace=True)
        out.update(metodo="Intermitente (Croston SBA)", fc=np.full(horizon, fc), prob=prob, size=size)
    else:
        yy = np.minimum(y, np.nanpercentile(y, 95)) if winsor else y
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = ExponentialSmoothing(yy, trend="add", damped_trend=True, initialization_method="estimated").fit(optimized=True)
                fc = np.asarray(fit.forecast(horizon))
                try:
                    sims = np.asarray(fit.simulate(horizon, repetitions=N_SIM, error="add", random_errors="bootstrap", anchor="end",
                                                   random_state=seed)).T
                except Exception:  # noqa: BLE001
                    sd = np.nanstd(np.asarray(fit.resid))
                    sims = fc + rng.normal(0, sd, (N_SIM, horizon)) * np.sqrt(np.arange(1, horizon + 1))
            out.update(metodo="Holt amortiguado", fc=np.clip(fc, 0, None), prob=nz / n, size=float(y[y > 0].mean()))
        except Exception:  # noqa: BLE001
            m = float(np.mean(y[-12:]))
            sims = rng.choice(y[-24:], size=(N_SIM, horizon), replace=True)
            out.update(metodo="Media de 12 meses", fc=np.full(horizon, m), prob=nz / n, size=float(y[y > 0].mean()))
    sims = np.clip(sims, 0, None)
    out["lo"] = np.percentile(sims, 100 * a, axis=0)
    out["hi"] = np.percentile(sims, 100 * (1 - a), axis=0)
    sums = sims[:, :H].sum(axis=1)
    out["sum_fc"] = float(out["fc"][:H].sum())
    out["sum_lo"], out["sum_hi"] = float(np.percentile(sums, 100 * a)), float(np.percentile(sums, 100 * (1 - a)))
    out["obs_H"] = float(y0[-H:].sum())
    out["cons_H"] = float(np.mean(y0[-3:]) * H)      # escenario conservador: media de los últimos 3 meses
    out["H"] = H
    return out


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------
def _fmt(v: float, metric: str) -> str:
    return soles_short(v) if metric == "gasto" else num(v, 0)


def _render_block(d: pd.DataFrame, by: str, cats: list, metric: str, idx: pd.DatetimeIndex, prm: dict, key: str,
                  titulo_total: str, nombre: str) -> None:
    if not cats:
        st.info("No hay componentes seleccionados.")
        return
    horizon, ci, n_train, excl_covid, winsor = prm["horizon"], prm["ci"], prm["n_train"], prm["excl_covid"], prm["winsor"]
    W = wide_series(d, by, cats, metric, idx)
    tot = prep(total_series(d, by, cats, metric, idx), n_train, excl_covid)
    pac = metric == "pacientes"
    with st.spinner("Proyectando componentes..."):
        res = {c: forecast_one(tuple(prep(W[c], n_train, excl_covid).values), horizon, ci, winsor) for c in W.columns}
        rt = forecast_one(tuple(tot.values), horizon, ci, winsor)
    H = min(12, horizon)
    per = f"{H} meses" if H != 12 else "12 meses"
    fut_idx = pd.date_range(tot.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS") if len(tot) else None

    # --- total de la selección ---
    if rt is not None:
        div = H if pac else 1
        var = (rt["sum_fc"] / rt["obs_H"] - 1) * 100 if rt["obs_H"] else np.nan
        k = st.columns(4)
        lbl_obs = f"Últimos {per} (promedio mensual)" if pac else f"Últimos {per} observados"
        k[0].metric(lbl_obs, _fmt(rt["obs_H"] / div, metric))
        k[1].metric(f"Próximos {per} · {titulo_total}", _fmt(rt["sum_fc"] / div, metric), f"{var:+.1f}%" if var == var else None)
        k[2].metric(f"Banda {int(ci * 100)}%", f"{_fmt(rt['sum_lo'] / div, metric)} – {_fmt(rt['sum_hi'] / div, metric)}")
        k[3].metric("Método del total", rt["metodo"].split(" (")[0])
    else:
        st.info("La serie total de la selección es demasiado corta o casi sin consumo para proyectar (mínimo 12 meses y 3 con consumo).")

    # --- tabla por componente ---
    rows, long = [], []
    for c in W.columns:
        r = res[c]
        hist = W[c]
        base = {"Componente": c, f"Gasto total (S/)": float(d.loc[d[by].astype(str) == c, "gasto"].sum()),
                "Meses con consumo (%)": float((hist > 0).mean() * 100)}
        if r is None:
            rows.append({**base, "Método": "Sin datos suficientes"})
            continue
        div = r["H"] if pac else 1
        rows.append({**base, "Método": r["metodo"],
                     ("Promedio mensual últimos " if pac else "Últimos ") + per: r["obs_H"] / div,
                     ("Promedio mensual próximos " if pac else "Próximos ") + per: r["sum_fc"] / div,
                     "Variación (%)": (r["sum_fc"] / r["obs_H"] - 1) * 100 if r["obs_H"] else np.nan,
                     "Conservador (media 3 meses)": r["cons_H"] / div,
                     "Banda inferior": r["sum_lo"] / div, "Banda superior": r["sum_hi"] / div,
                     "Prob. de consumo en un mes (%)": r["prob"] * 100})
        for j, t in enumerate(fut_idx):
            long.append({"componente": c, "mes": t.strftime("%Y-%m"), "proyeccion": r["fc"][j], "limite_inferior": r["lo"][j],
                         "limite_superior": r["hi"][j], "metodo": r["metodo"]})
    tb = pd.DataFrame(rows)
    for c_ in tb.columns:
        if c_ not in ("Componente", "Método"):
            tb[c_] = pd.to_numeric(tb[c_], errors="coerce")
    sin = tb.loc[tb["Método"] == "Sin datos suficientes", "Componente"].tolist()
    tb = tb[tb["Método"] != "Sin datos suficientes"].reset_index(drop=True)
    if len(tb):
        pct_cols = ["Meses con consumo (%)", "Variación (%)", "Prob. de consumo en un mes (%)"]
        show = tb.copy()
        for c_ in show.columns:
            if c_ in ("Componente", "Método"):
                continue
            if c_ in pct_cols:
                show[c_] = [("—" if v != v else num(v, 1) + "%") for v in tb[c_]]
            elif c_ == "Gasto total (S/)" or metric == "gasto":
                show[c_] = [("—" if v != v else "S/ " + num(v, 0)) for v in tb[c_]]
            else:
                show[c_] = [("—" if v != v else num(v, 1 if pac else 0)) for v in tb[c_]]
        st.dataframe(show, hide_index=True, **STRETCH)
        st.caption("Variación «—»: sin consumo en los últimos meses, no hay base para calcular el porcentaje. Banda inferior y superior: intervalo de la suma "
                   f"de los próximos {per} según la simulación.")
    if sin:
        st.caption(f"Sin datos suficientes para proyectar (menos de 2 meses con consumo o menos de 6 meses desde el primero): {', '.join(sin)}.")

    # --- paneles con histórico reciente, proyección y banda ---
    ok_cats = [c for c in W.columns if res[c] is not None][:MAX_PANELS]
    if ok_cats:
        ncol = 3 if len(ok_cats) > 4 else 2 if len(ok_cats) > 1 else 1
        nrow = int(np.ceil(len(ok_cats) / ncol))
        titles = [(c if len(c) <= 34 else c[:33] + "…") + f"<br><span style='font-size:10px'>{res[c]['metodo']}</span>" for c in ok_cats]
        fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=titles, vertical_spacing=min(0.12, 0.5 / max(nrow, 1)), horizontal_spacing=0.06)
        for i, c in enumerate(ok_cats):
            r, rr, cc = res[c], i // ncol + 1, i % ncol + 1
            h = prep(W[c], None, excl_covid).iloc[-36:]
            fig.add_trace(go.Scatter(x=h.index, y=h.values, mode="lines", line=dict(color=PRIMARY, width=1.4), showlegend=False,
                                     hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra></extra>"), row=rr, col=cc)
            fig.add_trace(go.Scatter(x=list(fut_idx) + list(fut_idx[::-1]), y=list(r["hi"]) + list(r["lo"][::-1]), fill="toself",
                                     fillcolor=hex_to_rgba(SECONDARY, .15), line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"),
                          row=rr, col=cc)
            fig.add_trace(go.Scatter(x=fut_idx, y=r["fc"], mode="lines", line=dict(color=SECONDARY, width=2.4), showlegend=False,
                                     hovertemplate="%{x|%Y-%m}: %{y:,.0f}<extra>proyección</extra>"), row=rr, col=cc)
            top = max(float(np.nanmax(h.values)) if len(h) else 0, float(np.nanmax(r["hi"])), float(np.nanmax(r["fc"]))) * 1.12
            fig.update_yaxes(range=[0, top if top > 0 else 1], row=rr, col=cc)
        fig.update_annotations(font_size=11)
        st.markdown(f"<div class='mini-title'>{MET_LABEL[metric]}: últimos 36 meses (azul) y proyección a {horizon} meses (naranja, con banda)</div>",
                    unsafe_allow_html=True)
        fig.update_layout(height=max(260, 215 * nrow + 40), margin=dict(l=10, r=10, t=45, b=20), plot_bgcolor="white", paper_bgcolor="white",
                          hovermode="closest")
        fig.update_xaxes(showgrid=False, linecolor="#ccc", tickfont=dict(size=9))
        fig.update_yaxes(gridcolor="#eceff4", tickfont=dict(size=9), tickprefix="S/ " if metric == "gasto" else "", tickformat="~s")
        st.plotly_chart(fig, **STRETCH)
        if len([c for c in W.columns if res[c] is not None]) > MAX_PANELS:
            st.caption(f"Se grafican los primeros {MAX_PANELS}; la tabla y la descarga incluyen todos.")

    # --- lectura ---
    tv = tb.dropna(subset=["Variación (%)"]) if "Variación (%)" in tb.columns else pd.DataFrame()
    n_int = int(tb["Método"].astype(str).str.startswith("Intermitente").sum())
    txt = ""
    if rt is not None and rt["obs_H"]:
        txt += (f"Para el conjunto seleccionado de {nombre} se proyecta <b>{_fmt(rt['sum_fc'] / (H if pac else 1), metric)}</b> "
                f"{'de pacientes activos por mes en promedio' if pac else ''} en los próximos {per}, "
                f"<b>{(rt['sum_fc'] / rt['obs_H'] - 1) * 100:+.1f}%</b> frente a los últimos {per}. ")
    if len(tv):
        up = tv.sort_values("Variación (%)", ascending=False).iloc[0]
        txt += f"El mayor aumento esperado es <b>{up['Componente']}</b> ({up['Variación (%)']:+.1f}%). "
    if n_int:
        txt += (f"{n_int} de {len(tb)} series son intermitentes (más del {int(INTERMITENTE * 100)}% de los meses sin consumo): para ellas se proyecta "
                "un promedio mensual esperado (probabilidad de consumo × monto típico), no la fecha en que ocurrirá el próximo consumo. "
                "Su banda suele ser ancha y empezar en cero, porque basta que entre o salga un paciente para cambiar el gasto.")
    if txt:
        st.markdown(interp(txt), unsafe_allow_html=True)
    if rt is not None and rt["obs_H"] and abs(rt["sum_fc"] / rt["obs_H"] - 1) > 0.5:
        st.markdown(warn(f"<b>Cambio proyectado mayor a ±50%.</b> El modelo prolonga un cambio reciente fuerte de la serie. Contrasta con la columna «Conservador», "
                         f"que repite el nivel de los últimos 3 meses ({_fmt(rt['cons_H'] / (H if pac else 1), metric)}{' por mes' if pac else ''}), "
                         "y con la ventana de entrenamiento de arriba antes de usar la cifra."), unsafe_allow_html=True)
    if long:
        st.download_button("Descargar proyección por componente (CSV)", pd.DataFrame(long).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"proyeccion_componentes_{key}.csv", mime="text/csv", key=f"pc_{key}_csv")


def panel_componentes(d: pd.DataFrame, name: str, idx: pd.DatetimeIndex, o: Opts) -> None:
    ss = st.session_state
    prm = {
        "horizon": int(ss.get(f"{name}_p_h", 12)),
        "ci": float(ss.get(f"{name}_p_ci", 0.90)),
        "n_train": {"Toda la serie": None, "Últimos 36 meses": 36, "Últimos 24 meses": 24, "Últimos 12 meses": 12}[ss.get(f"{name}_p_w", "Toda la serie")],
        "excl_covid": bool(ss.get(f"{name}_p_covid", False)),
        "winsor": bool(ss.get(f"{name}_p_wins", True)),
    }
    st.divider()
    st.markdown("#### 📐 Métricas especiales: proyección por componente")
    st.caption(f"Usa el horizonte ({prm['horizon']} meses), la ventana de entrenamiento, la banda ({int(prm['ci'] * 100)}%) y las opciones de pandemia y picos "
               "definidas arriba. El método se elige por serie: Holt amortiguado si hay consumo en la mayoría de los meses, Croston SBA si es intermitente.")
    t = st.tabs(["💊 Medicamentos", "🧰 Insumos", "🔬 Procedimientos", "🩺 Pacientes por diagnóstico"])

    # --- Medicamentos ---
    with t[0]:
        dm = d[d["tipo"] == "MEDICAMENTO"]
        if dm.empty:
            st.info("Sin consumo de medicamentos en la selección.")
        else:
            c = st.columns([1.6, 1.1, 1.1, 0.9])
            modo = c[0].radio("Selección", ["Alto costo (lista nominal no PNUME)", "Mayor gasto"], key=f"{name}_pc_mm", horizontal=True)
            nivel = c[1].selectbox("Agrupar por", ["Principio activo", "Presentación"], key=f"{name}_pc_mn")
            met = c[2].selectbox("Métrica", ["gasto", "cantidad", "pacientes"], format_func=MET_LABEL.get, key=f"{name}_pc_mmet")
            topn = c[3].number_input("Máx. componentes", 3, 40, 12, 1, key=f"{name}_pc_mt")
            by = "principio_activo" if nivel == "Principio activo" else "descripcion"
            if modo.startswith("Alto"):
                base = dm[~dm["alto_costo"].astype(str).isin(["", "nan"])]
                if base.empty:
                    st.info("No hay medicamentos de la lista de alto costo en la selección.")
                    base = None
            else:
                base = dm
            if base is not None:
                rank = base.groupby(by, observed=True)["gasto"].sum().sort_values(ascending=False)
                rank = rank[rank > 0]
                opts_ = [str(x) for x in rank.index]
                sel = st.multiselect("Componentes a proyectar", opts_, default=opts_[:int(topn)], key=f"{name}_pc_msel_{modo[:4]}_{by}")
                if modo.startswith("Alto"):
                    st.caption(f"{len(opts_)} {'moléculas' if by == 'principio_activo' else 'presentaciones'} de la lista con consumo; se marcan por código SIGA y misma molécula.")
                _render_block(base, by, sel, met, idx, prm, f"{name}_med", "total de la selección",
                              "medicamentos de alto costo" if modo.startswith("Alto") else "medicamentos")

    # --- Insumos y procedimientos ---
    for tab, tipo, nm in ((t[1], "INSUMO", "insumos"), (t[2], "PROCEDIMIENTO", "procedimientos")):
        with tab:
            dt = d[d["tipo"] == tipo]
            if dt.empty:
                st.info(f"Sin consumo de {nm} en la selección.")
                continue
            c = st.columns([1.2, 1, 3])
            met = c[0].selectbox("Métrica", ["gasto", "cantidad", "pacientes"], format_func=MET_LABEL.get, key=f"{name}_pc_{tipo}_met")
            topn = c[1].number_input("Top por gasto", 3, 40, 10, 1, key=f"{name}_pc_{tipo}_n")
            rank = dt.groupby("descripcion", observed=True)["gasto"].sum().sort_values(ascending=False)
            opts_ = [str(x) for x in rank[rank > 0].index]
            sel = c[2].multiselect(f"{nm.capitalize()} a proyectar", opts_, default=opts_[:int(topn)], key=f"{name}_pc_{tipo}_sel_{int(topn)}")
            _render_block(dt, "descripcion", sel, met, idx, prm, f"{name}_{tipo.lower()}", "total de la selección", nm)

    # --- Pacientes por diagnóstico ---
    with t[3]:
        c = st.columns([1.3, 1, 1])
        nivel = c[0].selectbox("Nivel", ["Grupo de diagnóstico", "Diagnóstico CIE-10"], key=f"{name}_pc_dxn")
        excl_sd = c[1].checkbox(f"Excluir «{SD_GROUP}»", value=True, key=f"{name}_pc_dxsd")
        topn = c[2].number_input("Máx. diagnósticos", 3, 60, 60 if nivel.startswith("Grupo") else 15, 1, key=f"{name}_pc_dxt_{nivel[:4]}")
        by = "dx_grupo" if nivel.startswith("Grupo") else "dx_etiqueta"
        dd = d[d["pid"] >= 0]
        if excl_sd:
            dd = dd[dd["dx_grupo"].astype(str) != SD_GROUP]
        if dd.empty:
            st.info("Sin pacientes con diagnóstico en la selección.")
        else:
            rank = dd.groupby(by, observed=True)["pid"].nunique().sort_values(ascending=False)
            opts_ = [str(x) for x in rank[rank > 0].index]
            sel = st.multiselect("Diagnósticos a proyectar", opts_, default=opts_[:int(topn)], key=f"{name}_pc_dxsel_{by}_{int(topn)}")
            st.caption("Pacientes activos = pacientes distintos con al menos una línea facturada con ese diagnóstico en el mes. Un paciente con dos diagnósticos cuenta en ambos, "
                       "así que el total de la selección (personas distintas) es menor que la suma de las filas.")
            _render_block(dd, by, sel, "pacientes", idx, prm, f"{name}_dx", "pacientes activos", "diagnósticos")
