"""ui.py: constantes visuales, formatos y helpers de gráficos compartidos por el dashboard."""

from __future__ import annotations

import colorsys
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stats_analysis import smooth_series

try:
    _V = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:  # noqa: BLE001
    _V = (1, 50)
# Streamlit >= 1.50 usa width="stretch"; versiones anteriores usan use_container_width=True
STRETCH = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}

PRIMARY = "#2C5F8A"
SECONDARY = "#D9542F"
ACCENT = "#1F4E78"
OK = "#2F9E44"
WARN = "#E8A33D"
BAD = "#C94141"
PALETTE = [PRIMARY, SECONDARY, OK, "#8850C4", "#C9A13B", "#5C6AC4", "#7D7D7D", "#4AA8C9", "#B5446E", "#3F8F7B"]
MACRO_COLOR = {"Ambulatorio": PRIMARY, "Emergencia": BAD, "Hospitalización": SECONDARY, "Centro quirúrgico": "#8850C4",
               "Sin procedimiento asociado": "#B0B0B0", "No clínico": "#7D7D7D"}
TIPO_COLOR = {"MEDICAMENTO": PRIMARY, "INSUMO": OK, "PROCEDIMIENTO": SECONDARY}

CSS = f"""
<style>
    .main .block-container {{ padding-top: 1.2rem; max-width: 1450px; }}
    h1, h2, h3 {{ color: {ACCENT}; }}
    .app-header {{ background: linear-gradient(90deg, {ACCENT} 0%, {PRIMARY} 100%); padding: 1.1rem 1.6rem;
        border-radius: 10px; color: white; margin-bottom: 1rem; }}
    .app-header h1 {{ color: white; margin: 0; font-size: 1.55rem; font-weight: 800; letter-spacing: .01em; line-height: 1.2; }}
    .app-header p {{ color: #e8eef5; margin: .25rem 0 0 0; font-size: .93rem; }}
    div[data-testid="stMetric"] {{ background: #f7f8fb; border: 1px solid #e2e6ee; border-radius: 10px; padding: .55rem .85rem; }}
    div[data-testid="stMetric"] label p {{ font-size: .8rem; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
    .interp-box {{ background: #f4f7fb; border-left: 4px solid {PRIMARY}; border-radius: 6px;
        padding: .7rem 1rem; margin: .6rem 0 1rem 0; font-size: .92rem; color: #333; }}
    .warn-box {{ background: #fff8ec; border-left: 4px solid {WARN}; border-radius: 6px;
        padding: .7rem 1rem; margin: .6rem 0 1rem 0; font-size: .92rem; color: #4a3b12; }}
    .reco-card {{ background: #fff; border: 1px solid #e2e6ee; border-radius: 8px; padding: .7rem 1rem; margin: .5rem 0; font-size: .92rem; color: #333; }}
    .reco-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: .25rem; font-size: 1rem; color: {ACCENT}; }}
    .reco-tag {{ font-size: .75rem; font-weight: 600; border-radius: 10px; padding: .1rem .6rem; }}
    .mini-title {{ font-size: .95rem; font-weight: 600; color: {ACCENT}; margin-bottom: .2rem; }}
    .source-note {{ font-size: 0.8rem; color: #666; margin-top: 0.4rem; }}
    .citation-box {{ background: #fafbfc; border: 1px solid #e2e6ee; border-radius: 8px;
                      padding: 1rem 1.2rem; margin-top: 1.2rem; font-size: 0.85rem; color: #333; }}
    .citation-box h4 {{ margin: 0 0 0.5rem 0; font-size: 0.8rem; letter-spacing: 0.04em; color: {ACCENT}; }}
    .citation-box p {{ margin: 0 0 0.6rem 0; line-height: 1.5; }}
</style>
"""


@dataclass
class Opts:
    """Opciones globales de la barra lateral que consumen los paneles."""
    date_lo: pd.Timestamp
    date_hi: pd.Timestamp
    chart_type: str = "Línea"
    markers: bool = True
    labels: bool = False
    lowess: bool = False
    log: bool = False
    alpha: float = 0.05
    umbral_alto_costo: float = 50000.0
    primas: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["iafas", "anio", "prima_anual"]))
    excl_ultimo_mes: bool = False
    afiliados: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["iafas", "anio", "afiliados"]))


NBSP = "\u00a0"


def num(v, dec: int = 0) -> str:
    """Número con miles separados por espacio y decimales con coma: 1 234,5."""
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", NBSP)


def pval(p, tabla: bool = False) -> str:
    """Valor p con tres decimales; si redondea a 0,000 se muestra como «< 0,001».
    En texto devuelve el punto decimal (la capa de localización lo convierte a coma); en tablas, ya con coma."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 0.0005:
        return "< 0,001" if tabla else "< 0.001"
    return num(p, 3) if tabla else f"{p:.3f}"


def ptxt(p) -> str:
    """Para frases: «p = 0,071» o «p < 0,001»."""
    s = pval(p)
    return f"p {s}" if s.startswith("<") else f"p = {s}"


_RX_NUM = re.compile(r"(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w,])|(?<![\w.,])\d+\.\d+(?![\w]|\.\d)")


def _conv(m) -> str:
    t = m.group(0)
    return t.replace(",", "\x00").replace(".", ",").replace("\x00", NBSP)


def es_text(t):
    """Convierte 1,234.5 a 1 234,5 en el texto visible (sin tocar etiquetas HTML)."""
    if not isinstance(t, str):
        return t
    if "<style" in t:
        return t
    t = t.replace("(s/)", "(S/)")
    parts = re.split(r"(<[^>]*>)", t)
    return "".join(p if p.startswith("<") else _RX_NUM.sub(_conv, p) for p in parts)


def soles(v: float, dec: int = 0) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return "S/ " + num(v, dec)


MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def mes(ts) -> str:
    """Mes abreviado en español: Ene-2019."""
    if ts is None or pd.isna(ts):
        return "—"
    ts = pd.Timestamp(ts)
    return f"{MESES_ES[ts.month - 1]}-{ts.year}"


def es_dates(fig: go.Figure) -> go.Figure:
    """Ejes de fecha con meses en español (plotly solo trae inglés): marcas calculadas a mano."""
    xs = []
    for tr in fig.data:
        x = getattr(tr, "x", None)
        if x is None or len(x) == 0:
            continue
        v0 = x[0]
        if isinstance(v0, (pd.Timestamp, np.datetime64)) or hasattr(v0, "year") and hasattr(v0, "month") and not isinstance(v0, str):
            try:
                xs.append(pd.to_datetime(pd.Series(list(x))))
            except Exception:
                pass
    if not xs:
        return fig
    lo, hi = min(s.min() for s in xs), max(s.max() for s in xs)
    months = (hi.year - lo.year) * 12 + hi.month - lo.month + 1
    step = 12 if months > 60 else 6 if months > 30 else 3 if months > 14 else 1
    ticks = [t for t in pd.date_range(lo.to_period("M").to_timestamp(), hi, freq="MS") if (t.month - 1) % step == 0]
    if not ticks:
        return fig
    txt = [str(t.year) if step == 12 else mes(t) for t in ticks]
    fig.update_xaxes(tickmode="array", tickvals=ticks, ticktext=txt, hoverformat="%Y-%m")
    return fig


def soles_short(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    a = abs(v)
    if a >= 1e6:
        return "S/ " + num(v / 1e6, 1) + " M"
    if a >= 1e3:
        return "S/ " + num(v / 1e3, 1) + " mil"
    return "S/ " + num(v, 0)


def pct(v: float, dec: int = 1) -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return num(v, dec) + "%"


def fmt_df(df: pd.DataFrame, money=(), ints=(), pcts=(), decs=(), dates=()):
    f = {}
    for c in money:
        if c in df.columns:
            f[c] = lambda v: "S/ " + num(v, 0)
    for c in ints:
        if c in df.columns:
            f[c] = lambda v: num(v, 0)
    for c in pcts:
        if c in df.columns:
            f[c] = lambda v: num(v, 1) + "%"
    for c in decs:
        if c in df.columns:
            f[c] = lambda v: num(v, 1)
    for c in dates:
        if c in df.columns:
            f[c] = mes
    return df.style.format(f, na_rep="—")


def hex_to_rgba(h: str, a: float) -> str:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def shades_of(hex_color: str, n: int, l_min: float = 0.30, l_max: float = 0.78) -> list[str]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    lights = [l_min] if n <= 1 else np.linspace(l_min, l_max, n)
    out = []
    for lig in lights:
        rr, gg, bb = colorsys.hls_to_rgb(hh, float(lig), ss)
        out.append("#%02x%02x%02x" % (int(round(rr * 255)), int(round(gg * 255)), int(round(bb * 255))))
    return out


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None, ytitle: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=48 if title else 18, b=70),
        title=dict(text=title or "", font=dict(size=14, color=ACCENT)),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
    )
    # título de ejes explícito (cadena vacía) para que ninguna versión de plotly.js escriba "undefined"
    fig.update_xaxes(showgrid=False, linecolor="#ccc", title_text="")
    fig.update_yaxes(gridcolor="#eceff4", zeroline=False, title_text=ytitle or "")
    return fig


def time_figure(
    wide: pd.DataFrame, o: Opts, *, title: str | None = None, ytitle: str | None = None,
    money: bool = True, stacked: bool = False, colors: dict | None = None, height: int = 400,
    allow_lowess: bool = True, force_kind: str | None = None,
) -> go.Figure:
    """Gráfico temporal multi-serie respetando las opciones globales (línea/barras,
    marcadores, etiquetas, LOWESS, escala log)."""
    fig = go.Figure()
    kind = force_kind or ("bar" if o.chart_type == "Barras" else "line")
    fmt_lbl = "%{y:,.0f}"
    for i, c in enumerate(wide.columns):
        col = (colors or {}).get(c, PALETTE[i % len(PALETTE)])
        y = wide[c].values
        if kind == "bar":
            fig.add_trace(go.Bar(x=wide.index, y=y, name=str(c), marker_color=col,
                                 text=y if o.labels else None, texttemplate=fmt_lbl if o.labels else None))
        else:
            fig.add_trace(go.Scatter(
                x=wide.index, y=y, name=str(c), mode="lines+markers" if o.markers else "lines",
                line=dict(width=2, color=col), marker=dict(size=5),
                stackgroup="s" if stacked else None,
                text=y if o.labels else None, texttemplate=fmt_lbl if o.labels else None, textposition="top center",
            ))
        if o.lowess and allow_lowess and not stacked and len(wide.columns) <= 3:
            sm = smooth_series(np.nan_to_num(y.astype(float)))
            if sm is not None:
                fig.add_trace(go.Scatter(x=wide.index, y=sm, name=f"{c} (LOWESS)", mode="lines",
                                         line=dict(width=2.5, color=col, dash="dot"), opacity=.8))
    if kind == "bar":
        fig.update_layout(barmode="stack" if stacked else "group")
    style_fig(fig, height, title, ytitle)
    if o.log:
        fig.update_yaxes(type="log")
    if money:
        fig.update_yaxes(tickprefix="S/ ", tickformat=",.0f")
    return fig


def interp(html: str) -> str:
    return f'<div class="interp-box">{html}</div>'


def warn(html: str) -> str:
    return f'<div class="warn-box">{html}</div>'


# --- parches de presentación: meses en español y números con espacio de miles y coma decimal ---
def es_fig(fig: go.Figure) -> go.Figure:
    es_dates(fig)
    fig.update_layout(separators=", ")
    for tr in fig.data:
        for attr in ("text", "name"):
            v = getattr(tr, attr, None)
            if isinstance(v, str):
                setattr(tr, attr, es_text(v))
            elif v is not None and attr == "text":
                try:
                    setattr(tr, attr, [es_text(x) for x in v])
                except Exception:
                    pass
    for ax in fig.select_xaxes():
        if ax.ticktext is not None:
            ax.ticktext = [es_text(x) for x in ax.ticktext]
    for ax in fig.select_yaxes():
        if ax.ticktext is not None:
            ax.ticktext = [es_text(x) for x in ax.ticktext]
    return fig


_orig_plotly_chart = st.plotly_chart


def _plotly_chart_es(fig, *a, **k):
    try:
        es_fig(fig)
    except Exception:
        pass
    return _orig_plotly_chart(fig, *a, **k)


st.plotly_chart = _plotly_chart_es


def _wrap_text(orig, method=False):
    def w(*a, **k):
        i = 1 if method else 0
        a = tuple(es_text(x) if j == i else x for j, x in enumerate(a))
        if "body" in k:
            k["body"] = es_text(k["body"])
        return orig(*a, **k)
    return w


def _wrap_metric(orig, method=False):
    def w(*a, **k):
        i = 1 if method else 0
        a = list(a)
        for j in (i, i + 1, i + 2):          # label, value, delta
            if j < len(a) and isinstance(a[j], str):
                a[j] = es_text(a[j])
        for key in ("label", "value", "delta", "help"):
            if isinstance(k.get(key), str):
                k[key] = es_text(k[key])
        return orig(*a, **k)
    return w


from streamlit.delta_generator import DeltaGenerator as _DG  # noqa: E402

for _n in ("markdown", "caption", "info", "warning", "success", "error"):
    _o = getattr(_DG, _n)
    if not getattr(_o, "_es_patched", False):
        _w = _wrap_text(_o, True)
        _w._es_patched = True
        setattr(_DG, _n, _w)
        _m = getattr(st, _n, None)
        if _m is not None:
            _w2 = _wrap_text(_m, False)
            _w2._es_patched = True
            setattr(st, _n, _w2)
_om = _DG.metric
if not getattr(_om, "_es_patched", False):
    _wm = _wrap_metric(_om, True)
    _wm._es_patched = True
    _DG.metric = _wm
    _wm2 = _wrap_metric(st.metric, False)
    _wm2._es_patched = True
    st.metric = _wm2
