"""
app.py  Customer Lifetime Value

Design system: Financial Ledger / Broadsheet.
Soft salmon paper, warm ink, deep ledger green + brass accent, Fraunces display
serif over Spectral body serif, tabular figures, hairline rules, zero rounding.
BG/NBD + Gamma-Gamma probabilistic CLV on the UCI Online Retail dataset.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from lifetimes import BetaGeoFitter, GammaGammaFitter

warnings.filterwarnings("ignore")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.pkl")
MODEL_RF  = os.path.join(BASE_DIR, "models", "rf_model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens · financial ledger
# ─────────────────────────────────────────────────────────────────────────────
PAPER  = "#FBEFE4"   # soft salmon broadsheet ground
PAPER2 = "#F3E3D5"   # deeper panel ground
INK    = "#2A211B"   # warm dark brown-black
INK60  = "#7A6E62"   # muted ink
LINE   = "#E2D0BD"   # hairline on salmon
GREEN  = "#1C5440"   # deep ledger green · structural / positive / money
GREEN2 = "#2F7355"
BRASS  = "#A8792C"   # gold accent
CRIMSON = "#9E2B25"  # loss / negative, used sparingly

# Segments graded within the brand (value + health), no rainbow
SEGMENT_COLORS = {
    "Champions":        "#1C5440",
    "Loyal Customers":  "#2F7355",
    "Recent Customers": "#5C9A6F",
    "Promising":        "#93B58C",
    "Need Attention":   "#B89043",
    "At Risk":          "#C46A3A",
    "Can't Lose Them":  "#9E2B25",
    "Lost":             "#8C8073",
}
# Brand sequential scales (replace Purples)
GREEN_SCALE = [[0.0, "#F1E2D2"], [0.5, "#7BA98C"], [1.0, GREEN]]
BRASS_GREEN = [[0.0, "#E6C98A"], [0.5, "#8FB07C"], [1.0, GREEN]]

PLOTLY_TEMPLATE = go.layout.Template(layout=dict(
    xaxis=dict(gridcolor=LINE, zerolinecolor="#D8C2AC", linecolor=INK, automargin=True,
               tickfont=dict(family="Spectral, serif", size=12)),
    yaxis=dict(gridcolor=LINE, zerolinecolor="#D8C2AC", linecolor=INK, automargin=True,
               tickfont=dict(family="Spectral, serif", size=12)),
))

def layout(**kw):
    base = dict(
        plot_bgcolor=PAPER, paper_bgcolor=PAPER,
        font=dict(color=INK, family="Spectral, serif", size=13),
        colorway=[GREEN, BRASS, GREEN2, CRIMSON, "#8C8073", "#5C9A6F"],
        template=PLOTLY_TEMPLATE,
        title=dict(font=dict(family="Fraunces, serif", size=16, color=INK)),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    base.update(kw)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Page config + stylesheet
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CLV · Ledger",
    page_icon="£",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700;9..144,900&family=Spectral:wght@300;400;500;600;700&display=swap');

:root {{ --ink:{INK}; --paper:{PAPER}; --green:{GREEN}; --brass:{BRASS}; }}

html, body, [class*="css"], .stMarkdown, p, li, span, div {{
  font-family:'Spectral', Georgia, serif;
}}
.stApp {{ background:{PAPER}; color:{INK}; }}
.stApp *, [data-testid] * {{ border-radius:0 !important; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
.block-container {{ max-width:1180px; padding-top:2.6rem; padding-bottom:4rem; }}

/* tabular figures everywhere numbers matter */
.num, .kpi .v, [data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; }}

h1 {{ font-family:'Fraunces',serif; font-weight:600; letter-spacing:-0.015em; line-height:1.02;
  font-size:clamp(2.3rem,5vw,3.9rem); color:{INK}; margin:0.1rem 0 0.5rem 0; }}
h2 {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.7rem; color:{INK};
  margin:0.3rem 0 0.7rem 0; }}
h3 {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.15rem; color:{INK}; }}
h4 {{ font-family:'Spectral',serif; font-weight:600; text-transform:uppercase;
  letter-spacing:0.14em; font-size:0.72rem; color:{INK60}; }}

/* masthead */
.masthead {{ display:flex; align-items:baseline; justify-content:space-between;
  border-top:2px solid {INK}; border-bottom:2px solid {INK}; padding:0.4rem 0; margin-bottom:0.2rem; }}
.masthead .brand {{ font-family:'Fraunces',serif; font-weight:700; letter-spacing:0.02em;
  font-size:0.95rem; color:{INK}; }}
.masthead .meta {{ font-family:'Spectral',serif; font-size:0.72rem; text-transform:uppercase;
  letter-spacing:0.14em; color:{INK60}; }}

/* top nav */
div[data-testid="stRadio"] > label {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] {{
  display:flex; gap:0; border-bottom:1px solid {INK}; margin-bottom:1.6rem; flex-wrap:wrap; }}
div[data-testid="stRadio"] [role="radiogroup"] label {{ margin:0; padding:0.5rem 0; cursor:pointer; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div:last-child p {{
  font-family:'Fraunces',serif; font-weight:600; letter-spacing:0.02em; font-size:0.95rem;
  color:{INK60}; padding:0 1.6rem 0 0; margin:0; }}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) > div:last-child p {{
  color:{INK}; box-shadow:inset 0 -3px 0 0 {GREEN}; }}

.lede {{ font-family:'Spectral',serif; font-size:1.16rem; line-height:1.6; color:{INK};
  max-width:64ch; font-weight:400; }}
.lede b {{ font-weight:600; color:{GREEN}; }}
.lede i {{ color:{INK60}; }}

/* KPI ledger rail */
.kpis {{ display:grid; border:1px solid {INK}; border-right:none; margin:1.3rem 0; }}
.kpi {{ border-right:1px solid {INK}; padding:0.9rem 1.05rem; }}
.kpi .v {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.95rem; line-height:1.05;
  color:{INK}; }}
.kpi .v.green {{ color:{GREEN}; }} .kpi .v.brass {{ color:{BRASS}; }}
.kpi .k {{ font-family:'Spectral',serif; font-size:0.7rem; text-transform:uppercase;
  letter-spacing:0.12em; color:{INK60}; margin-top:0.4rem; }}
.kpi .s {{ font-family:'Spectral',serif; font-size:0.74rem; color:{INK60}; margin-top:0.15rem; }}

/* method columns */
.steps {{ display:grid; grid-template-columns:repeat(3,1fr); border:1px solid {INK};
  border-right:none; margin-top:0.4rem; }}
.step {{ border-right:1px solid {INK}; padding:1rem 1.1rem; }}
.step .n {{ font-family:'Fraunces',serif; font-weight:700; font-size:1.6rem; color:{BRASS}; }}
.step .t {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.0rem; margin:0.2rem 0 0.5rem 0; }}
.step .d {{ font-size:0.88rem; color:{INK60}; line-height:1.55; }}
.step .d b {{ color:{INK}; font-weight:600; }}

/* segment cards */
.segcard {{ border:1px solid {INK}; border-left-width:5px; padding:0.7rem 0.9rem; margin-bottom:0.6rem;
  background:{PAPER}; }}
.segcard .nm {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.0rem; }}
.segcard .ct {{ font-family:'Fraunces',serif; font-weight:600; font-size:1.5rem; }}
.segcard .mt {{ font-family:'Spectral',serif; font-size:0.76rem; color:{INK60}; }}

/* blocks + callout */
.blk {{ border:1px solid {INK}; padding:1rem 1.1rem; background:{PAPER}; }}
.blk.fill {{ background:{PAPER2}; }}
.callout {{ border-left:3px solid {BRASS}; padding:0.6rem 0.95rem; margin:0.9rem 0;
  font-size:0.95rem; background:{PAPER2}; }}
.callout.green {{ border-left-color:{GREEN}; }}
.callout.loss {{ border-left-color:{CRIMSON}; }}

.find {{ font-size:1.0rem; line-height:1.65; }}
.find b {{ font-weight:600; color:{GREEN}; }}

/* buttons */
.stButton > button, .stFormSubmitButton > button {{
  border:1px solid {GREEN}; background:{GREEN}; color:{PAPER};
  font-family:'Fraunces',serif; font-weight:600; letter-spacing:0.02em; font-size:0.95rem;
  padding:0.5rem 1.4rem; }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
  background:{INK}; border-color:{INK}; color:{PAPER}; }}

hr {{ border:none; border-top:1px solid {LINE}; margin:1.6rem 0; }}
[data-testid="stDataFrame"] {{ border:1px solid {INK}; }}
[data-testid="stMetricValue"] {{ font-family:'Fraunces',serif; }}

textarea, input, .stNumberInput div[data-baseweb="input"],
.stSelectbox div[data-baseweb="select"] > div {{
  border:1px solid {INK} !important; background:{PAPER} !important; color:{INK} !important; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{GREEN} !important; }}
::placeholder {{ color:{INK60} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable blocks
# ─────────────────────────────────────────────────────────────────────────────
def masthead(meta):
    st.markdown(
        f"""<div class="masthead">
        <span class="brand">The Customer Ledger</span>
        <span class="meta">BG/NBD + Gamma-Gamma · {meta['n_customers_rfm']:,} customers ·
        £{meta['total_clv_12m']:,.0f} CLV / 12m</span></div>""",
        unsafe_allow_html=True)


def rule(left, right=""):
    st.markdown(
        f"""<div style="display:flex; align-items:baseline; justify-content:space-between;
        border-top:1px solid {INK}; padding-top:0.45rem; margin:2.1rem 0 1rem 0;">
        <span style="font-family:'Fraunces',serif; font-weight:600; font-size:1.35rem;">{left}</span>
        <span style="font-family:'Spectral',serif; font-size:0.72rem; text-transform:uppercase;
        letter-spacing:0.12em; color:{INK60};">{right}</span></div>""",
        unsafe_allow_html=True)


def callout(text, kind=""):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def chart(fig):
    st.plotly_chart(fig, use_container_width=True, theme=None)


@st.cache_resource
def load_meta():
    return joblib.load(META_PATH)


@st.cache_resource
def load_models(meta):
    bgf = BetaGeoFitter(); bgf.params_ = meta["bgf_params"]
    gg = GammaGammaFitter(); gg.params_ = meta["gg_params"]
    rf = joblib.load(MODEL_RF)
    return bgf, gg, rf


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
try:
    meta = load_meta()
    bgf, gg, rf = load_models(meta)
    rfm = meta["rfm"].set_index("CustomerID") if "CustomerID" in meta["rfm"].columns else meta["rfm"]
except Exception as e:
    st.error(f"Could not load models. Run `python src/train.py` first.\n\n`{e}`")
    st.stop()

seg_stats   = meta["segment_stats"]
monthly_rev = meta["monthly_revenue"]
country_rev = meta["country_revenue"]

masthead(meta)
NAV = ["Ledger", "Segments", "Forecast", "Simulator"]
page = st.radio("nav", NAV, horizontal=True, label_visibility="collapsed")


# ═════════════════════════════════════════════════════════════════════════════
# LEDGER · the brief + the book
# ═════════════════════════════════════════════════════════════════════════════
if page == "Ledger":
    st.markdown("<h4>The statement</h4>", unsafe_allow_html=True)
    st.markdown("# What is each customer worth?")
    st.markdown(
        '<p class="lede">Most churn models tell you <i>who is leaving</i>. This one puts a number '
        'on it: <b>how much revenue each customer will generate over their lifetime</b>, using a '
        'probabilistic BG/NBD + Gamma-Gamma model on real UK e-commerce transactions. Value, not '
        'just risk, drives the segmentation.</p>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpis" style="grid-template-columns:repeat(5,1fr);">
      <div class="kpi"><div class="v">{meta['n_customers_rfm']:,}</div><div class="k">Customers modeled</div></div>
      <div class="kpi"><div class="v">{meta['n_transactions']:,}</div><div class="k">Transactions</div></div>
      <div class="kpi"><div class="v green">£{meta['total_clv_12m']:,.0f}</div><div class="k">Projected CLV · 12m</div></div>
      <div class="kpi"><div class="v">£{meta['median_clv_12m']:,.0f}</div><div class="k">Median CLV / customer</div></div>
      <div class="kpi"><div class="v brass">{meta['avg_p_alive']:.0%}</div><div class="k">Avg P(alive)</div></div>
    </div>""", unsafe_allow_html=True)

    # The book · monthly revenue + countries
    rule("The book", "UCI Online Retail · 2010–2011")
    cl, cr = st.columns([2, 1])
    with cl:
        st.markdown("<h4>Monthly revenue (£)</h4>", unsafe_allow_html=True)
        fig = px.area(monthly_rev, x="date", y="revenue")
        fig.update_traces(line=dict(color=GREEN, width=2), fillcolor="rgba(28,84,64,0.12)")
        fig.update_layout(**layout(height=300, xaxis_title="", yaxis_title="Revenue (£)"))
        chart(fig)
    with cr:
        st.markdown("<h4>Top 10 markets</h4>", unsafe_allow_html=True)
        top10 = country_rev.head(10)
        fig2 = px.bar(top10, x="Revenue", y="Country", orientation="h",
                      color="Revenue", color_continuous_scale=GREEN_SCALE)
        fig2.update_layout(**layout(height=300, coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed", title=""), xaxis_title="Revenue (£)"))
        chart(fig2)

    rule("Cohort retention", "% still purchasing N months on")
    cohort = meta["cohort_retention"].iloc[:12, :13].copy()
    cohort.index = cohort.index.astype(str)
    cohort.columns = [f"M+{c}" for c in cohort.columns]
    cz = cohort.values * 100
    ctext = np.where(np.isnan(cz), "", np.vectorize(lambda v: "" if np.isnan(v) else f"{v:.0f}")(cz))
    fig3 = go.Figure(data=go.Heatmap(
        z=cz, x=cohort.columns.tolist(), y=cohort.index.tolist(),
        colorscale=GREEN_SCALE, text=ctext, texttemplate="%{text}",
        textfont=dict(family="Spectral", size=11),
        hoverongaps=False,
        hovertemplate="Cohort %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Ret. %")))
    fig3.update_layout(**layout(height=360, xaxis_title="Months since first purchase",
                       yaxis_title="Acquisition cohort"))
    chart(fig3)

    # Method
    rule("Method", "three models, one number")
    st.markdown(f"""
    <div class="steps">
      <div class="step"><div class="n">1</div><div class="t">RFM features</div>
        <div class="d">Every customer reduced to three behaviours: <b>Recency</b> (days since last
        order), <b>Frequency</b> (repeat purchases), <b>Monetary</b> (average order value). These
        feed the models below.</div></div>
      <div class="step"><div class="n">2</div><div class="t">BG/NBD</div>
        <div class="d">Beta-Geometric / Negative Binomial. Models how often a customer buys while
        active and when they silently <b>drop out</b> · giving expected future purchases and
        <b>P(still alive)</b>.</div></div>
      <div class="step"><div class="n">3</div><div class="t">Gamma-Gamma</div>
        <div class="d">Conditional on being alive, models the <b>monetary value</b> of each future
        order. Combined: <b>CLV = purchases × spend × horizon</b>, discounted.</div></div>
    </div>""", unsafe_allow_html=True)

    # Findings
    rule("What the ledger says", "key findings")
    _rfm = meta["rfm"].copy()
    _clv_sorted = _rfm["clv_12m"].sort_values(ascending=False)
    _top20_n = max(1, int(len(_clv_sorted) * 0.20))
    _top20_share = _clv_sorted.iloc[:_top20_n].sum() / _clv_sorted.sum()
    _hv_at_risk = int(((_rfm["clv_12m"] > _rfm["clv_12m"].median()) & (_rfm["p_alive"] < 0.5)).sum())
    st.markdown(f"""<div class="find">
    <p>① The <b>top 20% of customers carry {_top20_share:.0%} of projected 12-month CLV</b>. The
    model lets you act on that concentration before those accounts churn.</p>
    <p>② <b>{_hv_at_risk:,} customers</b> sit above median CLV but below 50% P(alive): valuable and
    disengaging, the highest-priority retention targets.</p>
    <p>③ Because the framework yields <b>probability distributions</b>, not point estimates, it
    supports confidence-aware "what-if" decisions, explored in the Simulator.</p></div>""",
    unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SEGMENTS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Segments":
    st.markdown("<h4>The accounts, sorted</h4>", unsafe_allow_html=True)
    st.markdown("# RFM segmentation")
    st.markdown('<p class="lede">Customers grouped by Recency, Frequency and Monetary value, each '
                'segment carrying its own book of projected revenue and its own play.</p>',
                unsafe_allow_html=True)

    rule("Segment summary", "ranked by total CLV")
    seg_sorted = seg_stats.sort_values("total_clv_12m", ascending=False).reset_index(drop=True)
    cols = st.columns(4)
    for i, row in seg_sorted.iterrows():
        c = SEGMENT_COLORS.get(row["segment"], "#8C8073")
        cols[i % 4].markdown(
            f"""<div class="segcard" style="border-left-color:{c};">
            <div class="nm" style="color:{c};">{row['segment']}</div>
            <div class="ct num">{int(row['customers']):,}<span style="font-family:Spectral;
            font-size:0.8rem;font-weight:400;"> customers</span></div>
            <div class="mt">Avg CLV £{row['avg_clv_12m']:,.0f} · P(alive) {row['avg_p_alive']:.0%}</div>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        rule("Headcount & value", "")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        ss = seg_stats.sort_values("customers", ascending=True)
        fig.add_trace(go.Bar(y=ss["segment"], x=ss["customers"], orientation="h", name="Customers",
                      marker_color=[SEGMENT_COLORS.get(s, "#8C8073") for s in ss["segment"]]),
                      secondary_y=False)
        fig.add_trace(go.Scatter(y=ss["segment"], x=ss["avg_clv_12m"], mode="markers+lines",
                      name="Avg CLV 12m", line=dict(color=INK, width=1.5),
                      marker=dict(size=9, color=BRASS, line=dict(width=1.5, color=INK))),
                      secondary_y=True)
        fig.update_layout(**layout(height=400, legend=dict(orientation="h", y=-0.16),
                          margin=dict(l=10, r=10, t=10, b=10)))
        fig.update_yaxes(title_text="Avg CLV (£)", secondary_y=True)
        chart(fig)
    with c2:
        rule("Revenue potential", "size = total CLV")
        fig2 = px.treemap(seg_stats, path=["segment"], values="total_clv_12m", color="avg_p_alive",
                          color_continuous_scale=BRASS_GREEN)
        fig2.update_traces(texttemplate="<b>%{label}</b><br>£%{value:,.0f}",
                           hovertemplate="<b>%{label}</b><br>Total CLV £%{value:,.0f}<br>"
                           "P(alive) %{color:.0%}<extra></extra>",
                           marker=dict(line=dict(color=PAPER, width=2)))
        fig2.update_layout(**layout(height=400, margin=dict(l=10, r=10, t=10, b=10)))
        chart(fig2)

    rule("RFM space", "frequency vs order value")
    st.markdown('<p class="lede">Bubble size = projected CLV. Colour = segment.</p>',
                unsafe_allow_html=True)
    sample = rfm.sample(min(1500, len(rfm)), random_state=42).reset_index()
    fig3 = px.scatter(sample, x="frequency", y="monetary_value",
                      size=np.clip(sample["clv_12m"], 1, sample["clv_12m"].quantile(0.95)),
                      color="segment", color_discrete_map=SEGMENT_COLORS, opacity=0.78,
                      hover_data={"p_alive": ":.1%", "clv_12m": ":.0f", "recency": True},
                      labels={"frequency": "Purchase frequency", "monetary_value": "Avg order value (£)"})
    fig3.update_layout(**layout(height=440, legend=dict(orientation="h", y=-0.22),
                       margin=dict(l=10, r=10, t=10, b=10)))
    chart(fig3)


# ═════════════════════════════════════════════════════════════════════════════
# FORECAST
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":
    st.markdown("<h4>The projection</h4>", unsafe_allow_html=True)
    st.markdown("# CLV forecast")
    st.markdown('<p class="lede">Projected lifetime value at 6 and 12 month horizons, and where '
                'the high-value, still-active accounts sit.</p>', unsafe_allow_html=True)

    g = (meta['total_clv_12m'] / meta['total_clv_6m'] - 1) * 100
    st.markdown(f"""
    <div class="kpis" style="grid-template-columns:repeat(3,1fr);">
      <div class="kpi"><div class="v">£{meta['total_clv_6m']:,.0f}</div><div class="k">Total CLV · 6 months</div></div>
      <div class="kpi"><div class="v green">£{meta['total_clv_12m']:,.0f}</div><div class="k">Total CLV · 12 months</div></div>
      <div class="kpi"><div class="v brass">+£{(meta['total_clv_12m']-meta['total_clv_6m']):,.0f}</div>
        <div class="k">Growth 6m → 12m</div><div class="s">+{g:.0f}%</div></div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        rule("CLV distribution", "12 months")
        clv_vals = rfm["clv_12m"].clip(upper=rfm["clv_12m"].quantile(0.97))
        fig = px.histogram(clv_vals, nbins=60, color_discrete_sequence=[GREEN])
        fig.add_vline(x=rfm["clv_12m"].median(), line_dash="dash", line_color=CRIMSON,
                      annotation_text=f"median £{rfm['clv_12m'].median():.0f}",
                      annotation_position="top right")
        fig.update_layout(**layout(height=320, showlegend=False, xaxis_title="CLV 12m (£)",
                          yaxis_title="Customers"))
        chart(fig)
    with c2:
        rule("6m vs 12m", "by segment")
        seg_6m = rfm.groupby("segment")["clv_6m"].mean().reset_index()
        seg_6m.columns = ["segment", "clv_6m"]
        sm = seg_stats[["segment", "avg_clv_12m"]].merge(seg_6m, on="segment").sort_values("avg_clv_12m")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=sm["segment"], x=sm["clv_6m"], name="6 months",
                       orientation="h", marker_color=BRASS))
        fig2.add_trace(go.Bar(y=sm["segment"], x=sm["avg_clv_12m"], name="12 months",
                       orientation="h", marker_color=GREEN))
        fig2.update_layout(**layout(barmode="group", height=320, xaxis_title="Avg CLV (£)",
                           legend=dict(orientation="h", y=-0.2)))
        chart(fig2)

    rule("Value vs engagement", "where to spend retention")
    st.markdown('<p class="lede">Top-right: high-value customers still active · <b>protect</b>. '
                'Top-left: high-value but disengaging · <b>reactivate</b>.</p>', unsafe_allow_html=True)
    sample = rfm.sample(min(2000, len(rfm)), random_state=1).reset_index()
    fig3 = px.scatter(sample, x="p_alive", y="clv_12m", color="segment",
                      color_discrete_map=SEGMENT_COLORS, opacity=0.7,
                      labels={"p_alive": "P(alive)", "clv_12m": "CLV 12m (£)"},
                      hover_data={"frequency": True, "monetary_value": ":.0f"})
    fig3.add_vline(x=0.5, line_dash="dot", line_color="rgba(42,33,27,0.3)")
    fig3.add_hline(y=rfm["clv_12m"].median(), line_dash="dot", line_color="rgba(42,33,27,0.3)")
    fig3.add_annotation(x=0.9, y=rfm["clv_12m"].quantile(0.9), text="PROTECT", showarrow=False,
                        font=dict(family="Fraunces", color=GREEN, size=13))
    fig3.add_annotation(x=0.1, y=rfm["clv_12m"].quantile(0.9), text="REACTIVATE", showarrow=False,
                        font=dict(family="Fraunces", color=CRIMSON, size=13))
    fig3.update_layout(**layout(height=440, legend=dict(orientation="h", y=-0.22)))
    chart(fig3)

    with st.expander("ML baseline · Random Forest regressor"):
        st.markdown(f"""As a benchmark, a Random Forest predicts CLV 12m from RFM features:
        **MAE £{meta['rf_metrics']['mae']:,.0f}**, **R² {meta['rf_metrics']['r2']:.3f}**.""")
        callout("The R² is near-tautological: the RF features (frequency, recency, T, monetary, "
                "RFM scores) all derive from the same transactions used to build the CLV target, so "
                "the high score is mechanical correlation, not independent validation. The BG/NBD + "
                "Gamma-Gamma approach is preferred: interpretable, grounded in purchase theory, and "
                "it yields full probability distributions. Treat the RF as a sanity check only.", "loss")


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Simulator":
    st.markdown("<h4>Price one account</h4>", unsafe_allow_html=True)
    st.markdown("# CLV simulator")
    st.markdown('<p class="lede">Enter a purchase history, get the projected lifetime value, the '
                'probability the customer is still active, and the segment with its play.</p>',
                unsafe_allow_html=True)

    cform, out = st.columns([1, 1])
    with cform:
        with st.form("sim"):
            st.markdown("<h4>Customer profile</h4>", unsafe_allow_html=True)
            frequency = st.slider("Repeat purchases (frequency)", 1, 100, 10)
            recency = st.slider("Recency · days between first and last order", 1, 730, 60)
            T = st.slider("Customer age T · days observed", int(recency) + 1, 730,
                          max(int(recency) + 10, 180))
            monetary = st.number_input("Avg order value (£)", 1.0, 5000.0, 75.0, step=5.0)
            horizon = st.selectbox("Forecast horizon (months)", [3, 6, 12, 24], index=1)
            submitted = st.form_submit_button("Predict CLV", use_container_width=True)

    with out:
        if submitted:
            p_alive = float(np.asarray(bgf.conditional_probability_alive(
                float(frequency), float(recency), float(T))).reshape(-1)[0])
            exp_purch = float(np.asarray(bgf.conditional_expected_number_of_purchases_up_to_time(
                horizon * 30, float(frequency), float(recency), float(T))).reshape(-1)[0])
            avg_profit = float(np.asarray(gg.conditional_expected_average_profit(
                pd.Series([float(frequency)]), pd.Series([float(monetary)]))).reshape(-1)[0])

            discount_rate = 0.01
            expected_cum = []
            for m in range(1, int(horizon) + 1):
                cum = bgf.conditional_expected_number_of_purchases_up_to_time(
                    m * 30, float(frequency), float(recency), float(T))
                expected_cum.append(float(np.asarray(cum).reshape(-1)[0]))
            expected_by_month = np.diff([0.0] + expected_cum)
            clv = float(np.sum((expected_by_month * avg_profit) /
                        np.power(1.0 + discount_rate, np.arange(1, len(expected_by_month) + 1))))

            rfm_ref = meta["rfm"]
            r_edges = rfm_ref["recency"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            f_edges = rfm_ref["frequency"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            m_edges = rfm_ref["monetary_value"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy()
            r_score = int(5 - np.digitize(recency, r_edges))
            f_score = int(1 + np.digitize(frequency, f_edges))
            m_score = int(1 + np.digitize(monetary, m_edges))

            def _seg(r, f, m):
                if r >= 4 and f >= 4: return "Champions"
                if r >= 3 and f >= 3: return "Loyal Customers"
                if r >= 4 and f <= 2: return "Recent Customers"
                if r >= 3 and f <= 2: return "Promising"
                if r <= 2 and f >= 4: return "Can't Lose Them"
                if r <= 2 and f >= 3: return "At Risk"
                if r <= 2 and f <= 2 and m <= 2: return "Lost"
                return "Need Attention"
            seg = _seg(r_score, f_score, m_score)
            c = SEGMENT_COLORS.get(seg, "#8C8073")

            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p_alive * 100,
                number={"suffix": "%", "font": {"size": 30, "family": "Fraunces"}},
                gauge={"axis": {"range": [0, 100], "tickcolor": INK}, "bar": {"color": c},
                       "bgcolor": PAPER, "borderwidth": 1, "bordercolor": INK,
                       "steps": [{"range": [0, 30], "color": "rgba(158,43,37,0.16)"},
                                 {"range": [30, 70], "color": "rgba(168,121,44,0.16)"},
                                 {"range": [70, 100], "color": "rgba(28,84,64,0.16)"}],
                       "threshold": {"line": {"color": INK, "width": 2}, "value": 50}}))
            fig.update_layout(height=220, margin=dict(l=20, r=20, t=14, b=0),
                              paper_bgcolor=PAPER, font=dict(color=INK))
            st.markdown("<h4>P(alive)</h4>", unsafe_allow_html=True)
            chart(fig)

            cc1, cc2 = st.columns(2)
            cc1.markdown(f'<div class="blk"><div class="num" style="font-family:Fraunces;'
                         f'font-size:1.7rem;font-weight:600;color:{GREEN};">£{clv:,.0f}</div>'
                         f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;'
                         f'color:{INK60};">CLV · {horizon}m</div></div>', unsafe_allow_html=True)
            cc2.markdown(f'<div class="blk"><div class="num" style="font-family:Fraunces;'
                         f'font-size:1.7rem;font-weight:600;">{exp_purch:.1f}</div>'
                         f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.1em;'
                         f'color:{INK60};">Expected purchases</div></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="segcard" style="border-left-color:{c};margin-top:0.8rem;">'
                        f'<div class="nm" style="color:{c};">Segment · {seg}</div>'
                        f'<div class="mt">RFM scores · R {r_score} · F {f_score} · M {m_score}</div>'
                        f'</div>', unsafe_allow_html=True)

            actions = {
                "Champions": "Reward them. Your best customers · loyalty perks, early access.",
                "Loyal Customers": "Upsell higher-value products. They respond to recommendations.",
                "Recent Customers": "Nurture the relationship · onboarding, engagement offers.",
                "Promising": "Build frequency · incentivise the second and third purchase.",
                "Need Attention": "Reconnect before they slip · targeted re-engagement offer.",
                "At Risk": "Send a win-back now · discount plus a personalised message.",
                "Can't Lose Them": "High value but disengaging · escalate to account-manager level.",
                "Lost": "Low ROI on retention spend · put the budget elsewhere.",
            }
            callout(f"<b>Play:</b> {actions.get(seg, 'Evaluate on customer context.')}",
                    "green" if seg in ("Champions", "Loyal Customers", "Recent Customers", "Promising")
                    else "loss")
        else:
            st.markdown(f'<div class="blk fill" style="height:300px;display:flex;align-items:center;'
                        f'justify-content:center;color:{INK60};">Fill in the profile and press '
                        f'<b style="color:{GREEN};">&nbsp;Predict CLV</b></div>', unsafe_allow_html=True)
