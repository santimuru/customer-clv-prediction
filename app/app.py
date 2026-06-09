"""
app.py  Customer Lifetime Value

Design system: Modern BI dashboard.
Cool light ground, white rounded cards with soft shadows, bold colour-coded
numerals (green money / blue accent), Manrope display over Inter body, pill nav.
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
# Design tokens · modern BI dashboard
# ─────────────────────────────────────────────────────────────────────────────
PAPER  = "#EEF1F6"   # cool light dashboard ground
PAPER2 = "#FFFFFF"   # white card
INK    = "#161A20"   # near-black
INK60  = "#5C6672"   # muted slate
LINE   = "#E2E7EE"   # hairline
GREEN  = "#1E9E63"   # money / positive
GREEN2 = "#15B179"
BRASS  = "#2F6BE0"   # BI blue · primary accent (name kept for section markup)
CRIMSON = "#E0483D"  # loss / negative
AMBER  = "#E0A11E"   # caution

# Segments graded by value + health: green (best) -> blue -> amber -> red -> grey
SEGMENT_COLORS = {
    "Champions":        "#16915C",
    "Loyal Customers":  "#3FB07E",
    "Recent Customers": "#2F6BE0",
    "Promising":        "#6FA0EF",
    "Need Attention":   "#E0A11E",
    "At Risk":          "#E0742E",
    "Can't Lose Them":  "#E0483D",
    "Lost":             "#8A93A0",
}
# Sequential scales
GREEN_SCALE = [[0.0, "#E3EFE9"], [0.5, "#6FC79E"], [1.0, GREEN]]
BRASS_GREEN = [[0.0, "#F0D9A8"], [0.5, "#7FC3A0"], [1.0, GREEN]]

PLOTLY_TEMPLATE = go.layout.Template(layout=dict(
    xaxis=dict(gridcolor=LINE, zerolinecolor="#D3DAE3", linecolor="#C9D1DC", automargin=True,
               tickfont=dict(family="Inter, sans-serif", size=12, color=INK60)),
    yaxis=dict(gridcolor=LINE, zerolinecolor="#D3DAE3", linecolor="#C9D1DC", automargin=True,
               tickfont=dict(family="Inter, sans-serif", size=12, color=INK60)),
))

def layout(**kw):
    base = dict(
        plot_bgcolor=PAPER2, paper_bgcolor=PAPER2,
        font=dict(color=INK, family="Inter, sans-serif", size=13),
        colorway=[BRASS, GREEN, AMBER, CRIMSON, "#8A93A0", GREEN2],
        template=PLOTLY_TEMPLATE,
        title=dict(font=dict(family="Manrope, sans-serif", size=15, color=INK)),
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
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {{ --ink:{INK}; --paper:{PAPER}; --green:{GREEN}; --accent:{BRASS}; }}

html, body, [class*="css"], .stMarkdown, p, li, span, div {{ font-family:'Inter', system-ui, sans-serif; }}
.stApp {{ background:{PAPER}; color:{INK}; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display:none !important; }}
.block-container {{ max-width:1200px; padding-top:2.4rem; padding-bottom:4rem; }}

.num, .kpi .v, [data-testid="stMetricValue"] {{ font-variant-numeric:tabular-nums; }}

h1 {{ font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-0.03em; line-height:1.04;
  font-size:clamp(2.2rem,4.6vw,3.4rem); color:{INK}; margin:0.2rem 0 0.5rem; }}
h2 {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:1.6rem; color:{INK}; margin:0.3rem 0 0.7rem; }}
h3 {{ font-family:'Manrope',sans-serif; font-weight:700; font-size:1.1rem; color:{INK}; }}
h4 {{ font-family:'Inter',sans-serif; font-weight:700; text-transform:uppercase; letter-spacing:0.10em;
  font-size:0.72rem; color:{INK60}; }}

/* masthead */
.masthead {{ display:flex; align-items:center; justify-content:space-between; padding:0 0 1.2rem; }}
.masthead .brand {{ font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-0.01em;
  font-size:1.05rem; color:{INK}; display:flex; align-items:center; gap:.55rem; }}
.masthead .brand::before {{ content:''; width:13px; height:13px; background:{BRASS}; border-radius:4px; }}
.masthead .meta {{ font-size:0.76rem; color:{INK60}; font-weight:500; }}

/* top nav · pill tabs */
div[data-testid="stRadio"] > label {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] {{
  display:flex; gap:.35rem; margin:0 0 1.8rem; flex-wrap:wrap;
  background:{PAPER2}; padding:.35rem; border-radius:12px; border:1px solid {LINE};
  box-shadow:0 1px 3px rgba(20,30,50,.05); width:fit-content; }}
div[data-testid="stRadio"] [role="radiogroup"] label {{ margin:0; cursor:pointer; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{ display:none; }}
div[data-testid="stRadio"] [role="radiogroup"] label > div:last-child p {{
  font-family:'Inter',sans-serif; font-weight:600; font-size:0.85rem; color:{INK60};
  padding:.4rem 1.1rem; margin:0; border-radius:8px; }}
div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) > div:last-child p {{
  color:#fff; background:{BRASS}; }}

.lede {{ font-size:1.12rem; line-height:1.6; color:#3A414C; max-width:64ch; font-weight:400; }}
.lede b {{ font-weight:700; color:{GREEN}; }}
.lede i {{ color:{INK60}; font-style:normal; }}

/* KPI cards */
.kpis {{ display:grid; gap:.9rem; margin:1.4rem 0; }}
.kpi {{ background:{PAPER2}; border:1px solid {LINE}; border-radius:14px; padding:1.1rem 1.2rem;
  box-shadow:0 1px 3px rgba(20,30,50,.06); }}
.kpi .v {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:2.1rem; line-height:1.05;
  color:{INK}; letter-spacing:-0.02em; }}
.kpi .v.green {{ color:{GREEN}; }} .kpi .v.brass {{ color:{BRASS}; }}
.kpi .k {{ font-size:0.7rem; text-transform:uppercase; letter-spacing:0.07em; font-weight:600;
  color:{INK60}; margin-top:0.5rem; }}
.kpi .s {{ font-size:0.74rem; color:{INK60}; margin-top:0.2rem; }}

/* method cards */
.steps {{ display:grid; grid-template-columns:repeat(3,1fr); gap:.9rem; margin-top:0.4rem; }}
.step {{ background:{PAPER2}; border:1px solid {LINE}; border-radius:14px; padding:1.1rem 1.2rem;
  box-shadow:0 1px 3px rgba(20,30,50,.06); }}
.step .n {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:1.5rem; color:{BRASS}; }}
.step .t {{ font-family:'Manrope',sans-serif; font-weight:700; font-size:1.02rem; margin:0.2rem 0 0.5rem; }}
.step .d {{ font-size:0.88rem; color:{INK60}; line-height:1.6; }}
.step .d b {{ color:{INK}; font-weight:600; }}

/* segment cards */
.segcard {{ background:{PAPER2}; border:1px solid {LINE}; border-left:5px solid {INK};
  border-radius:12px; padding:0.8rem 1rem; margin-bottom:0.6rem; box-shadow:0 1px 3px rgba(20,30,50,.06); }}
.segcard .nm {{ font-family:'Manrope',sans-serif; font-weight:700; font-size:0.98rem; }}
.segcard .ct {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:1.5rem; }}
.segcard .mt {{ font-size:0.78rem; color:{INK60}; }}

/* blocks + callout */
.blk {{ background:{PAPER2}; border:1px solid {LINE}; border-radius:12px; padding:1rem 1.2rem;
  box-shadow:0 1px 3px rgba(20,30,50,.06); }}
.blk.fill {{ background:{PAPER}; box-shadow:none; }}
.callout {{ border-left:4px solid {BRASS}; background:{PAPER2}; border-radius:0 10px 10px 0;
  padding:0.7rem 1rem; margin:0.9rem 0; font-size:0.94rem; color:#3A414C;
  box-shadow:0 1px 3px rgba(20,30,50,.05); }}
.callout.green {{ border-left-color:{GREEN}; }}
.callout.loss {{ border-left-color:{CRIMSON}; }}

.find {{ font-size:1.0rem; line-height:1.7; color:#3A414C; }}
.find b {{ font-weight:700; color:{GREEN}; }}

/* buttons */
.stButton > button, .stFormSubmitButton > button {{
  border:none; background:{BRASS}; color:#fff; border-radius:10px;
  font-family:'Inter',sans-serif; font-weight:700; font-size:0.9rem; padding:0.6rem 1.5rem;
  box-shadow:0 2px 6px rgba(47,107,224,.30); }}
.stButton > button:hover, .stFormSubmitButton > button:hover {{ background:#2456B8; }}

hr {{ border:none; border-top:1px solid {LINE}; margin:1.6rem 0; }}
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:10px; overflow:hidden; }}
[data-testid="stPlotlyChart"] {{ background:{PAPER2}; border:1px solid {LINE}; border-radius:14px;
  padding:10px 12px; box-shadow:0 1px 3px rgba(20,30,50,.06); }}
[data-testid="stMetricValue"] {{ font-family:'Manrope',sans-serif; }}

textarea, input, .stNumberInput div[data-baseweb="input"],
.stSelectbox div[data-baseweb="select"] > div {{
  border:1px solid {LINE} !important; background:{PAPER2} !important; color:{INK} !important;
  border-radius:8px; }}
.stSlider [data-baseweb="slider"] div[role="slider"] {{ background:{BRASS} !important; }}
::placeholder {{ color:{INK60} !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Reusable blocks
# ─────────────────────────────────────────────────────────────────────────────
def masthead(meta):
    st.markdown(
        f"""<div class="masthead">
        <span class="brand">Customer Lifetime Value</span>
        <span class="meta">BG/NBD + Gamma-Gamma · {meta['n_customers_rfm']:,} customers ·
        £{meta['total_clv_12m']:,.0f} CLV / 12m</span></div>""",
        unsafe_allow_html=True)


def rule(left, right=""):
    st.markdown(
        f"""<div style="display:flex; align-items:center; justify-content:space-between;
        margin:2.2rem 0 1rem 0;">
        <span style="font-family:'Manrope',sans-serif; font-weight:800; font-size:1.4rem;
        letter-spacing:-0.02em; display:flex; align-items:center; gap:.55rem;">
        <span style="width:5px; height:1.1em; background:{BRASS}; border-radius:3px;"></span>{left}</span>
        <span style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
        font-weight:600; color:{INK60};">{right}</span></div>""",
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
