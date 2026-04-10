"""
Customer Lifetime Value Dashboard
Built with Streamlit + lifetimes (BG/NBD + Gamma-Gamma)
"""

import os
import warnings
import joblib
import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from lifetimes import BetaGeoFitter, GammaGammaFitter

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_PATH  = os.path.join(BASE_DIR, "models", "model_meta.pkl")
MODEL_RF   = os.path.join(BASE_DIR, "models", "rf_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "data", "online_retail.xlsx")

DATASET_URL = (
    "https://raw.githubusercontent.com/databricks/Spark-The-Definitive-Guide"
    "/master/data/retail-data/all/online-retail-dataset.csv"
)

SEGMENT_COLORS = {
    "Champions":        "#6C63FF",
    "Loyal Customers":  "#3ECFCF",
    "Recent Customers": "#48BB78",
    "Promising":        "#9AE6B4",
    "Need Attention":   "#F6AD55",
    "At Risk":          "#FC8181",
    "Can't Lose Them":  "#E53E3E",
    "Lost":             "#718096",
}

# ─── Data loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models(meta):
    # Reconstruct lifetimes models from saved params (avoids pickle/lambda issues)
    bgf = BetaGeoFitter()
    bgf.params_ = meta["bgf_params"]
    gg = GammaGammaFitter()
    gg.params_ = meta["gg_params"]
    rf = joblib.load(MODEL_RF)
    return bgf, gg, rf

@st.cache_resource
def load_meta():
    return joblib.load(META_PATH)

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer CLV Dashboard",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 CLV Dashboard")
    st.markdown("*Customer Lifetime Value*")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "🗺️ RFM Segments", "📈 CLV Forecast", "🔮 Simulator"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small>BG/NBD + Gamma-Gamma model<br>"
        "UCI Online Retail dataset<br>"
        "4,300+ customers · 500K+ transactions</small>",
        unsafe_allow_html=True,
    )

# ─── Load ─────────────────────────────────────────────────────────────────────
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

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Overview")
    st.caption("Business summary — UCI Online Retail dataset (UK e-commerce, 2010-2011)")

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{meta['n_customers_raw']:,}")
    c2.metric("Transactions", f"{meta['n_transactions']:,}")
    c3.metric("CLV Forecast (12m)", f"£{meta['total_clv_12m']:,.0f}")
    c4.metric("Median CLV / Customer", f"£{meta['median_clv_12m']:,.2f}")
    c5.metric("Avg P(Alive)", f"{meta['avg_p_alive']:.1%}")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Monthly Revenue")
        fig = px.area(
            monthly_rev,
            x="date", y="revenue",
            labels={"date": "", "revenue": "Revenue (£)"},
            color_discrete_sequence=["#6C63FF"],
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Countries")
        top10 = country_rev.head(10)
        fig2 = px.bar(
            top10, x="Revenue", y="Country",
            orientation="h",
            color="Revenue",
            color_continuous_scale="Purples",
            labels={"Revenue": "Revenue (£)", "Country": ""},
        )
        fig2.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Cohort retention heatmap
    st.subheader("Cohort Retention Heatmap")
    st.caption("% of customers from each acquisition cohort still purchasing N months later")

    cohort = meta["cohort_retention"].iloc[:12, :13]  # last 12 cohorts, 12 periods
    cohort.index = cohort.index.astype(str)
    cohort.columns = [f"M+{c}" for c in cohort.columns]

    fig3 = go.Figure(data=go.Heatmap(
        z=cohort.values * 100,
        x=cohort.columns.tolist(),
        y=cohort.index.tolist(),
        colorscale="Purples",
        text=np.round(cohort.values * 100, 1),
        texttemplate="%{text}%",
        hovertemplate="Cohort %{y}<br>%{x}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Retention %"),
    ))
    fig3.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Months since first purchase",
        yaxis_title="Acquisition cohort",
        height=350,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RFM SEGMENTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ RFM Segments":
    st.title("🗺️ RFM Segmentation")
    st.caption("Customers grouped by Recency · Frequency · Monetary value using the BG/NBD model output")

    # Segment cards
    st.subheader("Segment Summary")
    cols = st.columns(4)
    for i, row in seg_stats.sort_values("total_clv_12m", ascending=False).iterrows():
        col = cols[i % 4]
        color = SEGMENT_COLORS.get(row["segment"], "#718096")
        col.markdown(
            f"""
            <div style="background:{color}22;border-left:4px solid {color};
                        padding:12px 16px;border-radius:6px;margin-bottom:8px">
                <b style="color:{color}">{row['segment']}</b><br>
                <span style="font-size:1.4rem;font-weight:700">{int(row['customers']):,}</span>
                <span style="font-size:0.8rem"> customers</span><br>
                <small>Avg CLV: £{row['avg_clv_12m']:,.0f} &nbsp;|&nbsp; P(alive): {row['avg_p_alive']:.0%}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customers & CLV by Segment")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        seg_sorted = seg_stats.sort_values("customers", ascending=True)
        fig.add_trace(go.Bar(
            y=seg_sorted["segment"], x=seg_sorted["customers"],
            orientation="h", name="Customers",
            marker_color=[SEGMENT_COLORS.get(s, "#718096") for s in seg_sorted["segment"]],
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            y=seg_sorted["segment"], x=seg_sorted["avg_clv_12m"],
            mode="markers+lines", name="Avg CLV 12m (£)",
            marker=dict(size=10, color="white", line=dict(width=2, color="#6C63FF")),
        ), secondary_y=True)
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.15),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            height=380,
        )
        fig.update_yaxes(title_text="Avg CLV (£)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Total Revenue Potential by Segment")
        fig2 = px.treemap(
            seg_stats,
            path=["segment"],
            values="total_clv_12m",
            color="avg_p_alive",
            color_continuous_scale="Purples",
            hover_data={"customers": True, "avg_clv_12m": ":.0f"},
            labels={"total_clv_12m": "Total CLV 12m", "avg_p_alive": "P(Alive)"},
        )
        fig2.update_traces(
            texttemplate="<b>%{label}</b><br>£%{value:,.0f}",
            hovertemplate="<b>%{label}</b><br>Total CLV: £%{value:,.0f}<br>P(Alive): %{color:.1%}<extra></extra>",
        )
        fig2.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # RFM scatter
    st.subheader("RFM Space — Frequency vs Monetary Value")
    st.caption("Bubble size = CLV 12m · Color = Segment · Opacity = P(alive)")
    sample = rfm.sample(min(1500, len(rfm)), random_state=42).reset_index()

    fig3 = px.scatter(
        sample,
        x="frequency", y="monetary_value",
        size=np.clip(sample["clv_12m"], 1, sample["clv_12m"].quantile(0.95)),
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.75,
        hover_data={"p_alive": ":.1%", "clv_12m": ":.2f", "recency": True},
        labels={
            "frequency": "Purchase Frequency",
            "monetary_value": "Avg Order Value (£)",
            "clv_12m": "CLV 12m (£)",
        },
    )
    fig3.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — CLV FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 CLV Forecast":
    st.title("📈 CLV Forecast")
    st.caption("Projected Customer Lifetime Value using BG/NBD + Gamma-Gamma probabilistic models")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total CLV — 6 months",  f"£{meta['total_clv_6m']:,.0f}")
    c2.metric("Total CLV — 12 months", f"£{meta['total_clv_12m']:,.0f}")
    c3.metric("Growth (6m → 12m)",
              f"+£{(meta['total_clv_12m'] - meta['total_clv_6m']):,.0f}",
              f"+{(meta['total_clv_12m']/meta['total_clv_6m']-1)*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("CLV Distribution (12 months)")
        clv_vals = rfm["clv_12m"].clip(upper=rfm["clv_12m"].quantile(0.97))
        fig = px.histogram(
            clv_vals, nbins=60,
            color_discrete_sequence=["#6C63FF"],
            labels={"value": "CLV 12m (£)", "count": "Customers"},
        )
        fig.add_vline(x=rfm["clv_12m"].median(), line_dash="dash", line_color="#FC8181",
                      annotation_text=f"Median £{rfm['clv_12m'].median():.0f}", annotation_position="top right")
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("6m vs 12m CLV by Segment")
        seg_melt = seg_stats.melt(
            id_vars="segment",
            value_vars=["avg_clv_12m"],
            var_name="horizon", value_name="avg_clv",
        )
        # Manually add 6m from rfm
        seg_6m = rfm.groupby("segment")["clv_6m"].mean().reset_index()
        seg_6m.columns = ["segment", "clv_6m"]
        seg_merged = seg_stats[["segment", "avg_clv_12m"]].merge(seg_6m, on="segment")
        seg_merged = seg_merged.sort_values("avg_clv_12m", ascending=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=seg_merged["segment"], x=seg_merged["clv_6m"],
            name="6 months", orientation="h", marker_color="#3ECFCF",
        ))
        fig2.add_trace(go.Bar(
            y=seg_merged["segment"], x=seg_merged["avg_clv_12m"],
            name="12 months", orientation="h", marker_color="#6C63FF",
        ))
        fig2.update_layout(
            barmode="group",
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)", title="Avg CLV (£)"),
            legend=dict(orientation="h", y=-0.2),
            height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # P(alive) vs CLV scatter
    st.subheader("Probability Alive vs CLV 12m")
    st.caption("Top-right quadrant: high-value customers still active — priority retention targets")
    sample = rfm.sample(min(2000, len(rfm)), random_state=1).reset_index()

    fig3 = px.scatter(
        sample, x="p_alive", y="clv_12m",
        color="segment",
        color_discrete_map=SEGMENT_COLORS,
        opacity=0.65,
        labels={"p_alive": "P(Alive)", "clv_12m": "CLV 12m (£)"},
        hover_data={"frequency": True, "monetary_value": ":.2f"},
    )
    fig3.add_vline(x=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig3.add_hline(y=rfm["clv_12m"].median(), line_dash="dot", line_color="rgba(255,255,255,0.3)")
    fig3.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
        height=400,
        legend=dict(orientation="h", y=-0.2),
    )
    # Annotate quadrants
    fig3.add_annotation(x=0.9, y=rfm["clv_12m"].quantile(0.9),
        text="Protect", showarrow=False,
        font=dict(color="#6C63FF", size=11), opacity=0.7)
    fig3.add_annotation(x=0.1, y=rfm["clv_12m"].quantile(0.9),
        text="Reactivate", showarrow=False,
        font=dict(color="#FC8181", size=11), opacity=0.7)
    st.plotly_chart(fig3, use_container_width=True)

    # RF baseline note
    with st.expander("ML Baseline (Random Forest Regressor)"):
        st.markdown(
            f"""
            As a benchmark, a **Random Forest** was trained to predict CLV 12m from RFM features.

            | Metric | Value |
            |--------|-------|
            | MAE | £{meta['rf_metrics']['mae']:,.2f} |
            | R² | {meta['rf_metrics']['r2']:.4f} |

            The BG/NBD + Gamma-Gamma approach is preferred because it is **interpretable**,
            grounded in purchase behavior theory, and produces **probability distributions**
            rather than point estimates — enabling richer segmentation and "what-if" analysis.
            """
        )

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Simulator":
    st.title("🔮 Customer CLV Simulator")
    st.caption("Predict CLV and segment for any customer given their purchase history")

    col_form, col_out = st.columns([1, 1])

    with col_form:
        st.subheader("Customer Profile")
        with st.form("simulator_form"):
            frequency = st.slider(
                "Repeat purchases (frequency)",
                min_value=1, max_value=100, value=10,
                help="Number of repeat purchases (transactions after the first)",
            )
            recency = st.slider(
                "Recency (days since last purchase)",
                min_value=1, max_value=730, value=60,
                help="Days between first and last purchase",
            )
            T = st.slider(
                "Customer age T (days observed)",
                min_value=int(recency) + 1, max_value=730, value=max(int(recency) + 10, 180),
                help="Total days the customer has been observed",
            )
            monetary = st.number_input(
                "Avg order value (£)",
                min_value=1.0, max_value=5000.0, value=75.0, step=5.0,
            )
            horizon = st.selectbox("CLV forecast horizon", [3, 6, 12, 24], index=1)
            submitted = st.form_submit_button("Predict CLV", use_container_width=True)

    with col_out:
        if submitted:
            # BG/NBD predictions
            p_alive = float(bgf.conditional_probability_alive(frequency, recency, T))
            exp_purch = float(bgf.conditional_expected_number_of_purchases_up_to_time(
                horizon * 30, frequency, recency, T
            ))

            # Gamma-Gamma CLV
            clv = float(gg.customer_lifetime_value(
                bgf,
                pd.Series([frequency]),
                pd.Series([recency]),
                pd.Series([T]),
                pd.Series([monetary]),
                time=horizon,
                freq="D",
                discount_rate=0.01,
            ).iloc[0])

            # Segment assignment
            r_score = int(pd.cut([recency], bins=5, labels=[5, 4, 3, 2, 1])[0])
            f_score = min(5, max(1, int(np.digitize(frequency, [1, 3, 7, 15, 30]))))
            m_score = min(5, max(1, int(np.digitize(monetary, [10, 30, 75, 150, 300]))))

                def _seg(row):
                r, f, m = row["r_score"], row["f_score"], row["m_score"]
                if r >= 4 and f >= 4:
                    return "Champions"
                elif r >= 3 and f >= 3:
                    return "Loyal Customers"
                elif r >= 4 and f <= 2:
                    return "Recent Customers"
                elif r >= 3 and f <= 2:
                    return "Promising"
                elif r <= 2 and f >= 3:
                    return "At Risk"
                elif r <= 2 and f >= 4:
                    return "Can't Lose Them"
                elif r <= 2 and f <= 2 and m <= 2:
                    return "Lost"
                else:
                    return "Need Attention"
            seg = _seg({"r_score": r_score, "f_score": f_score, "m_score": m_score})
            seg_color = SEGMENT_COLORS.get(seg, "#718096")

            st.subheader("Prediction Results")

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_alive * 100,
                title={"text": "P(Alive) %", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": seg_color},
                    "steps": [
                        {"range": [0, 30], "color": "#FC818133"},
                        {"range": [30, 70], "color": "#F6AD5533"},
                        {"range": [70, 100], "color": "#48BB7833"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": 50},
                },
                number={"suffix": "%", "font": {"size": 28}},
            ))
            fig_gauge.update_layout(
                height=220,
                margin=dict(l=20, r=20, t=30, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.metric(f"CLV ({horizon}m)", f"£{clv:,.2f}")
            c2.metric("Expected purchases", f"{exp_purch:.1f}")

            st.markdown(
                f"""
                <div style="background:{seg_color}22;border-left:4px solid {seg_color};
                            padding:12px 16px;border-radius:6px;margin-top:8px">
                    <b style="color:{seg_color};font-size:1.1rem">Segment: {seg}</b><br>
                    <small>RFM scores — R: {r_score} · F: {f_score} · M: {m_score}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interpretation
            st.markdown("---")
            st.markdown("**Recommended action:**")
            actions = {
                "Champions":        "Reward them. They are your best customers — loyalty programs, early access.",
                "Loyal Customers":  "Upsell higher-value products. They respond to recommendations.",
                "Recent Customers": "Nurture the relationship — onboarding sequences, engagement offers.",
                "Promising":        "Build frequency — incentivize the second and third purchase.",
                "Need Attention":   "Reconnect before they slip away — targeted re-engagement offer.",
                "At Risk":          "Send a win-back campaign now. Discount + personalized message.",
                "Can't Lose Them":  "High-value but disengaging — escalate to account manager level.",
                "Lost":             "Low ROI on retention spend. Focus budget on other segments.",
            }
            st.info(actions.get(seg, "Evaluate based on specific customer context."))

        else:
            st.markdown(
                """
                <div style="display:flex;align-items:center;justify-content:center;
                            height:300px;opacity:0.4;flex-direction:column">
                    <span style="font-size:3rem">🔮</span>
                    <p>Fill in the form and click <b>Predict CLV</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>Built by <a href='https://santimuru.github.io' target='_blank'>Santiago Martínez</a> "
    "· BG/NBD + Gamma-Gamma probabilistic CLV model · UCI Online Retail dataset</small>",
    unsafe_allow_html=True,
)
