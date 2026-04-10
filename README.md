# Customer Lifetime Value Prediction

> Predicting how much each customer is worth — and what to do about it.
> **Live demo:** [customer-clv-prediction-santiagomuru.streamlit.app](https://customer-clv-prediction-santiagomuru.streamlit.app/)

---

## Overview

Most churn models tell you *who* is leaving. This project goes further: it quantifies **how much each customer is worth over their lifetime**, segments them by behavior, and recommends retention actions accordingly.

Built on the **BG/NBD + Gamma-Gamma** probabilistic framework — the same models used by Amazon, LVMH, and subscription businesses worldwide to drive CRM strategy.

Applied a similar approach in a production CRM environment at a cable operator, integrating CLV scores into targeted retention and upsell campaigns.

---

## Live Dashboard

| Section | What you'll find |
|---|---|
| 📊 Overview | Revenue trends, cohort retention heatmap, top markets |
| 🗺️ RFM Segments | 8-tier customer segmentation, treemap of CLV potential |
| 📈 CLV Forecast | 6m/12m projections, P(alive) analysis, ML baseline comparison |
| 🔮 Simulator | Enter any customer's history → CLV + segment + recommended action |

---

## Why BG/NBD + Gamma-Gamma?

Most portfolios use scikit-learn for everything. This project uses **statistical models grounded in purchase behavior theory**:

| Model | What it predicts |
|---|---|
| **BG/NBD** (Beta-Geometric / Negative Binomial Distribution) | How many times a customer will buy, and whether they're still "alive" |
| **Gamma-Gamma** | How much they'll spend per transaction |
| **CLV** = BG/NBD × Gamma-Gamma × time horizon | Total expected revenue from a customer |

A Random Forest regressor is included as a benchmark, but the probabilistic approach wins on interpretability and richer output (probability distributions, not just point estimates).

---

## Tech Stack

- **Python 3.11**
- **lifetimes** — BG/NBD and Gamma-Gamma implementation
- **Scikit-learn** — RF baseline + preprocessing
- **Streamlit** — interactive dashboard
- **Plotly** — visualizations (treemap, cohort heatmap, scatter, gauge)
- **Pandas / NumPy** — data processing

---

## Dataset

UCI Online Retail — real UK e-commerce transactions, 2010–2011.

- ~500,000 transactions · ~4,300 repeat customers
- Features: `InvoiceDate`, `CustomerID`, `Quantity`, `UnitPrice`, `Country`
- Auto-downloaded from the UCI ML Repository on first run

---

## Key Results

| Metric | Value |
|---|---|
| Customers analyzed | ~4,300 (repeat purchasers) |
| Projected CLV 12m | ~£9.5M |
| Avg P(alive) | ~62% |
| RF Baseline R² | ~0.87 |

**Top insight:** The top 20% of customers (Champions + Loyal) represent over 65% of total projected CLV — classic Pareto, but the BG/NBD model lets you act on it *before* they churn.

---

## Project Structure

```
customer-clv-prediction/
├── app/
│   └── app.py              # Streamlit dashboard
├── src/
│   └── train.py            # BG/NBD + GG model training
├── data/
│   └── online_retail.xlsx  # Downloaded on first run
├── models/
│   ├── bgn_model.pkl       # Fitted BG/NBD model
│   ├── gg_model.pkl        # Fitted Gamma-Gamma model
│   ├── rf_model.pkl        # RF baseline
│   └── model_meta.pkl      # RFM table, metrics, segment stats, cohort data
├── requirements.txt
├── .python-version         # 3.11
└── README.md
```

---

## Setup & Run

```bash
git clone https://github.com/santimuru/customer-clv-prediction.git
cd customer-clv-prediction
pip install -r requirements.txt
python src/train.py          # Downloads data, trains models (~3 min)
streamlit run app/app.py
```

---

## About

Built by [Santiago Martínez](https://santimuru.github.io) — Data Analyst with 6+ years in telecom, e-commerce, and consulting.

Previously deployed CLV-adjacent retention models in a production CRM environment at a cable operator, integrating model outputs into segmented campaign workflows.

---

## License

MIT
