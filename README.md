# Customer Lifetime Value Prediction

> Predicting how much each customer is worth · and what to do about it.
> **Live demo:** [customer-clv-prediction-santiagomuru.streamlit.app](https://customer-clv-prediction-santiagomuru.streamlit.app/)

---

## Overview

Most churn models tell you _who_ is leaving. This project goes further: it quantifies **how much each customer is worth over their lifetime**, segments them by behavior, and recommends retention actions accordingly.

Built on the **BG/NBD + Gamma-Gamma** probabilistic framework · a principled approach grounded in purchase-behavior theory that produces probability distributions, not just point estimates.

---

## Live Dashboard

| Section   | What you'll find                                                              |
| --------- | ----------------------------------------------------------------------------- |
| Ledger    | The brief, headline KPIs, monthly revenue, top markets, cohort retention      |
| Segments  | 8-tier RFM segmentation, headcount vs value, treemap of CLV potential         |
| Forecast  | 6m/12m projections, CLV distribution, P(alive) vs value, ML baseline          |
| Simulator | Enter any customer's history to get CLV + segment + recommended play          |

---

## Why BG/NBD + Gamma-Gamma?

This project uses **statistical models grounded in purchase behavior theory** rather than a generic regression baseline:

| Model                                                        | What it predicts                                                      |
| ------------------------------------------------------------ | --------------------------------------------------------------------- |
| **BG/NBD** (Beta-Geometric / Negative Binomial Distribution) | How many times a customer will buy, and whether they're still "alive" |
| **Gamma-Gamma**                                              | How much they'll spend per transaction                                |
| **CLV** = BG/NBD × Gamma-Gamma × time horizon                | Total expected revenue from a customer                                |

A Random Forest regressor is included as a benchmark, but the probabilistic approach wins on interpretability and richer output (probability distributions, not just point estimates).

---

## Tech Stack

- **Python 3.11**
- **lifetimes** · BG/NBD and Gamma-Gamma implementation
- **Scikit-learn** · RF baseline + preprocessing
- **Streamlit** · interactive dashboard
- **Plotly** · visualizations (treemap, cohort heatmap, scatter, gauge)
- **Pandas / NumPy** · data processing

---

## Dataset

UCI Online Retail · real UK e-commerce transactions, 2010–2011.

- ~500,000 transactions · ~4,300 repeat customers
- Features: `InvoiceDate`, `CustomerID`, `Quantity`, `UnitPrice`, `Country`
- Auto-downloaded from the UCI ML Repository on first run

---

## Key Results

Computed from the trained artifact (see the Introduction page for live figures):

- ~4,300 repeat-purchasing customers modeled from the UCI Online Retail dataset
- BG/NBD + Gamma-Gamma CLV projections at 6m and 12m horizons
- RF baseline R² is near-tautological (features derive mechanically from the CLV target) · treat as a sanity check
- P(alive) is a leading indicator: customers with high CLV but low P(alive) are the highest-priority retention targets

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
│   ├── rf_model.pkl        # RF baseline (joblib)
│   └── model_meta.pkl      # BG/NBD + GG params, RFM table, metrics, segment stats, cohort data
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

Built by [Santiago Martínez](https://santimuru.github.io) · Data Analyst with 6+ years in telecom, e-commerce, and consulting.

---

## License

MIT
