import streamlit as st
import pandas as pd
import numpy as np

from modules.data_loader import (
    load_policy_rates,
    load_cpi,
    load_growth,
    load_fx_prices,
    load_risk,
)

from modules.score_calculation import calculate_macro_scores
from modules.trend_filter import calculate_fx_momentum
from modules.regime_module import (
    build_regime_features,
    fit_hmm_regime,
    compute_transition_matrix,
    REGIME_MAP,
)
from modules.pair_matrix import calculate_pair_scores, rank_pairs
from modules.backtest import run_backtest

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------

st.set_page_config(
    page_title="FX Macro Allocation Dashboard",
    layout="wide",
)

st.title("🌍 FX Macro Allocation – G10")

# --------------------------------------------------
# Universe Definition
# --------------------------------------------------

G10 = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK"]

POLICY_RATES = {
    "USD": "FEDFUNDS",
    "EUR": "ECBDFR",
    "JPY": "IRSTCI01JPM156N",
    "GBP": "BOERUKM",
    "CHF": "SNBFR",
    "AUD": "IR3TIB01AUM156N",
    "CAD": "IRSTCI01CAM156N",
    "NZD": "IRSTCI01NZM156N",
    "SEK": "IRSTCI01SEM156N",
    "NOK": "IRSTCI01NOM156N",
}

CPI = {
    "USD": "CPIAUCSL",
    "EUR": "CP0000EZ19M086NEST",
    "JPY": "JPNCPIALLMINMEI",
    "GBP": "GBRCPIALLMINMEI",
    "CHF": "CHECPIALLMINMEI",
}

IP = {
    "USD": "INDPRO",
    "EUR": "EUROINDPRO",
    "JPY": "JPNPROINDMISMEI",
    "GBP": "GBRPROINDMISMEI",
}

FX_TICKERS = {
    "USD": "DX-Y.NYB",
    "EUR": "EURUSD=X",
    "JPY": "JPYUSD=X",
    "GBP": "GBPUSD=X",
    "CHF": "CHFUSD=X",
    "AUD": "AUDUSD=X",
    "CAD": "CADUSD=X",
    "NZD": "NZDUSD=X",
    "SEK": "SEKUSD=X",
    "NOK": "NOKUSD=X",
}

# --------------------------------------------------
# Data Loading
# --------------------------------------------------

with st.spinner("📥 Lade Makro- und FX-Daten..."):
    policy = load_policy_rates(POLICY_RATES)
    inflation = load_cpi(CPI)
    growth = load_growth(IP)
    fx_prices = load_fx_prices(FX_TICKERS)
    risk = load_risk()

# --------------------------------------------------
# Macro Scores
# --------------------------------------------------

macro = calculate_macro_scores(
    policy=policy,
    growth=growth,
    inflation=inflation,
    risk=risk,
)

st.subheader("📊 Makro-Scores (stabilisiert)")
st.dataframe(macro["stable_score"].dropna().tail(12))

# --------------------------------------------------
# Trend Filter
# --------------------------------------------------

trend = calculate_fx_momentum(fx_prices)

st.subheader("📈 FX Trend-Momentum (Z-Score)")
st.dataframe(trend["momentum_z"].dropna().tail(6))

# --------------------------------------------------
# Regime Module
# --------------------------------------------------

features = build_regime_features(growth, inflation, risk)
hmm = fit_hmm_regime(features)

current_regime = hmm["regimes"].iloc[-1]
current_regime_label = REGIME_MAP[current_regime]

transition = compute_transition_matrix(hmm["regimes"])

st.subheader(f"🧭 Aktuelles Makro-Regime: **{current_regime_label}**")
st.dataframe((transition * 100).round(1))

# --------------------------------------------------
# Pair Matrix
# --------------------------------------------------

pair_scores = calculate_pair_scores(
    macro["stable_score"],
    trend["trend_signal"],
)

top_pairs = rank_pairs(pair_scores, top_n=5)

st.subheader("🔀 Top FX-Paare (Makro + Trend)")
st.dataframe(top_pairs)

# --------------------------------------------------
# Backtest
# --------------------------------------------------

with st.spinner("📈 Berechne Backtest..."):
    bt = run_backtest(
        fx_prices=fx_prices,
        pair_scores=pair_scores,
        top_n=5,
    )

st.subheader("📈 Backtest – Equity Curve")
st.line_chart(bt["equity"])

col1, col2 = st.columns(2)

with col1:
    sharpe = bt["returns"].mean() / bt["returns"].std() * np.sqrt(12)
    st.metric("Sharpe Ratio", round(sharpe, 2))

with col2:
    max_dd = (bt["equity"] / bt["equity"].cummax() - 1).min()
    st.metric("Max Drawdown", f"{round(max_dd * 100, 1)}%")