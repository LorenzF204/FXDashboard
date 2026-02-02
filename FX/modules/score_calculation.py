import pandas as pd
import numpy as np

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def zscore(df: pd.DataFrame, window: int = 36) -> pd.DataFrame:
    """
    Rolling Z-Score Normalisierung (Makro-Standard)
    """
    mean = df.rolling(window).mean()
    std = df.rolling(window).std()
    return (df - mean) / std


# --------------------------------------------------
# Macro Score Calculation
# --------------------------------------------------

def calculate_macro_scores(
    policy: pd.DataFrame,
    growth: pd.DataFrame,
    inflation: pd.DataFrame,
    risk: pd.Series,
) -> dict:
    """
    Berechnet Makro-Scores für FX-Allocation (monatlich)

    Inputs:
    - policy: Policy Rates (DataFrame)
    - growth: Growth Proxy (IP YoY)
    - inflation: CPI YoY
    - risk: Global Risk Proxy (z. B. VIX)

    Output:
    Dictionary mit allen Layern und Scores
    """

    # -----------------------------
    # Z-Scores
    # -----------------------------
    policy_z = zscore(policy)
    growth_z = zscore(growth)
    infl_z = zscore(inflation)

    risk_z = zscore(risk.to_frame()).iloc[:, 0]

    # -----------------------------
    # Real Rate
    # -----------------------------
    real_rate = policy - inflation
    real_rate_z = zscore(real_rate)

    # -----------------------------
    # Raw Macro Score (Baseline)
    # -----------------------------
    raw_score = (
        0.30 * policy_z
        + 0.25 * growth_z
        + 0.15 * real_rate_z
        - 0.20 * infl_z
        - 0.10 * risk_z
    )

    # -----------------------------
    # Score Stabilization
    # -----------------------------
    stable_score = raw_score.rolling(3).mean()

    # -----------------------------
    # Output
    # -----------------------------
    return {
        "policy_z": policy_z,
        "growth_z": growth_z,
        "inflation_z": infl_z,
        "real_rate_z": real_rate_z,
        "risk_z": risk_z,
        "raw_score": raw_score,
        "stable_score": stable_score,
    }