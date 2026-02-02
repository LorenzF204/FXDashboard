import pandas as pd
import numpy as np
from itertools import permutations

# --------------------------------------------------
# FX Pair Construction
# --------------------------------------------------

def build_fx_pairs(currencies: list) -> list:
    """
    Erzeugt alle gerichteten FX-Paare (Base / Quote)
    """
    return list(permutations(currencies, 2))


# --------------------------------------------------
# Pair Score Calculation
# --------------------------------------------------

def calculate_pair_scores(
    macro_scores: pd.DataFrame,
    trend_signal: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Berechnet Pair-Scores als Differenz der Makro-Scores
    Optional: Trend-Filter als Veto
    """

    pairs = build_fx_pairs(macro_scores.columns)
    pair_scores = {}

    for base, quote in pairs:
        score = macro_scores[base] - macro_scores[quote]

        # Optionaler Trend-Filter
        if trend_signal is not None:
            # Sicherstellen, dass die Ticker in trend_signal vorhanden sind
            if base in trend_signal.columns and quote in trend_signal.columns:
                trend_ok = (
                    trend_signal[base] == 1
                ) & (
                    trend_signal[quote] == -1
                )
                score = score.where(trend_ok)
            else:
                # Falls Ticker fehlen, setzen wir den Score auf NaN (Veto greift)
                score = score * np.nan

        pair_name = f"{base}/{quote}"
        pair_scores[pair_name] = score

    return pd.DataFrame(pair_scores)


# --------------------------------------------------
# Pair Ranking
# --------------------------------------------------

def rank_pairs(
    pair_scores: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Ranked FX-Pair Matrix (Top / Bottom)
    """

    latest = pair_scores.iloc[-1].dropna()
    ranked = latest.sort_values(ascending=False)

    return pd.DataFrame({
        "score": ranked,
        "rank": range(1, len(ranked) + 1),
    }).head(top_n)