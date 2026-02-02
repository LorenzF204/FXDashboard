import pandas as pd
import numpy as np

# --------------------------------------------------
# Helper
# --------------------------------------------------

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Monatliche FX Returns
    """
    if prices.index.freq is None or str(prices.index.freq) not in ["M", "ME"]:
        prices = prices.resample("ME").last()

    return prices.pct_change()


# --------------------------------------------------
# Position Construction
# --------------------------------------------------

def build_positions(
    pair_scores: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Long Top-N / Short Bottom-N FX Pairs (Equal Weight)
    """

    positions = pd.DataFrame(0.0, index=pair_scores.index, columns=pair_scores.columns)

    for t in pair_scores.index:
        scores = pair_scores.loc[t].dropna()
        if len(scores) < top_n:
            continue

        ranked = scores.sort_values(ascending=False)

        longs = ranked.head(top_n).index
        shorts = ranked.tail(top_n).index

        positions.loc[t, longs] = 1.0 / top_n
        positions.loc[t, shorts] = -1.0 / top_n

    return positions


# --------------------------------------------------
# Pair Returns
# --------------------------------------------------

def compute_pair_returns(
    fx_prices: pd.DataFrame,
    pairs: list,
) -> pd.DataFrame:
    """
    Berechnet FX Pair Returns aus Spotpreisen
    """

    fx_returns = compute_returns(fx_prices)
    pair_returns = {}

    for pair in pairs:
        base, quote = pair.split("/")
        # Sicherstellen, dass die Ticker in fx_returns vorhanden sind
        if base in fx_returns.columns and quote in fx_returns.columns:
            pair_returns[pair] = fx_returns[base] - fx_returns[quote]
        else:
            # Falls Ticker fehlen, setzen wir die Returns auf NaN
            pair_returns[pair] = pd.Series(np.nan, index=fx_returns.index)

    return pd.DataFrame(pair_returns)


# --------------------------------------------------
# Backtest Engine
# --------------------------------------------------

def run_backtest(
    fx_prices: pd.DataFrame,
    pair_scores: pd.DataFrame,
    top_n: int = 5,
    vol_target: float = 0.10,
    vol_window: int = 12,
) -> pd.DataFrame:
    """
    Monats-Backtest mit Vol-Targeting
    """

    # -----------------------------
    # Pair Returns
    # -----------------------------
    pairs = pair_scores.columns.tolist()
    pair_returns = compute_pair_returns(fx_prices, pairs)

    # -----------------------------
    # Positions
    # -----------------------------
    positions = build_positions(pair_scores, top_n)

    # -----------------------------
    # Strategy Returns
    # -----------------------------
    strat_ret = (positions.shift(1) * pair_returns).sum(axis=1)

    # -----------------------------
    # Vol Targeting
    # -----------------------------
    rolling_vol = strat_ret.rolling(vol_window).std() * np.sqrt(12)
    leverage = vol_target / rolling_vol
    leverage = leverage.clip(0, 3)

    strat_ret = strat_ret * leverage.shift(1)

    # -----------------------------
    # Equity Curve
    # -----------------------------
    equity = (1 + strat_ret.fillna(0)).cumprod()

    return pd.DataFrame({
        "returns": strat_ret,
        "equity": equity,
        "leverage": leverage,
    })