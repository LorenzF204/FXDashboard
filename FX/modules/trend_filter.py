import pandas as pd
import numpy as np

# --------------------------------------------------
# FX Trend / Momentum Filter
# --------------------------------------------------

def calculate_fx_momentum(
    fx_prices: pd.DataFrame,
    lookback_months: int = 12,
    zscore_window: int = 36,
) -> dict:
    """
    Berechnet FX-Preis-Momentum als Trend-Filter

    Inputs:
    - fx_prices: FX Spot Prices (DataFrame, monatlich oder täglich)
    - lookback_months: Momentum-Horizont (Default: 12M)
    - zscore_window: Fenster für Z-Score-Normalisierung

    Output:
    Dictionary mit:
    - momentum: 12M Returns
    - momentum_z: normalisierte Momentum-Scores
    - trend_signal: +1 / -1 / 0 Trend-Indikator
    """

    # -----------------------------
    # Monatsultimo sicherstellen
    # -----------------------------
    # Check if frequency is monthly (M or ME)
    is_monthly = False
    if hasattr(fx_prices.index, 'freq') and fx_prices.index.freq is not None:
        freq_str = str(fx_prices.index.freq)
        if any(m in freq_str for m in ["M", "ME"]):
            is_monthly = True
            
    if not is_monthly:
        fx_prices = fx_prices.resample("ME").last()

    # -----------------------------
    # 12M Momentum (Total Return)
    # -----------------------------
    momentum = fx_prices.pct_change(lookback_months)

    # -----------------------------
    # Z-Score Normalisierung
    # -----------------------------
    mean = momentum.rolling(zscore_window).mean()
    std = momentum.rolling(zscore_window).std()
    momentum_z = (momentum - mean) / std

    # -----------------------------
    # Trend Signal (hart)
    # -----------------------------
    trend_signal = pd.DataFrame(
        np.where(momentum_z > 0, 1, np.where(momentum_z < 0, -1, 0)),
        index=momentum_z.index,
        columns=momentum_z.columns,
    )

    return {
        "momentum": momentum,
        "momentum_z": momentum_z,
        "trend_signal": trend_signal,
    }