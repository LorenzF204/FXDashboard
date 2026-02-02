import pandas as pd
import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

    class GaussianHMM:
        """
        Fallback-Klasse falls hmmlearn nicht installiert ist.
        Simuliert ein 3-Regime HMM basierend auf einfachen Schwellenwerten.
        """
        def __init__(self, n_components=3, covariance_type="full", n_iter=500, random_state=42):
            self.n_components = n_components
            self.random_state = random_state

        def fit(self, X):
            # Simulation: Wir speichern nichts Besonderes, da predict auf X basiert
            pass

        def predict(self, X):
            # Einfache Heuristik zur Simulation von Regimen:
            # Wir nehmen die Summe der Features als Proxy für das Regime
            # (In build_regime_features sind diese standardisiert)
            scores = np.sum(X, axis=1)
            regimes = np.zeros(len(scores), dtype=int)
            
            # Schwellenwerte für 3 Regime
            regimes[scores < -0.5] = 0  # Risk-Off
            regimes[(scores >= -0.5) & (scores <= 0.5)] = 1  # Neutral
            regimes[scores > 0.5] = 2  # Risk-On
            
            return regimes

# --------------------------------------------------
# Regime Definition
# --------------------------------------------------

REGIME_MAP = {
    0: "Risk-Off / Recession",
    1: "Neutral / Transition",
    2: "Risk-On / Expansion",
}

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def build_regime_features(
    growth: pd.DataFrame,
    inflation: pd.DataFrame,
    risk: pd.Series,
) -> pd.DataFrame:
    """
    Baut aggregierte Makro-Features für Regime-Erkennung
    """

    features = pd.DataFrame(index=growth.index)

    features["growth"] = growth.mean(axis=1)
    features["inflation"] = inflation.mean(axis=1)
    features["risk"] = risk

    features = features.dropna()

    # Standardisierung (wichtig für HMM-Stabilität)
    features = (features - features.mean()) / features.std()

    return features


# --------------------------------------------------
# HMM Regime Model
# --------------------------------------------------

def fit_hmm_regime(
    features: pd.DataFrame,
    n_regimes: int = 3,
    covariance_type: str = "full",
    n_iter: int = 500,
) -> dict:
    """
    Trainiert ein Gaussian HMM zur Regime-Erkennung
    """

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=42,
    )

    model.fit(features.values)

    regimes = model.predict(features.values)

    regime_series = pd.Series(
        regimes,
        index=features.index,
        name="regime",
    )

    return {
        "model": model,
        "regimes": regime_series,
    }


# --------------------------------------------------
# Regime Transition Matrix
# --------------------------------------------------

def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """
    Berechnet die Regime-Transition-Matrix (monatlich)
    """

    transition = pd.crosstab(
        regimes.shift(1),
        regimes,
        normalize="index",
    )

    transition.index = transition.index.map(REGIME_MAP)
    transition.columns = transition.columns.map(REGIME_MAP)

    return transition


# --------------------------------------------------
# Regime-adjustierte Gewichtung
# --------------------------------------------------

def regime_weight_adjustment(
    regime: int,
    base_score: pd.Series,
) -> pd.Series:
    """
    Passt Makro-Scores abhängig vom Regime an
    """

    if regime == 0:  # Risk-Off
        return 0.5 * base_score

    elif regime == 1:  # Neutral
        return 1.0 * base_score

    elif regime == 2:  # Risk-On
        return 1.2 * base_score

    else:
        return base_score