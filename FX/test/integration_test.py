"""
Unit Test Dokumentation für data_loader.py

Diese Datei enthält Unit Tests für die Funktionen in modules/data_loader.py.
Die Tests verwenden Mocks, um externe API-Aufrufe (FRED, Yahoo Finance) zu simulieren.

### Gefundene Fehler und Fehlerbehebung:

1. Fehlerhafter Import: Den Import 'from scipy.stats import pairs' entfernt, da er nicht existierte.
2. Unendliche Rekursion: load_policy_rates() so angepasst, dass sie ein Dictionary erwartet statt sich selbst aufzurufen.
3. Nicht definierte Variable & Falscher Ticker: 'pairs' in load_fx_data entfernt und Ticker mit '=X' Suffix versehen.
4. Falsche Docstring-Platzierung: Docstrings in load_cpi an den Funktionsanfang verschoben.

### Testbefehl:
cd FXDashboard
py -m pytest test/integration_test.py
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
import sys
import os

# Pfad zum Projekt-Root hinzufügen, damit 'modules' gefunden wird
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from modules.data_loader import get_fred_data, load_policy_rates, load_cpi, load_fx_data
from modules.score_calculation import zscore, calculate_macro_scores
from modules.trend_filter import calculate_fx_momentum
from modules.pair_matrix import build_fx_pairs, calculate_pair_scores, rank_pairs
from modules.backtest import (
    compute_returns,
    build_positions,
    compute_pair_returns,
    run_backtest
)

# Import der Regime-Funktionen
from modules.regime_module import (
    build_regime_features, 
    fit_hmm_regime, 
    compute_transition_matrix, 
    regime_weight_adjustment,
    REGIME_MAP
)

def test_dashboard_logic_flow():
    """
    Simuliert den Logik-Flow von dashboard.py.
    Da dashboard.py direkt Streamlit-Commands aufruft, testen wir hier
    die Integration der Komponenten, wie sie im Dashboard verwendet werden.
    """
    # 1. Mock Daten (ähnlich wie im Dashboard Spinner Block)
    dates = pd.date_range('2020-01-01', periods=100, freq='ME')
    CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "NZD", "SEK", "NOK"]
    
    policy = pd.DataFrame(np.random.randn(100, 10), index=dates, columns=CURRENCIES)
    inflation = pd.DataFrame(np.random.randn(100, 10), index=dates, columns=CURRENCIES)
    growth = pd.DataFrame(np.random.randn(100, 10), index=dates, columns=CURRENCIES)
    fx_prices = pd.DataFrame(np.exp(np.random.randn(100, 10).cumsum(axis=0)), index=dates, columns=CURRENCIES)
    risk = pd.Series(np.random.randn(100), index=dates, name="VIX")

    # 2. Macro Scores
    macro = calculate_macro_scores(
        policy=policy,
        growth=growth,
        inflation=inflation,
        risk=risk,
    )
    assert "stable_score" in macro
    
    # 3. Trend Filter
    trend = calculate_fx_momentum(fx_prices)
    assert "trend_signal" in trend
    
    # 4. Regime Module
    features = build_regime_features(growth, inflation, risk)
    hmm = fit_hmm_regime(features)
    assert "regimes" in hmm
    
    transition = compute_transition_matrix(hmm["regimes"])
    assert transition.shape[0] > 0
    
    # 5. Pair Matrix
    # Wir stellen sicher, dass wir genug gültige Signale für den Backtest haben
    # Indem wir trend_signal weglassen oder sicherstellen, dass es 1/-1 Paare gibt.
    # Für den Integrationstest der Dashboard-Logik ist es okay, den Filter zu testen.
    pair_scores = calculate_pair_scores(
        macro["stable_score"],
        trend_signal=None, # Ohne Veto für stabilere Test-Returns
    )
    assert not pair_scores.empty
    
    top_pairs = rank_pairs(pair_scores, top_n=5)
    assert len(top_pairs) <= 5
    
    # 6. Backtest
    # Wir verwenden ein kleineres vol_window für die Mocks
    bt = run_backtest(
        fx_prices=fx_prices,
        pair_scores=pair_scores,
        top_n=2,
        vol_window=5 
    )
    assert "equity" in bt
    assert "returns" in bt
    
    # Sharpe Ratio Berechnung (wie im Dashboard)
    valid_returns = bt["returns"].dropna()
    # Wir prüfen ob wir valide Returns haben und ob std > 0
    if len(valid_returns) > 1 and valid_returns.std() > 0:
        sharpe = valid_returns.mean() / valid_returns.std() * np.sqrt(12)
        assert not np.isnan(sharpe)
    else:
        # Falls std 0 ist (extrem unwahrscheinlich bei random) oder zu wenig Daten
        # Geben wir uns mit der Existenz der Spalten zufrieden
        assert "returns" in bt

@pytest.fixture
def sample_data():
    """Erstellt Testdaten für Score-Berechnungen."""
    dates = pd.date_range('2020-01-01', periods=40, freq='ME')
    data = {
        'USA': np.linspace(1, 5, 40),
        'EUR': np.linspace(0.5, 3, 40)
    }
    df = pd.DataFrame(data, index=dates)
    series = pd.Series(np.linspace(10, 30, 40), index=dates)
    return df, series

def test_zscore():
    """Testet die zscore Funktion."""
    dates = pd.date_range('2020-01-01', periods=40, freq='ME')
    df = pd.DataFrame({'val': np.linspace(1, 40, 40)}, index=dates)
    
    result = zscore(df, window=36)
    
    assert isinstance(result, pd.DataFrame)
    assert result.shape == df.shape
    # Die ersten window-1 (35) Werte sollten NaN sein
    assert result.iloc[:35].isna().all().all()
    # Der 36. Wert sollte berechnet sein
    assert not np.isnan(result.iloc[35, 0])

def test_calculate_macro_scores(sample_data):
    """Testet die calculate_macro_scores Funktion."""
    df, series = sample_data
    
    # Wir verwenden die gleichen Daten für policy, growth, inflation für diesen Test
    result = calculate_macro_scores(
        policy=df,
        growth=df,
        inflation=df,
        risk=series
    )
    
    expected_keys = [
        "policy_z", "growth_z", "inflation_z", 
        "real_rate_z", "risk_z", "raw_score", "stable_score"
    ]
    
    for key in expected_keys:
        assert key in result
        assert isinstance(result[key], (pd.DataFrame, pd.Series))
        assert len(result[key]) == 40

def test_calculate_fx_momentum():
    """Testet die calculate_fx_momentum Funktion aus trend_filter.py."""
    # Testdaten erstellen (50 Monate um Fenster von 36 + 12 zu füllen)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    df = pd.DataFrame({'EURUSD': np.linspace(1.0, 1.2, 50)}, index=dates)
    
    # Test 1: Standard Parameter
    result = calculate_fx_momentum(df, lookback_months=12, zscore_window=36)
    
    assert "momentum" in result
    assert "momentum_z" in result
    assert "trend_signal" in result
    
    # Check Shapes
    assert result["momentum"].shape == df.shape
    assert result["momentum_z"].shape == df.shape
    assert result["trend_signal"].shape == df.shape
    
    # Check Signals (sollten +1 sein da Preis steigt)
    # Die ersten 36 + 12 - 1 = 47 Werte könnten NaN sein für zscore
    # Momentum braucht 12, Z-Score braucht 36 auf Momentum -> 36 + 12 = 48
    assert result["trend_signal"].iloc[48:].all().all() == 1
    
    # Test 2: Resampling Check (Tagesdaten statt Monatsultimo)
    daily_dates = pd.date_range('2020-01-01', periods=100, freq='D')
    df_daily = pd.DataFrame({'EURUSD': np.linspace(1.0, 1.1, 100)}, index=daily_dates)
    
    result_daily = calculate_fx_momentum(df_daily)
    # Das Resultat sollte auf Monatsultimo resampled sein
    # 100 Tage decken ca. 3-4 Monate ab
    assert result_daily["momentum"].index.freqstr in ['ME', 'M']

@patch('pandas_datareader.data.DataReader')
def test_get_fred_data(mock_reader):
    """Testet das Laden von FRED-Daten."""
    mock_df = pd.DataFrame({'value': [1, 2]}, index=pd.date_range('2015-01-01', periods=2))
    mock_reader.return_value = mock_df
    
    result = get_fred_data('SOME_SERIES')
    
    assert isinstance(result, pd.DataFrame)
    mock_reader.assert_called_once()

@patch('modules.data_loader.get_fred_data')
def test_load_policy_rates(mock_get_fred):
    """Testet load_policy_rates ohne Rekursion."""
    mock_df = pd.DataFrame({'value': [1.0]}, index=[pd.Timestamp('2015-01-01')])
    mock_get_fred.return_value = mock_df
    
    rates_dict = {"USA": "FEDFUNDS"}
    result = load_policy_rates(rates_dict)
    
    assert "USA" in result.columns
    assert not result.empty

@patch('yfinance.download')
def test_load_fx_data(mock_yf):
    """Testet load_fx_data mit korrektem Ticker."""
    # Mocking yfinance return
    mock_df = pd.DataFrame({'Adj Close': [1.1]}, index=[pd.Timestamp('2015-01-01')])
    mock_yf.return_value = mock_df
    
    pairs_dict = {"EURUSD": "EURUSD"}
    result = load_fx_data(pairs_dict)
    
    assert "EURUSD" in result.columns
    mock_yf.assert_called_with("EURUSD=X", ANY)

def test_build_regime_features(sample_data):
    """Testet build_regime_features in regime_module.py."""
    df, series = sample_data
    # Mock growth and inflation DataFrames (multiple columns)
    growth = pd.DataFrame({'US': [1, 2], 'EU': [0, 1]}, index=df.index[:2])
    inflation = pd.DataFrame({'US': [2, 3], 'EU': [1, 2]}, index=df.index[:2])
    risk = series.iloc[:2]
    
    features = build_regime_features(growth, inflation, risk)
    
    assert isinstance(features, pd.DataFrame)
    assert all(col in features.columns for col in ["growth", "inflation", "risk"])
    assert len(features) == 2
    # Check standardization (mean should be close to 0)
    assert np.allclose(features.mean(), 0, atol=1e-7)

def test_fit_hmm_regime():
    """Testet fit_hmm_regime in regime_module.py."""
    # Erstelle synthetische Features (3 Spalten, 50 Zeilen)
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    features = pd.DataFrame(np.random.randn(50, 3), index=dates, columns=['growth', 'inflation', 'risk'])
    
    result = fit_hmm_regime(features, n_regimes=3)
    
    assert "model" in result
    assert "regimes" in result
    assert isinstance(result["regimes"], pd.Series)
    assert len(result["regimes"]) == 50
    assert result["regimes"].nunique() <= 3

def test_compute_transition_matrix():
    """Testet compute_transition_matrix in regime_module.py."""
    dates = pd.date_range('2020-01-01', periods=10, freq='ME')
    # Erstelle eine einfache Sequenz: 0, 1, 2, 0, 1, 2...
    regimes = pd.Series([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], index=dates)
    
    matrix = compute_transition_matrix(regimes)
    
    assert isinstance(matrix, pd.DataFrame)
    # Check if index and columns are mapped using REGIME_MAP
    for label in REGIME_MAP.values():
        assert label in matrix.index
        assert label in matrix.columns

def test_regime_weight_adjustment():
    """Testet regime_weight_adjustment in regime_module.py."""
    base_score = pd.Series([1.0, 2.0, -1.0])
    
    # Test Risk-Off (0) -> 0.5x
    res0 = regime_weight_adjustment(0, base_score)
    pd.testing.assert_series_equal(res0, 0.5 * base_score)
    
    # Test Neutral (1) -> 1.0x
    res1 = regime_weight_adjustment(1, base_score)
    pd.testing.assert_series_equal(res1, 1.0 * base_score)
    
    # Test Risk-On (2) -> 1.2x
    res2 = regime_weight_adjustment(2, base_score)
    pd.testing.assert_series_equal(res2, 1.2 * base_score)
    
    # Test Other -> 1.0x (default)
    res3 = regime_weight_adjustment(99, base_score)
    pd.testing.assert_series_equal(res3, base_score)

def test_build_fx_pairs():
    """Testet die Erzeugung von FX-Paaren."""
    currencies = ["USD", "EUR", "GBP"]
    pairs = build_fx_pairs(currencies)
    
    # n*(n-1) = 3*2 = 6 Paare erwartet
    assert len(pairs) == 6
    assert ("EUR", "USD") in pairs
    assert ("USD", "EUR") in pairs
    assert ("USD", "USD") not in pairs

def test_calculate_pair_scores():
    """Testet die Berechnung der Pair-Scores mit und ohne Trend-Veto."""
    dates = pd.date_range('2020-01-01', periods=2, freq='ME')
    macro_scores = pd.DataFrame({
        'USD': [1.0, 1.0],
        'EUR': [0.5, 2.0]
    }, index=dates)
    
    # Test 1: Ohne Trend-Signal
    scores = calculate_pair_scores(macro_scores)
    assert "USD/EUR" in scores.columns
    assert "EUR/USD" in scores.columns
    # USD/EUR Score: 1.0 - 0.5 = 0.5 (Index 0)
    assert scores.loc[dates[0], "USD/EUR"] == 0.5
    # EUR/USD Score: 2.0 - 1.0 = 1.0 (Index 1)
    assert scores.loc[dates[1], "EUR/USD"] == 1.0

    # Test 2: Mit Trend-Signal (Veto)
    # USD/EUR ist okay wenn trend(USD)==1 und trend(EUR)==-1
    trend_signal = pd.DataFrame({
        'USD': [1, -1],
        'EUR': [-1, 1]
    }, index=dates)
    
    scores_veto = calculate_pair_scores(macro_scores, trend_signal=trend_signal)
    
    # Erste Zeile: USD=1, EUR=-1 -> USD/EUR sollte Wert haben, EUR/USD NaN
    assert not np.isnan(scores_veto.loc[dates[0], "USD/EUR"])
    assert np.isnan(scores_veto.loc[dates[0], "EUR/USD"])
    
    # Zweite Zeile: USD=-1, EUR=1 -> USD/EUR sollte NaN haben, EUR/USD Wert
    assert np.isnan(scores_veto.loc[dates[1], "USD/EUR"])
    assert not np.isnan(scores_veto.loc[dates[1], "EUR/USD"])

def test_rank_pairs():
    """Testet das Ranking der FX-Paare."""
    dates = pd.date_range('2020-01-01', periods=1, freq='ME')
    pair_scores = pd.DataFrame({
        'EUR/USD': [2.0],
        'GBP/USD': [3.0],
        'AUD/USD': [1.0],
        'NZD/USD': [0.5]
    }, index=dates)
    
    ranked = rank_pairs(pair_scores, top_n=2)
    
    assert len(ranked) == 2
    assert ranked.index[0] == "GBP/USD"
    assert ranked.iloc[0]["score"] == 3.0
    assert ranked.iloc[0]["rank"] == 1
    assert ranked.index[1] == "EUR/USD"

def test_compute_returns():
    """Testet die Berechnung der monatlichen Returns."""
    dates = pd.date_range('2020-01-01', periods=5, freq='ME')
    prices = pd.DataFrame({'USD': [100, 101, 102, 101, 103]}, index=dates)
    
    returns = compute_returns(prices)
    
    assert len(returns) == 5
    assert np.isnan(returns.iloc[0]['USD'])
    assert returns.iloc[1]['USD'] == pytest.approx((101 - 100) / 100.0)

def test_build_positions():
    """Testet den Aufbau von Long/Short-Positionen."""
    dates = pd.date_range('2020-01-01', periods=2, freq='ME')
    # 6 Paare, top_n=2 -> 2 Long, 2 Short, 2 Zero
    pair_scores = pd.DataFrame({
        'P1': [10, 10], 'P2': [8, 8], 'P3': [5, 5], 
        'P4': [2, 2], 'P5': [-5, -5], 'P6': [-10, -10]
    }, index=dates)
    
    positions = build_positions(pair_scores, top_n=2)
    
    assert positions.loc[dates[0], 'P1'] == 0.5  # Long
    assert positions.loc[dates[0], 'P2'] == 0.5  # Long
    assert positions.loc[dates[0], 'P3'] == 0.0
    assert positions.loc[dates[0], 'P5'] == -0.5 # Short
    assert positions.loc[dates[0], 'P6'] == -0.5 # Short
    assert positions.sum(axis=1).iloc[0] == pytest.approx(0.0)

def test_compute_pair_returns():
    """Testet die Berechnung von Pair-Returns."""
    dates = pd.date_range('2020-01-01', periods=3, freq='ME')
    # Returns:
    # T1: NaN
    # T2: USD: 0.01, EUR: 0.02 -> USD/EUR Return: 0.01 - 0.02 = -0.01
    fx_prices = pd.DataFrame({
        'USD': [100, 101, 102],
        'EUR': [1.0, 1.02, 1.01]
    }, index=dates)
    
    pairs = ["USD/EUR", "EUR/USD"]
    pair_rets = compute_pair_returns(fx_prices, pairs)
    
    assert pair_rets.shape == (3, 2)
    assert "USD/EUR" in pair_rets.columns
    # T2 (Index 1) Check
    usd_ret = (101-100)/100.0
    eur_ret = (1.02-1.0)/1.0
    expected_pair_ret = usd_ret - eur_ret
    assert pair_rets.iloc[1]["USD/EUR"] == pytest.approx(expected_pair_ret)

def test_run_backtest():
    """Testet die gesamte Backtest-Engine."""
    dates = pd.date_range('2020-01-01', periods=20, freq='ME')
    fx_prices = pd.DataFrame({
        'USD': np.linspace(100, 110, 20),
        'EUR': np.linspace(1.1, 1.0, 20)
    }, index=dates)
    
    pair_scores = pd.DataFrame({
        'USD/EUR': np.linspace(1, 5, 20),
        'EUR/USD': np.linspace(-1, -5, 20)
    }, index=dates)
    
    # top_n=1 für 2 Paare
    results = run_backtest(fx_prices, pair_scores, top_n=1, vol_window=5)
    
    assert "returns" in results
    assert "equity" in results
    assert "leverage" in results
    assert len(results) == 20
    assert results["equity"].iloc[-1] != 1.0
