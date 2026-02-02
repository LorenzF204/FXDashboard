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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.data_loader import get_fred_data, load_policy_rates, load_cpi, load_fx_data

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
