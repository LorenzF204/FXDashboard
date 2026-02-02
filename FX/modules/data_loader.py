import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from pandas_datareader import data as pdr
from datetime import datetime

# from scipy.stats import pairs

START_DATE = datetime(2015, 1, 1)

# FRED-Daten laden

def get_fred_data(series):
    """Lädt FRED-Daten für die angegebene Serie

    Args:
        series (str): Name der FRED-Serie

    Returns:
        pandas.DataFrame: Daten für die angegebene Serie
    """
    return pdr.DataReader(series, 'fred', START_DATE)

def load_policy_rates(rates_dict):
    data=[]
    for c, s in rates_dict.items():
        try:
            df = get_fred_data(s)
            df.columns = [c]
            data.append(df)
        except Exception as e:
            pass 
    return pd.concat(data, axis=1)

def load_cpi(CPI):
    """
    Lädt CPI-Daten für die angegebene Serie

    Args:
        CPI (dict): Dictionary mit Währungsnamen und FRED-Seriennamen

    Returns:
        pandas.DataFrame: Daten für die angegebene CPI-Serie
    """
    data = []
    for c, s in CPI.items():
        try:
            df = get_fred_data(s).pct_change(12)*100
            df.columns = [c]
            data.append(df)
        except Exception as e:
            pass
    return pd.concat(data, axis=1)

def load_growth(IP):
    data = []
    for c, s in IP.items():
        try:
            df = get_fred_data(s).pct_change(12)*100
            df.columns = [c]
            data.append(df)
        except Exception as e:
            pass
    return pd.concat(data, axis=1)

#FX-Daten laden

def load_fx_prices(Pairs):
    prices = {}
    for c, t in Pairs.items():
        ticker = t if t.endswith("=X") or t.endswith(".NYB") else f"{t}=X"
        df = yf.download(ticker, start=START_DATE)
        if df.empty or "Adj Close" not in df:
            continue
        prices[c] = df["Adj Close"]
    if not prices:
        return pd.DataFrame()
    return pd.concat(prices, axis=1)

#VIX laden
def load_risk():
    vix = yf.download("^VIX", start=START_DATE)["Adj Close"]
    vix.name = "VIX"
    return vix
