# Author: mikeal888 (github)
# Description: This file contains all the utility functions for the options pricing model

# ------------------ Imports ------------------ #

import numpy as np
import pandas as pd
import datetime as dt
import pandas_datareader.data as web
import yfinance as yf
import scipy.stats as stats
from scipy.optimize import brentq

# overide default yfinance download method
yf.pdr_override()


## ---------------- Retrieve Data ---------------- ##


def get_options_data(ticker: str, date: dt.date, type: str = "call") -> pd.DataFrame:
    """
    This function will return a pandas dataframe of the options data for a given ticker,
    date, and type. It will return the closest expirey date to the given date

    Parameters
    ----------
    ticker : str
        The ticker of the stock you want to get the options data for
    date : datetime.date
        The date you want to get the options data for
    type : str
        The type of option you want to get the data for. Must be either 'call' or 'put'

    Returns
    -------
    data : pd.DataFrame
        A pandas dataframe of the options data for the given ticker, date, and type
    """
    try:
        options = web.YahooOptions(ticker)
        options.headers = {"User-Agent": "Firefox"}
    except:
        print("Error: Could not get options data for ticker: ", ticker)
        return None

    # Get closest expirey date to date
    expirey = min(options._get_expiry_dates(), key=lambda x: abs(x - date))

    # Get the call data
    if type == "call":
        data = options.get_call_data(expiry=expirey)
    elif type == "put":
        data = options.get_put_data(expiry=expirey)
    else:
        raise ValueError("type must be either 'call' or 'put'")

    # The JSON data is the actual useful data that we want
    data.reset_index(inplace=True)
    data.set_index("Strike", inplace=True)
    data = data.JSON.apply(pd.Series)

    # Drop strike column since it is now the index
    data.drop("strike", axis=1, inplace=True)

    return data


def get_stock_data(
    ticker: str, start_date: dt.date, end_date: dt.date, interval: str = "1d"
) -> pd.DataFrame:
    """
    This function will return a pandas dataframe of the underlying data for a given ticker,
    start date, end date, and interval

    Parameters
    ----------
    ticker : str
        The ticker of the stock you want to get the options data for
    start_date : datetime.date
        The start date you want to get the underlying data for
    end_date : datetime.date
        The end date you want to get the underlying data for
    interval : str
        The interval you want to get the underlying data for. Must be either '1d', '1wk', '1mo'

    Returns
    -------
    data : pd.DataFrame
        A pandas dataframe of the underlying data for the given ticker, start date, end date, and interval
    """
    try:
        print("#------------------------------------------#")
        print("Getting underlying data for ticker: ", ticker)
        print("#------------------------------------------#")
        data = web.get_data_yahoo(ticker, start_date, end_date, interval=interval)
    except:
        print("Error: Could not get underlying data for ticker: ", ticker)
        return None

    return data


def get_repo_rate(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """
    This function will return the repo rate for a given date range
    Found at https://fred.stlouisfed.org/series/DFF
    Parameters
    ----------
    start_date : datetime.date
        The start date you want to get the repo rate for
    end_date : datetime.date
        The end date you want to get the repo rate for

    Returns
    -------
    data : pd.DataFrame
        A pandas dataframe of the repo rate for the given date range
    """

    return web.FredReader(symbols="DFF", start=start_date, end=end_date).read()


## ------------- Utility Functions ------------- ##


def stockprice_evolution(S0, t, dt, mu, sigma, n):
    "Simulate stock price according to geometric brownian motion"

    St = np.zeros((n, len(t)))
    St[:, 0] = S0

    for i in range(1, len(t)):
        St[:, i] = St[:, i - 1] * (
            1 + mu * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, n)
        )

    return St


## ------------- Pricing Models -------------- ##


def OptionsPriceBSM(S0, K, tau, sigma, r, option_type="call"):
    """
    Black Scholes Merton model for pricing European options

    Parameters
    ----------
    S0 : float
        The current stock price
    K : float
        The strike price
    tau : float
        The time to maturity
    sigma : float
        The volatility
    r : float
        The risk free rate
    option_type : str
        The type of option. Must be either 'call' or 'put'

    """

    def dp(S0, K, tau, sigma, r):
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return d1, d2

    d1, d2 = dp(S0, K, tau, sigma, r)

    if option_type == "call":
        return S0 * stats.norm.cdf(d1) - K * np.exp(-r * tau) * stats.norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * tau) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")


def OptionDeltaBSM(S0, K, tau, sigma, r, option_type="call"):
    """
    Compute the delta of a European option using the Black Scholes Merton model

    Parameters
    ----------
    S0 : float
        The current stock price
    K : float
        The strike price
    tau : float
        The time to maturity
    sigma : float
        The volatility
    r : float
        The risk free rate
    option_type : str
        The type of option. Must be either 'call' or 'put'

    """

    def dp(S0, K, tau, sigma, r):
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)
        return d1, d2

    d1, d2 = dp(S0, K, tau, sigma, r)

    if option_type == "call":
        return stats.norm.cdf(d1)
    elif option_type == "put":
        return stats.norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be either 'call' or 'put'")


def ImpliedVolatilityBSM(S0, K, tau, r, price, option_type="call"):
    """
    Compute the implied volatility of a European option using the Black Scholes Merton model
    using the Brentq method in scipy

    Parameters
    ----------
    S0 : float
        The current stock price
    K : float
        The strike price
    tau : float
        The time to maturity
    sigma : float
        The realised volatility
    r : float
        The risk free rate
    option_type : str
        The type of option. Must be either 'call' or 'put'
    """

    # compute the implied volatility
    implied_vol = brentq(
        lambda x: OptionsPriceBSM(S0, K, tau, x, r, option_type=option_type) - price,
        a=0.0001,
        b=10,
    )

    return implied_vol
