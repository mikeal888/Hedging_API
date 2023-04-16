"""

Author: @mikeal888

The goal of this script is to format all the data for a given ticker.
We will create a dataframe that returns all the options data for a given 
1. expirey date, 
2. option type
3. Underlying stock price over a given time period
We will then format this into a useful pandas dataframe for use in the next script
"""
# Import the libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import datetime as dt
import pandas_datareader.data as web
import yfinance as yf

# overide default yfinance download method
yf.pdr_override()
# set backend of pandas to plotly
pd.options.plotting.backend = "plotly"

class OptionsData:

    def __init__(self, ticker: str, expirey: dt.date, option_type: str):
        self.ticker = ticker
        self.expirey = expirey
        self.option_type = option_type
        self.options_data = self.get_options_data()

    
    def get_options_data(self) -> pd.DataFrame:
        """ 
        This function will return a pandas dataframe of the options data for a given ticker, 
        expirey date, and option type. It will return the closest expirey date to the given date
        
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
            A pandas dataframe of the options data for the given ticker, expirey date, and option type
        """
        try:    
            options = web.YahooOptions(self.ticker)
            options.headers = {'User-Agent': 'Firefox'}
        except:
            print("Error: Could not get options data for ticker: ", self.ticker)
            return None
    
        # Get closest expirey date to date and reset self.expirey to
        expirey_new = min(options._get_expiry_dates(), key=lambda x: abs(x - self.expirey))
        if self.expirey != expirey_new:
            print("#---------------------#")
            print("Closest expirey date to", self.expirey, "is", expirey_new)
            print("Using expirey date:", expirey_new)
            print("#---------------------#")
            self.expirey = expirey_new
        

        # Get the call data
        if type == "call":
            data = options.get_call_data(expiry=self.expirey)
        elif type == "put":
            data = options.get_put_data(expiry=self.expirey)
        else:
            raise ValueError("option_type must be either 'call' or 'put'")

        # The JSON data is the actual useful data that we want
        data.reset_index(inplace=True)
        data.set_index("Strike", inplace=True)
        data = data.JSON.apply(pd.Series)

        #Drop strike column since it is now the index
        data.drop("strike", axis=1, inplace=True)

        return data

    def get_underlying_data(self, start: dt.date, end: dt.date, interval: str = "1d") -> pd.DataFrame:
        """ 
        This function will return a pandas dataframe of the underlying data for a given ticker, 
        start date, end date, and interval

        Parameters
        ----------
        ticker : str
            The ticker of the stock you want to get the options data for
        start : datetime.date
            The start date you want to get the underlying data for
        end : datetime.date
            The end date you want to get the underlying data for
        interval : str
            The interval you want to get the underlying data for. Must be either '1d', '1wk', '1mo'
        
        Returns
        -------
        data : pd.DataFrame
            A pandas dataframe of the underlying data for the given ticker, start date, end date, and interval
        """
        try:
            print("#---------------------#")
            print("Getting underlying data for ticker: ", self.ticker)
            print("#---------------------#")
            data = web.get_data_yahoo(self.ticker, start, end, interval=interval)
        except:
            print("Error: Could not get underlying data for ticker: ", self.ticker)
            return None
        
        return data
    
def get_stock_data(ticker: str, start_date: dt.date, end_date: dt.date, interval: str = "1d") -> pd.DataFrame:
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
        print("#---------------------#")
        print("Getting underlying data for ticker: ", ticker)
        print("#---------------------#")
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


    

if __name__ == "__main__":
    # get the data
    data = OptionsData("AAPL", dt.date(2021, 5, 7), "call").data
    # print the data
    print(data)