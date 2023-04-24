"""
Author: Mikeal888

Here we provide several useful fitting algorithms for fitting the data to a given model. 
We will use these functions in the next script to fit the data to a given model and then use the fitted model to make predictions.

We will use the following models:
1. Black Scholes
2. Heston

We will use the following fitting algorithms are in scikit-learn:
1. Linear Regression

We will use the following fitting algorithms are in scipy:
1. Least Squares
2. Maximum Likelihood
3. Bayesian

"""

# Import the libraries
import numpy as np
import pymc as pm
import scipy.stats as stats
from pymc.distributions.timeseries import EulerMaruyama


def stock_pdf(s, S0, t, mu, sigma):

    """
    This function computes the probability density function of the stock price at time t given the initial stock price S0, 
    the drift mu, the volatility sigma, and the stock price s

    Parameters
    ----------
    s : stock price at time t
    S0 : initial stock price
    t : time
    mu : drift
    sigma : volatility

    Returns
    -------
    pdf : probability density function of the stock price at time t
    """

    # Calculate the stock price at time t
    pdf = (1 / (s * sigma * np.sqrt(2 * np.pi * t))) * np.exp(-((np.log(s / S0) - (mu - 0.5 * sigma**2) * t)**2) / (2 * sigma**2 * t))

    return pdf


def stockprice_evolution(S0, t, dt, mu, sigma, n):

    "Simulate stock price according to geometric brownian motion"

    St = np.zeros((n, len(t)))
    St[:, 0] = S0

    for i in range(1, len(t)):
        St[:, i] = St[:, i-1] * (1 + mu * dt + sigma * np.sqrt(dt) * np.random.normal(0, 1, n))

    return St

def estimate_coeffs(series, n, dt):

    """
    Get a rough estimate of the parameters of geometric brownian motion

    Parameters
    ----------
    series : time series data of the log_underlying 
    n : number of time steps
    dt : time step

    Returns
    -------
    mu : drift
    sigma : volatility
    """

    # mu1 = mu - 0.5 * sigma**2
    mu1 = (series[-1] - series[0]) / (n * dt)
    sigma = np.sqrt( ( np.diff(series)**2 ).sum() / (n * dt) )
    mu =  mu1 + 0.5 * sigma**2

    return mu, sigma


def fit_geometric_brownian_motion(data, params_guess, dt):

    """
    This function uses Bayesian inference to provide an estimate of the parameters of geometric brownian motion.

    Parameters
    ----------
    data : time series data for some underlying stock
    params_guess : initial guess for the parameters of the model 0: mu, 1: variance in mu, 2: sigma
    dt : time step
    n_samples : number of samples to draw from the posterior distribution

    Returns
    -------
    params : the parameters of the model
    """

    def lin_sde(x, mu, sigma):
        return mu * x, sigma * x
    
    with pm.Model() as stock_model:
        # Define the parameters
        mu = pm.Normal("mu", mu=params_guess[0], sigma=params_guess[1])
        sigma = pm.HalfNormal("sigma", sigma=params_guess[2])

        # Define the stock price evolution
        stock = EulerMaruyama("stock", dt, lin_sde, (mu, sigma), shape=len(data), initval=data)

        # Define the likelihood
        likelihood = pm.Normal("likelihood", mu=stock, sigma=0.1, observed=data)

        # Inference button (TM)!
        trace = pm.sample()

    return trace


def normal_pdf(normaldata):

    """
    Create a normal probability density function using stats.norm.pdf
    """

    mean = np.mean(normaldata)
    std = np.std(normaldata)

    x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
    y = stats.norm.pdf(x, mean, std)

    return x, y

def infer_stock_pdf(mu_post, sigma_post, S0, t, s_range, ds = 0.01):

    """
    Infer the stock distrubition given data, initial parameters, time step, and time.
    Compute distribution using P(s) = \int \int P(s|\mu,\sigma) P(\mu)P(\sigma) d\mu d\sigma
    """

    # Get the pdfs of posterieor distributions
    mu_pdf_x, mu_pdf = normal_pdf(mu_post)
    sigma_pdf_x, sigma_pdf = normal_pdf(sigma_post)

    # Get the step size
    dmu = mu_pdf_x[1] - mu_pdf_x[0]
    dsigma = sigma_pdf_x[1] - sigma_pdf_x[0]

    # Get the stock price distribution
    s = np.arange(s_range[0], s_range[1], ds)

    # Compute maginal distribution
    pdfs = np.zeros((len(mu_pdf_x), len(sigma_pdf_x), len(s)))
    for i, mu in enumerate(mu_pdf_x):
        for j, sigma in enumerate(sigma_pdf_x):
            pdfs[i, j, :] = stock_pdf(s, S0, t, mu, sigma) * mu_pdf[i] * sigma_pdf[j]

    # sum over mu and sigma
    estimated_pdf = np.sum(pdfs, axis=(0, 1))*dmu*dsigma

    return s, estimated_pdf
    

def estimate_stock_distribution(data, params_guess, dt, t, S0, ds=0.01):

    """
    Estimate the stock distrubition given data, initial parameters, time step, and time.
    Compute distribution using P(s) = \int \int P(s|\mu,\sigma) P(\mu)P(\sigma) d\mu d\sigma
    """

    trace = fit_geometric_brownian_motion(data, params_guess, dt)

    # Extract the parameters
    mu_post = trace.posterior.get("mu").values.flatten()
    sigma_post = trace.posterior.get("sigma").values.flatten()

    # return the stock distribution
    return infer_stock_pdf(mu_post, sigma_post, S0, t, ds)

def log_normal_MGF(n, mu, sigma):

    """
    Compute the moment generating function of the log normal distribution
    """

    return np.exp(n * mu + 0.5 * n**2 * sigma**2)

