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
import pymc3 as pm
from pymc3.distributions.timeseries import EulerMaruyama


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

def fit_geometric_brownian_motion(data, params_guess, dt, n_samples=1000):

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
    
    with pm.Model() as model:
        # Define the parameters
        mu = pm.Normal("mu", mu=params_guess[0], sigma=params_guess[1])
        sigma = pm.HalfNormal("sigma", sigma=params_guess[2])

        # Define the stock price evolution
        stock = EulerMaruyama("stock", dt, lin_sde, (mu, sigma), shape=len(data), testval=data)

        # Define the likelihood
        likelihood = pm.Normal("likelihood", mu=stock, sigma=0.1, observed=data)

        # Inference button (TM)!
        trace = pm.sample(n_samples, tune=1000, cores=1)

    # Extract the parameters
    mu_vals = trace.get_values("mu")
    sigma_vals = trace.get_values("sigma")

    return mu_vals, sigma_vals


def estimate_stock_distribution(S0, t, mu_est, sigma_est):

    """
    Using our estimates of mu and sigma, compute the distribution of stock price using marginals
    P(s) = \int \int P(s|\mu,\sigma) P(\mu)P(\sigma) d\mu d\sigma

    Parameters
    ----------
    S0 : initial stock price
    t : final time
    mu_est : estimated mu (list)
    sigma_est : estimated sigma (list)

    Returns
    -------
    s : stock price
    """

    # Compute mean and stdof mu and sigma
    mu_mean = np.mean(mu_est)
    mu_std = np.std(mu_est)
    sigma_mean = np.mean(sigma_est)
    sigma_std = np.std(sigma_est)

    # get probabilities of mu histogram
    dmu = 0.01
    dsigma = 0.01
    mus = np.arange(mu_mean - 4 * mu_std, mu_mean + 4 * mu_std, dmu)
    sigmas = np.arange(sigma_mean - 4 * sigma_std, sigma_mean + 4 * sigma_std, dsigma)

    # get normal distribution for mu
    mu_pdf = stats.norm.pdf(mus, mu_mean, mu_std)
    sigma_pdf = stats.norm.pdf(sigmas, sigma_mean, sigma_std)
