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
import pandas as pd
import sklearn.linear_model as lm
import pymc3 as pm
import matplotlib.pyplot as plt


obs_y = np.random.normal(0.7, 0.3, 2000)

with pm.Model() as exercise1:

    std = pm.HalfNormal('std', sd=1)
    mu = pm.Normal('mu', mu=0, sd=1)

    y = pm.Normal('y', mu=mu, sd=std, observed=obs_y)

    trace = pm.sample(1000, tune=1000, cores=1)

    pm.traceplot(trace)
    plt.show()