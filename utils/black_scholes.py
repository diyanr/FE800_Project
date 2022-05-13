import numpy as np
from numpy.random import Generator, PCG64
from scipy.stats import norm

EPS = 1.0e-6


def BSCallValue(sigma, S0, K, r, T):
    if T < EPS:
        return max(S0 - K, 0)

    if sigma < EPS:
        return max(S0 - K * np.exp(-r * T), 0)

    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d_1) - norm.cdf(d_2) * K * np.exp(-r * T)


def BSPutValue(sigma, S0, K, r, T):
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return -S0 * norm.cdf(-d_1) + norm.cdf(-d_2) * K * np.exp(-r * T)


def callDelta(sigma, S0, K, r, T):
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d_1)


def putDelta(sigma, S0, K, r, T):
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d_1) - 1


def vega(sigma, S0, K, r, T):
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return S0 * norm.pdf(d_1) * np.sqrt(T)


def gamma(sigma, S0, K, r, T):
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d_1) / (S0 * sigma * np.sqrt(T))


class BlackScholesModel:
    """Data generator that simulates Geometrical Brownian Motion"""
    def __init__(self, model_params={}):
        self.rand_generator = Generator(PCG64(model_params.get("seed")))
        self.s_0 = model_params["initial_price"]
        self.k = model_params["strike_price"]
        self.v_0 = model_params["initial_vol"]
        self.r = model_params["discount_rate"]

        self.s_t = self.s_0
        self.v_t = self.v_0

    def reset(self):
        self.s_t = self.s_0
        self.v_t = self.v_0

    def new_price(self, dt):
        nudt = (self.r - 0.5 * self.v_t) * dt
        sigsdt = np.sqrt(self.v_t * dt)
        z = self.rand_generator.standard_normal()
        s_1 = self.s_t * np.exp(nudt + sigsdt * z)
        v_1 = self.v_t

        self.s_t = s_1

        return s_1, v_1

