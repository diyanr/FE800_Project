import numpy as np
from numpy.random import Generator, PCG64
from scipy.integrate import quad


def heston_charfunc(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    # constants
    a = kappa * theta
    b = kappa + lambd

    # common terms w.r.t phi
    rspi = rho * sigma * phi * 1j

    # define d parameter given phi and b
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 + (phi * 1j + phi ** 2) * sigma ** 2)

    # define g parameter given phi, b and d
    g = (b - rspi + d) / (b - rspi - d)

    # calculate characteristic function by components
    exp1 = np.exp(r * phi * 1j * tau)
    term2 = S0 ** (phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g)) ** (-2 * a / sigma ** 2)
    exp2 = np.exp(a * tau * (b - rspi + d) / sigma ** 2 + v0 * (b - rspi + d) * (
            (1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma ** 2)

    return exp1 * term2 * exp2


def integrand(phi, S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = np.exp(r * tau) * heston_charfunc(phi - 1j, *args) - K * heston_charfunc(phi, *args)
    denominator = 1j * phi * K ** (1j * phi)
    return numerator / denominator


class HestonModel:
    def __init__(self, model_params={}):
        self.rand_generator = Generator(PCG64(model_params.get("seed")))
        self.rho = model_params["rho"]
        self.kappa = model_params["kappa"]
        self.theta = model_params["theta"]
        self.sigma = model_params["sigma"]
        self.lambd = model_params["lambda"]
        self.s_0 = model_params["initial_price"]
        self.k = model_params["strike_price"]
        self.v_0 = model_params["initial_vol"]
        self.r = model_params["discount_rate"]

        self.s_t = self.s_0
        self.v_t = self.v_0

        self.mu = np.array([0, 0])
        self.cov = np.array([[1, self.rho],
                             [self.rho, 1]])

    def reset(self):
        self.s_t = self.s_0
        self.v_t = self.v_0

    def new_price(self, dt):
        z_1, z_2 = self.rand_generator.multivariate_normal(self.mu, self.cov)
        s_1 = self.s_t * np.exp((self.r - 0.5 * self.v_t) * dt + np.sqrt(self.v_t * dt) * z_1)
        v_1 = np.maximum(
            self.v_t + self.kappa * (self.theta - self.v_t) * dt + self.sigma * np.sqrt(self.v_t * dt) * z_2,
            0)
        self.s_t = s_1
        self.v_t = v_1
        return s_1, v_1

    # def call_value(self, tau):
    #     # args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    #     args = (self.s_t, self.k, self.v_t, self.kappa, self.theta, self.sigma, self.rho, self.lambd, tau, self.r)
    #
    #     real_integral, err = np.real(quad(integrand, 0, 100, args=args))
    #
    #     return (self.s_t - self.k * np.exp(-self.r * tau)) / 2 + real_integral / np.pi

    def call_value(self, tau):
        # args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
        args = (self.s_t, self.v_t, self.kappa, self.theta, self.sigma, self.rho, self.lambd, tau, self.r)

        P, umax, N = 0, 100, 10000
        dphi = umax / N  # dphi is width

        for i in range(1, N):
            # rectangular integration
            phi = dphi * (2 * i + 1) / 2  # midpoint to calculate height
            numerator = np.exp(self.r * tau) * heston_charfunc(phi - 1j, *args) - self.k * heston_charfunc(phi, *args)
            denominator = 1j * phi * self.k ** (1j * phi)

            P += dphi * numerator / denominator

        return np.real((self.s_t - self.k * np.exp(-self.r * tau)) / 2 + P / np.pi)


if __name__ == '__main__':
    heston_params = {
        "seed": 123,
        "strike_price": 1.0,
        "initial_price": 1.0,
        "initial_vol": 0.1,
        "discount_rate": 0.05,
        "kappa": 4.998769458253997,
        "theta": 0.1,
        "sigma": 0.7838890228345996,
        "rho": 0.0,
        "lambda": -0.4537780742202474
    }
    T = 1 / 4
    N = 90
    dt = T / N
    model = HestonModel(heston_params)
    for i in range(N):
        print(model.new_price(dt))
