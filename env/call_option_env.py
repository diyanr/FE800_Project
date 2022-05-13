import numpy as np
from numpy.random import Generator, PCG64
import gym
from gym import spaces
import matplotlib
from utils.black_scholes import *

EPS = 1.0e-6
TICK_SIZE = 0.1
COST_MULTIPLIER = 1.0
TRADING_COST = 0.01
RISK_AVERSION = 1
KAPPA = 0.1


class CallOptionEnv(gym.Env):
    """A call option environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """Initialize an empty environment object"""
        self.rand_generator = None
        self.size = None
        self.maturity = None
        self.strike = None
        self.initial_price = None
        self.discount_rate = None
        self.returns = None
        self.volatility = None
        self.sigma = None
        self.frequency = None
        self.dt = None
        self.trading_cost = None
        self.kappa = None

        self.terminal = None
        self.reward = None
        self.state = None

        self.portfolio = 0.0
        self.pnl_lst = []
        self.cost = 0.0

        self.sim_class = None
        self.sim_params = None
        self.simulator = None

    def env_init(self, env_info):
        """Initialize an environment object with the environment parameters"""
        # set random seed for each run
        self.rand_generator = Generator(PCG64(env_info.get("seed")))

        # load data from env_info
        self.size = env_info["size"]
        self.maturity = env_info["maturity"]
        self.strike = env_info["strike_price"]
        self.initial_price = env_info["initial_price"]
        self.discount_rate = env_info["discount_rate"]
        self.volatility = env_info["volatility"]
        self.sigma = np.sqrt(self.volatility)
        self.frequency = env_info["frequency"]
        self.dt = self.maturity / self.frequency
        self.trading_cost = env_info["trading_cost"]
        self.kappa = env_info["kappa"]

        self.sim_class = env_info["sim_class"]
        self.sim_params = env_info["sim_params"]
        self.simulator = self.sim_class(self.sim_params)

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

    def step(self, actions):
        """Take a single step in the environment
        :param actions: the action taken by the agent
        :return: the state, reward, terminal state and any additional information about the environment
        """
        old_price, old_ttm, old_holding = self.state

        # Calculate old option price based on Black-Scholes-Merton
        old_opt_price = BSCallValue(self.sigma, old_price, self.strike, self.discount_rate, old_ttm)

        # Calculate new stock price and vol based on the data simulator used
        new_price, new_vol = self.simulator.new_price(self.dt)
        self.sigma = np.sqrt(new_vol)

        # Advance the time step
        new_ttm = max(old_ttm - self.dt, 0.0)
        new_holding = actions

        # Calculate new option price based on Black-Scholes-Merton
        new_opt_price = BSCallValue(self.sigma, new_price, self.strike, self.discount_rate, new_ttm)

        # Calculate the incremental change in costs and rewards
        cost = abs(new_holding - old_holding) * old_price * self.trading_cost
        reward = (new_opt_price - old_opt_price) - (new_price - old_price) * new_holding - cost

        # Calculate new portfolio value
        portfolio = self.portfolio * np.exp(
            self.discount_rate * self.dt) if self.maturity - old_ttm > EPS else self.portfolio
        portfolio = portfolio + old_holding * old_price - new_holding * old_price - cost

        # Determine if maturity and liquidate any positions to payoff the contract
        self.terminal = bool(new_ttm < EPS)
        if self.terminal:
            cost = cost + abs(new_holding) * new_price * self.trading_cost
            reward = reward - new_holding * new_price * self.trading_cost
            portfolio = portfolio - max(new_price - self.strike, 0)

        # Calculate reward to be returned
        self.reward = float(self.size * reward - self.kappa * (self.size * reward) ** 2)

        # Setup the variables to be returned
        self.state = np.array([new_price, new_ttm, new_holding], dtype=np.float32)
        self.portfolio = portfolio
        self.pnl_lst.append(self.portfolio)
        self.cost = self.cost + cost

        return self.state, self.reward, self.terminal, {"portfolio": portfolio}

    def env_step(self, actions):
        """Same as step to meet RL Glue specification"""
        return self.step(actions)

    def reset(self):
        """Reset the environment for each new episode"""
        self.cost = 0.0
        self.pnl_lst = []
        self.simulator.reset()
        self.sigma = np.sqrt(self.volatility)

        time = self.maturity
        price = self.initial_price
        holding = 0
        portfolio = BSCallValue(self.sigma, price, self.strike, self.discount_rate, time)
        # portfolio = self.simulator.call_value(time)

        self.state = np.array([price, time, holding], dtype=np.float32)
        self.portfolio = portfolio
        self.pnl_lst.append(self.portfolio)

        return self.state

    def env_start(self):
        """Same as reset for the RL Glue specification"""
        return self.reset()

    def render(self, mode='human'):
        return self.reward

    def close(self):
        pass

    def env_message(self, message):
        """A message asking the environment for information

        Args:
            message: the message passed to the environment

        Returns:
            the response (or answer) to the message
        """
        if message == 'get portfolio value':
            return self.portfolio
        if message == 'get pnl list':
            return np.array(self.pnl_lst, dtype=np.float32)
        if message == 'get transaction cost':
            return self.cost
