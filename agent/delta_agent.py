#!/usr/bin/env python

"""Random agent class for RL-Glue-py.
"""

from agent.agent import BaseAgent
from utils.black_scholes import *


# Create SimpleAgent
class DeltaAgent(BaseAgent):
    def __init__(self):
        """Initialize an empty agent object"""
        self.rand_generator = None
        self.size = None
        self.strike = None
        self.discount_rate = None
        self.volatility = None
        self.last_state = None
        self.last_action = None

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Assume agent_info dict contains:
        {
            size: int,
            strike_price: float,
            discount_rate: float,
            volatility: float,
            seed: int
        }
        """

        # set random seed for each run
        self.rand_generator = Generator(PCG64(agent_info.get("seed")))

        # set class attributes
        self.size = agent_info.get("size")
        self.strike = agent_info.get("strike_price")
        self.discount_rate = agent_info.get("discount_rate")
        self.volatility = agent_info.get("volatility")

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            self.last_action [float] : The first action the agent takes.
        """

        old_price, old_ttm, old_holding = state
        # Calculate action based on Delta hedging
        self.last_action = callDelta(np.sqrt(self.volatility), old_price, self.strike, self.discount_rate, old_ttm)
        self.last_state = state

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward [float]: the reward received for taking the last action taken
            state [int]: the state from the environment's step, where the agent ended up after the last step
        Returns:
            self.last_action [float] : The action the agent is taking.
        """

        self.last_state = state
        old_price, old_ttm, old_holding = self.last_state

        self.last_action = callDelta(np.sqrt(self.volatility), old_price, self.strike, self.discount_rate, old_ttm)
        # ----------------
        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        return
