#!/usr/bin/env python

"""Random agent class for RL-Glue-py.
"""

from numpy.random import Generator, PCG64

from agent.agent import BaseAgent


# Create SimpleAgent
class ZeroAgent(BaseAgent):
    def __init__(self):
        self.seed = 0
        self.rand_generator = None
        self.last_state = None
        self.last_action = None

    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            size: 500 [int],
            step_size: float, 
            discount_factor: float,
            seed: int
        }
        """

        # set random seed for each run
        self.rand_generator = Generator(PCG64(agent_info.get("seed")))

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            self.last_action [int] : The first action the agent takes.
        """

        # your code here
        # self.last_state = state       
        self.last_action = 0.0
        self.last_state = state
        # ----------------

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward [float]: the reward received for taking the last action taken
            state [int]: the state from the environment's step, where the agent ended up after the last step
        Returns:
            self.last_action [int] : The action the agent is taking.
        """
        
        self.last_state = state
        self.last_action = 0.0
        # ----------------
        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        return
        
