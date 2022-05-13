#!/usr/bin/env python

"""Random agent class for RL-Glue-py.
"""

import os

from sb3_contrib import TRPO

from agent.agent import BaseAgent


# Create SimpleAgent
class TRPOAgent(BaseAgent):
    def __init__(self):
        """Initialize and empty agent object"""
        self.last_action = None
        self.last_state = None
        self.model = None
        self.log_dir = None
        self.save_path = None
        
    def agent_init(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Set parameters needed to setup the semi-gradient TD(0) state aggregation agent.

        Assume agent_info dict contains:
        {
            log_dir: string
        }
        """

        # Create model based on best model saved during training
        self.log_dir = agent_info.get("log_dir")
        self.save_path = os.path.join(self.log_dir, 'best_model.zip')
        self.model = TRPO.load(self.save_path)

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            self.last_action [int] : The first action the agent takes.
        """

        # Use model to predict next action
        action, _ = self.model.predict(state, deterministic=True)
        self.last_action = action
        self.last_state = state

        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward [float]: the reward received for taking the last action taken
            state [int]: the state from the environment's step, where the agent ended up after the last step
        Returns:
            self.last_action [int] : The action the agent is taking.
        """

        # Use model to predict next action
        action, _ = self.model.predict(state, deterministic=True)
        self.last_action = action
        self.last_state = state

        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        return
        
