import numpy as np
import pandas as pd

from agent.delta_agent import DeltaAgent
from agent.trpo_agent import TRPOAgent
from env.call_option_env import CallOptionEnv

# Agent parameters
from utils.black_scholes import BlackScholesModel
from utils.heston import HestonModel

# Parameters used by the environment
environment_parameters = {
    "seed": 123,
    "size": 100,
    "maturity": 1 / 4,
    "strike_price": 1.0,
    "initial_price": 1.0,
    "discount_rate": 0.05,
    "returns": 0.1,
    "volatility": 0.1,
    "kappa": 0.5,
}

# Parameters used by the GBM Model data generator
blackscholes_parameters = {
    "seed": 123,
    "size": 100,
    "maturity": 1 / 4,
    "strike_price": 1.0,
    "initial_price": 1.0,
    "initial_vol": 0.1,
    "discount_rate": 0.05,
}

# Parameters used by the Heston Model data generator
heston_parameters = {
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


def run_trpo_path(environment, agent, env_info, agent_info):
    """
    :param environment: the environment class
    :param agent: the agent class
    :param env_info: the environment parameters for initializing the environment object
    :param agent_info: the agent parameters for initializing the agent object
    :return: the price movements by the environment and the actions taken by the agent
    """
    current_agent = agent()
    current_env = environment()

    current_agent.agent_init(agent_info)
    current_env.env_init(env_info)

    prices = []
    actions = []

    last_state = current_env.env_start()
    prices.append(last_state[0])

    [last_action] = current_agent.agent_start(last_state)
    actions.append(last_action)

    is_terminal = False

    while not is_terminal:
        (last_state, reward, term, _) = current_env.env_step(last_action)
        if term:
            current_agent.agent_end(reward)
        else:
            [last_action] = current_agent.agent_step(reward, last_state)
            prices.append(last_state[0])
            actions.append(last_action)
        # print(f'last state: {last_state}, last action: {last_action}, reward: {reward}')
        is_terminal = term

    return prices, actions


def run_delta_path(environment, agent, env_info, agent_info):
    """
    :param environment: the environment class
    :param agent: the agent class
    :param env_info: the environment parameters for initializing the environment object
    :param agent_info: the agent parameters for initializing the agent object
    :return: the price movements by the environment and the actions taken by the agent
    """
    current_agent = agent()
    current_env = environment()

    current_agent.agent_init(agent_info)
    current_env.env_init(env_info)

    prices = []
    actions = []

    last_state = current_env.env_start()
    prices.append(last_state[0])

    last_action = current_agent.agent_start(last_state)
    actions.append(last_action)

    is_terminal = False

    while not is_terminal:
        (last_state, reward, term, _) = current_env.env_step(last_action)
        if term:
            current_agent.agent_end(reward)
        else:
            last_action = current_agent.agent_step(reward, last_state)
            prices.append(last_state[0])
            actions.append(last_action)
        # print(f'last state: {last_state}, last action: {last_action}, reward: {reward}')
        is_terminal = term

    return prices, actions


def run_trpo(sim_class, sim_params):
    """
    :param sim_class: data generator simulation class
    :param sim_params: parameters for initializing the simulator object
    :return: pandas dataframe containing the time, underlying price and hedging action taken
    """
    env_info = {
        "seed": environment_parameters["seed"],
        "size": environment_parameters["size"],
        "maturity": environment_parameters["maturity"],
        "strike_price": environment_parameters["strike_price"],
        "initial_price": environment_parameters["initial_price"],
        "discount_rate": environment_parameters["discount_rate"],
        "returns": environment_parameters["returns"],
        "volatility": environment_parameters["volatility"],
        "kappa": environment_parameters["kappa"],
        "sim_class": sim_class,
        "sim_params": sim_params
    }
    maturity = env_info["maturity"]

    for cost in [100]:
        for freq in [90, 30, 15]:
            env_info["trading_cost"] = cost / 10000
            env_info["frequency"] = freq

            time = np.linspace(0, maturity, freq)

            log_dir = f"./{sim_class.__name__}/models/TRPO_{cost}_{freq}/"
            agent_info = {
                "log_dir": log_dir
            }

            current_env = CallOptionEnv
            current_agent = TRPOAgent

            price, action = run_trpo_path(current_env,
                                          current_agent,
                                          env_info,
                                          agent_info)
            # return pnl, stats, cost
            print(f"cost: {cost}, freq: {freq}")
            pd.DataFrame({'time': time,
                          'price': price,
                          'action': action}).to_csv(f"{log_dir}test_run.csv", index=False)


def run_delta(sim_class, sim_params):
    """
    :param sim_class: data generator simulation class
    :param sim_params: parameters for initializing the simulator object
    :return: pandas dataframe containing the time, underlying price and hedging action taken
    """
    env_info = {
        "seed": environment_parameters["seed"],
        "size": environment_parameters["size"],
        "maturity": environment_parameters["maturity"],
        "strike_price": environment_parameters["strike_price"],
        "initial_price": environment_parameters["initial_price"],
        "discount_rate": environment_parameters["discount_rate"],
        "returns": environment_parameters["returns"],
        "volatility": environment_parameters["volatility"],
        "kappa": environment_parameters["kappa"],
        "sim_class": sim_class,
        "sim_params": sim_params
    }
    maturity = env_info["maturity"]

    for cost in [100]:
        for freq in [90, 30, 15]:
            env_info["trading_cost"] = cost / 10000
            env_info["frequency"] = freq

            time = np.linspace(0, maturity, freq)

            log_dir = f"./{sim_class.__name__}/models/DELTA_{cost}_{freq}/"

            agent_info = {
                "size": environment_parameters["size"],
                "strike_price": environment_parameters["strike_price"],
                "discount_rate": environment_parameters["discount_rate"],
                "volatility": environment_parameters["volatility"],
                "seed": environment_parameters["seed"]
            }

            current_env = CallOptionEnv
            current_agent = DeltaAgent

            price, action = run_delta_path(current_env,
                                           current_agent,
                                           env_info,
                                           agent_info)
            # return pnl, stats, cost
            print(f"cost: {cost}, freq: {freq}")
            pd.DataFrame({'time': time,
                          'price': price,
                          'action': action}).to_csv(f"{log_dir}test_run.csv", index=False)


# Main program to run different agents using different data simulators
if __name__ == '__main__':
    run_trpo(BlackScholesModel, blackscholes_parameters)
    run_delta(BlackScholesModel, blackscholes_parameters)
    run_trpo(HestonModel, heston_parameters)
    run_delta(HestonModel, heston_parameters)
