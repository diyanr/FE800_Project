import os

import pandas as pd
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from agent.delta_agent import DeltaAgent
from agent.trpo_agent import TRPOAgent
from env.call_option_env import CallOptionEnv
from models.trpo_model import TrpoModel
from utils.black_scholes import BlackScholesModel
from utils.experiment import run_experiment_trpo, run_experiment_delta
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


def train_trpo(sim_class, sim_params):
    """
    :param sim_class: data generator simulation class
    :param sim_params: parameters for initializing the simulator object
    :return: None. Trained model is saved during training.
    """
    TIMESTEPS = 100000

    tb_log_dir = f"./{sim_class.__name__}/models/tensorboard/"
    os.makedirs(tb_log_dir, exist_ok=True)

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

    for cost in [100]:
        for freq in [90, 30, 15]:
            env_info["trading_cost"] = cost / 10000
            env_info["frequency"] = freq

            log_dir = f"./{sim_class.__name__}/models/TRPO_{cost}_{freq}/"
            os.makedirs(log_dir, exist_ok=True)

            env = CallOptionEnv()
            env.env_init(env_info)
            env = Monitor(env, log_dir)

            model = TrpoModel()
            model.model_init(env, tb_log_dir)
            trained_model = model.model_train(log_dir, TIMESTEPS)

            mean_reward, std_reward = evaluate_policy(trained_model, env, n_eval_episodes=100)
            print(f"cost: {cost}, freq: {freq}, mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


def test_trpo(sim_class, sim_params):
    """
    :param sim_class: data generator simulation class
    :param sim_params: parameters for initializing the simulator object
    :return: None. Saves the underlying prices, PnL stats and transaction costs for the test runs.
    """
    experiment_info = {
        "num_runs": 1,
        "num_episodes": 10000
    }

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

    for cost in [100]:
        for freq in [90, 30, 15]:
            env_info["trading_cost"] = cost / 10000
            env_info["frequency"] = freq

            log_dir = f"./{sim_class.__name__}/models/TRPO_{cost}_{freq}/"
            agent_info = {
                "log_dir": log_dir
            }

            current_env = CallOptionEnv
            current_agent = TRPOAgent

            pnl, stats, transaction_cost = run_experiment_trpo(current_env,
                                                               current_agent,
                                                               env_info,
                                                               agent_info,
                                                               experiment_info)
            # return pnl, stats, cost
            print(f"cost: {cost}, freq: {freq}")
            pd.DataFrame({'pnl': pnl,
                          'stats': stats,
                          'cost': transaction_cost}).to_csv(f"{log_dir}test_results.csv",
                                                            index=False)


def test_delta(sim_class, sim_params):
    """
    :param sim_class: data generator simulation class
    :param sim_params: parameters for initializing the simulator object
    :return: None. Saves the underlying prices, PnL stats and transaction costs for the test runs.
    """
    experiment_info = {
        "num_runs": 1,
        "num_episodes": 10000
    }

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

    for cost in [100]:
        for freq in [90, 30, 15]:
            env_info["trading_cost"] = cost / 10000
            env_info["frequency"] = freq

            log_dir = f"./{sim_class.__name__}/models/DELTA_{cost}_{freq}/"
            os.makedirs(log_dir, exist_ok=True)

            agent_info = {
                "size": environment_parameters["size"],
                "strike_price": environment_parameters["strike_price"],
                "discount_rate": environment_parameters["discount_rate"],
                "volatility": environment_parameters["volatility"],
                "seed": environment_parameters["seed"]
            }

            current_env = CallOptionEnv
            current_agent = DeltaAgent

            pnl, stats, transaction_cost = run_experiment_delta(current_env,
                                                                current_agent,
                                                                env_info,
                                                                agent_info,
                                                                experiment_info)
            # return pnl, stats, cost
            print(f"cost: {cost}, freq: {freq}")
            pd.DataFrame({'pnl': pnl,
                          'stats': stats,
                          'cost': transaction_cost}).to_csv(f"{log_dir}test_results.csv",
                                                            index=False)


# Main program to train and test the different agents using different data simulators
if __name__ == '__main__':
    train_trpo(HestonModel, heston_parameters)
    train_trpo(BlackScholesModel, blackscholes_parameters)
    test_trpo(HestonModel, heston_parameters)
    test_trpo(BlackScholesModel, blackscholes_parameters)
    test_delta(BlackScholesModel, blackscholes_parameters)
    test_delta(HestonModel, heston_parameters)
