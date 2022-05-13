from scipy import stats

from agent.trpo_agent import TRPOAgent
from env.call_option_env import CallOptionEnv
from utils.rl_glue import RLGlue


def run_experiment_trpo(environment, agent, env_info, agent_info, experiment_parameters):
    """
    :param environment: the environment class
    :param agent: the agent class
    :param env_info: the environment parameters for initializing the environment object
    :param agent_info: the agent parameters for initializing the agent object
    :param experiment_parameters: the parameters for the experiment
    :return: a tuple of PnL, PnL stats and transaction costs arrays
    """
    rl_glue = RLGlue(environment, agent)

    # one agent setting
    stat = []
    pnl = []
    cost = []
    # pnl = np.empty(experiment_parameters["num_episodes"]+1)
    for run in range(1, experiment_parameters["num_runs"] + 1):
        rl_glue.rl_init(agent_info, env_info)
        # pnl.append(rl_glue.rl_env_message("get portfolio value"))

        for episode in range(1, experiment_parameters["num_episodes"] + 1):
            # run episode
            rl_glue.rl_episode(0)  # no step limit
            env_pnl_lst = rl_glue.rl_env_message("get pnl list")
            # print(env_pnl_lst)
            stat.append(stats.ttest_1samp(env_pnl_lst, 0.0)[0])
            [env_pnl] = rl_glue.rl_env_message("get portfolio value")
            # print(env_pnl)
            pnl.append(env_pnl)
            [env_cost] = rl_glue.rl_env_message("get transaction cost")
            # print(env_cost)
            cost.append(env_cost)

    return pnl, stat, cost


def run_experiment_delta(environment, agent, env_info, agent_info, experiment_parameters):
    """
    :param environment: the environment class
    :param agent: the agent class
    :param env_info: the environment parameters for initializing the environment object
    :param agent_info: the agent parameters for initializing the agent object
    :param experiment_parameters: the parameters for the experiment
    :return: a tuple of PnL, PnL stats and transaction costs arrays
    """
    rl_glue = RLGlue(environment, agent)

    # one agent setting
    stat = []
    pnl = []
    cost = []
    # pnl = np.empty(experiment_parameters["num_episodes"]+1)
    for run in range(1, experiment_parameters["num_runs"] + 1):
        rl_glue.rl_init(agent_info, env_info)
        # pnl.append(rl_glue.rl_env_message("get portfolio value"))

        for episode in range(1, experiment_parameters["num_episodes"] + 1):
            # run episode
            rl_glue.rl_episode(0)  # no step limit
            env_pnl_lst = rl_glue.rl_env_message("get pnl list")
            # print(env_pnl_lst)
            stat.append(stats.ttest_1samp(env_pnl_lst, 0.0)[0])
            env_pnl = rl_glue.rl_env_message("get portfolio value")
            # print(env_pnl)
            pnl.append(env_pnl)
            env_cost = rl_glue.rl_env_message("get transaction cost")
            # print(env_cost)
            cost.append(env_cost)

    return pnl, stat, cost


# Main program for testing an experimental run
if __name__ == '__main__':
    experiment_parameters = {
        "num_runs": 1,
        "num_episodes": 10,
    }

    # Create the environment
    environment_parameters = {
        "size": 100,
        "maturity": 1 / 4,
        "strike_price": 100,
        "initial_price": 100,
        "discount_rate": 0.05,
        "returns": 0.1,
        "volatility": 0.3,
        "frequency": 90,
        "seed": 123,
        "trading_cost": 0.0,
        "kappa": 0.5
    }
    agent_parameters = {
        "log_dir": "../models/TRPO1/"
    }

    current_env = CallOptionEnv
    current_agent = TRPOAgent

    pnl, stats, cost = run_experiment_trpo(current_env,
                                           current_agent,
                                           environment_parameters,
                                           agent_parameters,
                                           experiment_parameters)
