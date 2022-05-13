from sb3_contrib import TRPO

from utils.model_utils import SaveOnBestTrainingRewardCallback, SaveLearningCurve


class TrpoModel:
    def __init__(self):
        self.model = None

    def model_init(self, environment, tb_logs):
        self.model = TRPO("MlpPolicy", environment,  # gamma=1.0,
                          verbose=0, tensorboard_log=tb_logs)

    def model_train(self, tr_logs, time_steps):
        callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=tr_logs)
        for i in range(2):
            self.model.learn(total_timesteps=time_steps,
                             reset_num_timesteps=False,
                             tb_log_name="TRPO",
                             callback=callback)
        SaveLearningCurve(tr_logs).plot_results()
        return self.model


if __name__ == '__main__':
    log_dir = "./TRPO_0_15/"
    SaveLearningCurve(log_dir).plot_results()
    # os.makedirs(log_dir, exist_ok=True)
    #
    # tb_log_dir = "./tensorboard/"
    # os.makedirs(tb_log_dir, exist_ok=True)
    #
    # # Create the environment
    # environment_parameters = {
    #     "size": 100,
    #     "maturity": 1 / 4,
    #     "strike_price": 100,
    #     "initial_price": 100,
    #     "discount_rate": 0.05,
    #     "returns": 0.1,
    #     "volatility": 0.3,
    #     "frequency": 90,
    #     "seed": 123,
    #     "trading_cost": 0.0,
    #     "kappa": 0.5
    # }
    # env = CallOptionEnv()
    # env.env_init(environment_parameters)
    # # Logs will be saved in log_dir/monitor.csv
    # env = Monitor(env, log_dir)
    #
    # TIMESTEPS = 100000
    # model = TrpoModel()
    # model.model_init(env, tb_log_dir)
    # trained_model = model.model_train(log_dir, TIMESTEPS)
    #
    # mean_reward, std_reward = evaluate_policy(trained_model, env, n_eval_episodes=100)
    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
