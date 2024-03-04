import numpy as np
import pandas as pd
import seaborn as sns
from reactor_environments import Environment
from agents_pg import PGAgent
# import tensorflow as tf
import matplotlib.pyplot as plt
import time
    
    
if __name__ == "__main__":
    env = Environment(timesteps=40, num_j_temp=40)
    agent = PGAgent(learning_rate=1e-6, decay_rate=1e-7, environment=env, nn_arch=[400, 300, 200])
    #agent.P = tf.keras.models.load_model(r"C:\Users\Dr Nabil\Downloads\Policy gradient\Pg_final\P_network.h5")
    env.testing = True
    start_time = time.perf_counter()
    episode_versus_reward = agent.train(10)
    cpu_time = time.perf_counter() - start_time
    agent.P.save("P_network.h5")
    state_arr = np.zeros_like(env.time_list)
    conc_arr = np.zeros_like(env.time_list)
    action_arr = np.zeros_like(env.time_list)
    reward_arr = np.zeros_like(env.time_list)
    episode_vs_reward_df = pd.DataFrame({"Episodes": episode_versus_reward[:, 0], "Reward": episode_versus_reward[:, 1]})
    env.curr_state = np.vstack([0, 298, 0.6])
    state = env.curr_state
    for i in range(env.n_tf):
        state_arr[i] = state[env.T, 0]
        conc_arr[i] = state[env.Ca, 0]
        action_arr[i] = agent.choose_action(state)
        #reward_computation
        next_state, reward, done, info = env.step(action_arr[i])
        reward_arr[i] = reward
        state = next_state
    df = pd.DataFrame({"Temperature": state_arr, "Time": env.time_list, "Reference": env.Tref, "Jacket Temperature": action_arr, "Concentration [A]": conc_arr, "Reward": reward_arr})
#    df.to_excel("PG.xlsx")
    sns.set_theme()
    plt.figure(1)
    sns.lineplot(
        data=df,
        x="Time",
        y="Temperature",
        label="Reactor Temperature",
    )
    sns.lineplot(
        data=df,
        x="Time",
        y="Reference",
        legend="full",
        label="Reference Temperature",
    )
    plt.figure(2)
    sns.lineplot(
        data=df,
        x="Time",
        y="Jacket Temperature",
        legend="full",
        label="Action",
    )
    
    # concentration plot
    plt.figure(3)
    sns.lineplot(data=df, x="Time", y="Concentration [A]")
    window_size = 100
    rolling_avg = pd.Series(episode_versus_reward[:,1]).rolling(window=window_size, min_periods=1).mean()
    data = pd.DataFrame({'Episode': range(1, episode_versus_reward.shape[0] + 1),
                         'Reward':episode_versus_reward[:,1] ,
                         'Rolling Average': rolling_avg})
    plt.figure(4)
    sns.set_style("darkgrid")
    sns.lineplot(data=data, x='Episode', y = 'Reward', label='Episode Reward', color='blue')
    sns.lineplot(data=data, x='Episode', y = 'Rolling Average', label='Rolling Average', color='red')
    #sns.lineplot(episode_vs_reward_df, x="Episodes", y="Reward")
    plt.show()
    # control signal plot
    # sns.lineplot(x=env.time_list, y=action_arr, drawstyle='steps-pre', label="Jacket Temperature")
    # episode versus reward plot
    # sns.lineplot(episode_vs_reward_df, x="Episodes", y="Reward")
    # plt.show()
    MAE = np.sum(np.abs(state_arr-env.Tref))/len(state_arr)
    RMSE = (np.sum((state_arr-env.Tref)**2)/len(state_arr))**0.5
    days = int(cpu_time // 86400)
    hrs = int((cpu_time -  86400 * days)// 3600)
    mins = int((cpu_time - 3600 * hrs - 86400 * days) // 60)
    seconds = int((cpu_time - 60 * mins - 3600 * hrs - 86400 * days) // 1)
    print(f"{days = }\n{hrs = }\n{mins = }\n{seconds = }")  