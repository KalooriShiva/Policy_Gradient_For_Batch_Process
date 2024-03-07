import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt



class PGAgent:
    
    def __init__(self, environment, learning_rate, decay_rate, discount_factor=0.95, epsilon=0.05,entropy_coeff=1e-1,  nn_arch=[400, 300, 200]):
        self.env = environment
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.P = self._get_model(nn_arch)
        self.entropy_coeff = tf.cast(entropy_coeff, dtype=tf.dtypes.float32)
        self.opt = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate, decay=decay_rate)
        self.action_memory = []
        self.reward_memory = []
        self.state_memory = []
        
        
    def _get_model(self, nn_arch):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(nn_arch[0], input_shape=(3,), activation="relu"))
        for num_neurons in nn_arch[1:]:
            model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.num_j_temp, activation="softmax"))
        model.build()
        return model
    
    def choose_action(self, state):
        scaled_state = state.copy()
        scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
        scaled_state[2,0] = (scaled_state[2,0])*80
        action_index = np.argmax(self.P(np.reshape(scaled_state, (1, -1)))[0])
        return action_index
   
    def choose_action_training(self,state):
        scaled_state = state.copy()
        scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
        scaled_state[2,0] = (scaled_state[2,0])*80
        if np.random.uniform() > self.epsilon: action_index = self.choose_action(state)
        else: action_index = np.random.randint(self.env.num_j_temp)
        self.action_memory.append(tf.constant(action_index))
        
        # print("Prob array ",self.P(np.reshape(scaled_state, (1, -1)))[0],"Action ",action_index)
        return action_index
    # def choose_action_training(self,state):
    #     scaled_state = state.copy()
    #     scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
    #     scaled_state[2,0] = (scaled_state[2,0])*80
    #     # print("Choose_action : " ,np.array([state[0,0],state[1,0],state[2,0]]))
    #     # print("Choose_action : " ,np.array([scaled_state[0,0],scaled_state[1,0],scaled_state[2,0]]))
    #     prob = self.P(np.array([[scaled_state[0,0], scaled_state[1, 0], scaled_state[2, 0]]]))
        
    #     dist = tfp.distributions.Categorical(probs=prob,dtype=tf.float32)
    #     action = dist.sample()
    #     #print("Prob : ", prob)
    #     # print("Action :", action)
    #     # print("\n")
    #     self.action_memory.append(action)
    #     #print(action)
    #     return int(action.numpy())
    
    # def choose_action(self, state):
    #     scaled_state = state.copy()
    #     scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
    #     scaled_state[2,0] = (scaled_state[2,0])*80
    #     #print("Choose_action : " ,np.array([scaled_state[0,0],scaled_state[1,0],scaled_state[2,0]]))
    #     action_index = tf.argmax(self.P(tf.reshape(scaled_state, (1, -1)))[0])
    #     return action_index
    
    
    def calc_loss(self,prob,action,returns,i):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob((action))
        #print("logprob : ",log_prob)
        loss = -1*(self.discount_factor**i)*log_prob*returns
        return loss
    
    def update_p_weights(self):
        sum_reward = 0
        discnt_rewards = []
        self.reward_memory.reverse()
        for r in self.reward_memory:
            sum_reward = r + self.discount_factor*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()
        discnt_rewards = (discnt_rewards-np.mean(discnt_rewards))/(np.std(discnt_rewards)+1e-10)
        i = 0    
        for state,action,returns in zip(self.state_memory,self.action_memory,discnt_rewards):
            with tf.GradientTape() as tape:
                scaled_state = state.copy()
                scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
                scaled_state[2,0] = (scaled_state[2,0])*80
                # print("Update_weights : ",np.array([scaled_state[0,0],scaled_state[1,0],scaled_state[2,0]]))
                prob = self.P(np.reshape(scaled_state, (1, -1)))
                loss = self.calc_loss(prob,action,returns,i)
                grads = tape.gradient(loss,self.P.trainable_variables)
                self.opt.apply_gradients(zip(grads,self.P.trainable_variables))
            i = i + 1
        #If done with updating empty all memories of this episode
        self.reward_memory = []
        self.action_memory = []
        self.state_memory = []


    def train(self, num_episodes):
        iter_num = 0
        episode_versus_reward = np.zeros((num_episodes, 2))
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        line, = ax.plot([], [])  # Empty plot to be updated dynamically

        for episode_index in range(num_episodes):
            # initialize markov chain with initial state
            state = self.env.reset()
            cumulative_reward = 0
            iter_num = 0
            while not self.env.done:
                action_index = self.choose_action_training(state)
                action = self.env.tj_list[action_index]
                # executing action, observing reward and next state to store experience in tuple
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += (self.discount_factor ** iter_num) * reward
                self.state_memory.append(state)
                self.reward_memory.append(reward)
                iter_num += 1
                state = next_state

            self.update_p_weights()
            if episode_index % 10000 == 0:
                self.P.save(f"P_{episode_index}.h5")
            if episode_index % 1 == 0:
                print(f"[episodes]: {episode_index}")
                # Update the plot dynamically
                episode_versus_reward[episode_index] = np.array([episode_index, cumulative_reward])
                line.set_data(episode_versus_reward[:episode_index+1, 0], episode_versus_reward[:episode_index+1, 1])
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)  # Adjust the pause time as needed
            
        plt.ioff()
        plt.show()
        return episode_versus_reward

class PolicyGradientAgent:
    def __init__(self, environment, learning_rate=1e-3, decay_rate=1e-4, discount_factor=1, entropy_coeff=1e-1, nn_arch=[400, 300, 200]):
        self.env = environment
        self.policy = self._get_policy(nn_arch)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.01,
                        decay_steps=10000,
                        decay_rate=0.9)
        #optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
        self.discount_factor = discount_factor
        self.entropy_coeff = tf.cast(entropy_coeff, dtype=tf.dtypes.float32)
        self.t_, self.T_, self.Ca_, self.Tj_, self.R_ = range(5)
    
    def _get_policy(self, nn_arch):
        policy = tf.keras.models.Sequential()
        policy.add(tf.keras.layers.Dense(nn_arch[0], input_shape=(3,), activation="relu"))
        for num_neurons in nn_arch[1:]:
            policy.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
        policy.add(tf.keras.layers.Dense(self.env.num_j_temp, activation="softmax"))
        policy.build()
        return policy
    
    def get_action(self, state):
        return self.env.tj_list[np.argmax(self.policy(state.reshape(1, -1))[0])]
    
    @tf.function
    def get_sampled_action_index(self, state):
        dist = tfp.distributions.Categorical(probs=self.policy(tf.reshape(state, (1, -1)))[0])
        return dist.sample()
    
    def gen_trajectory(self):
        self.env.reset()
        episode = np.zeros((self.env.n_tf, 5))
        i = 0
        while not self.env.done:
            state = self.env.curr_state.copy()
            episode[i, self.t_] = state[self.t_, 0]
            episode[i, self.T_] = state[self.T_, 0]
            episode[i, self.Ca_] = state[self.Ca_, 0]
            episode[i, self.Tj_] = self.get_sampled_action_index(state)
            # get new state from reactor
            next_state, reward, done, info = self.env.step(self.env.tj_list[int(episode[i, self.Tj_])])
            # calculate reward for action
            # add state action pair to state_action list
            episode[i, self.R_] = reward
            # print(episode[i])
            # get state from discrete T and Ca values
            i += 1
        return episode
    
    def get_return(self, episode, timestep):
        # reward_list
        reward_list = tf.cast(episode[timestep:, self.R_], dtype=tf.dtypes.float32)
        # discount_array_list
        discount_array = tf.constant([self.discount_factor**i for i in range(0, len(reward_list))], dtype=tf.dtypes.float32)
        # return calculation
        return_val = tf.tensordot(reward_list, discount_array, axes=1)
        return return_val
    
    def get_returns(self, episode):
        # reward_list
        reward_list = episode[:, 4]
        return_list = np.zeros_like(reward_list)
        return_list[-1] = reward_list[-1]
        for i in range(self.env.n_tf-2, -1, -1):
            return_list[i] = reward_list[i] + self.discount_factor * return_list[i+1]
        return return_list
    
    @tf.function
    def update_policy(self, episode, returns):
        for i in range(0, self.env.n_tf):
            Gt = tf.cast(returns[i], dtype=tf.dtypes.float32)
            # obtain discrete index of action
            action_taken = tf.cast(episode[i, self.Tj_], dtype=tf.dtypes.int32)
            # print(f"{action_taken = }, {masks = }")
            with tf.GradientTape() as tape:
                action_probabilities = self.policy(tf.reshape(episode[i, self.t_:self.Tj_], (1, -1)))[0]
                dist = tfp.distributions.Categorical(probs=action_probabilities)
                prob = dist.prob(action_taken)
                loss = -1 * ((self.discount_factor ** i) * tf.math.log(prob + 1e-30) * Gt + self.entropy_coeff * dist.entropy())
            gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
    
    def train(self, num_episodes):
        episode_versus_reward = np.zeros((num_episodes, 2))
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        line, = ax.plot([], [])  # Empty plot to be updated dynamically
        self.env.reset()
        for episode_index in range(num_episodes):
            episode = self.gen_trajectory()
            returns = self.get_returns(episode)
            self.update_policy(episode, returns)
            # for i in range(0, self.env.n_tf):
            #     Gt = self.get_return(episode, i)
            #     # obtain discrete index of action
            #     action_taken = np.rint(episode[i, self.Tj_]).astype(int)
            #     # print(f"{action_taken = }, {masks = }")
            #     with tf.GradientTape() as tape:
            #         action_probabilities = self.policy(episode[i, self.t_:self.Tj_].reshape(1, -1))[0]
            #         dist = tfp.distributions.Categorical(probs=action_probabilities)
            #         prob = dist.prob(action_taken)
            #         loss = -1 * ((self.discount_factor ** i) * tf.math.log(prob + 1e-30) * Gt)
            #     gradients = tape.gradient(loss, self.policy.trainable_variables)
            #     self.optimizer.apply_gradients(zip(gradients, self.policy.trainable_variables))
            if episode_index % 1 == 0: 
                print(f"[episodes]: {episode_index}")
                episode_versus_reward[episode_index] = np.array([episode_index, returns[0]])
                # Update the plot dynamically
                line.set_data(episode_versus_reward[:episode_index+1, 0], episode_versus_reward[:episode_index+1, 1])
                ax.relim()
                ax.autoscale_view()
            plt.draw()
            plt.pause(0.001)  # Adjust the pause time as needed
        return episode_versus_reward