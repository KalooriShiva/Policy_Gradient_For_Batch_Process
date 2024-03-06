import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow_probability as tfp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt



class PGAgent:
    
    def __init__(self, environment, learning_rate, decay_rate, discount_factor=0.95, epsilon=0.05,  nn_arch=[400, 300, 200]):
        self.env = environment
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.P = self._get_model(nn_arch)
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
    #     # print("Prob : ", prob)
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
    #     return self.env.tj_list[action_index]
    
    
    def calc_loss(self,prob,action,returns):
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob((action))
        #print("logprob : ",log_prob)
        loss = -log_prob*returns
        return loss
    
    def update_p_weights(self):
        sum_reward = 0
        discnt_rewards = []
        self.reward_memory.reverse()
        for r in self.reward_memory:
            sum_reward = r + self.discount_factor*sum_reward
            discnt_rewards.append(sum_reward)
        discnt_rewards.reverse()
            
        for state,action,returns in zip(self.state_memory,self.action_memory,discnt_rewards):
            with tf.GradientTape() as tape:
                scaled_state = state.copy()
                scaled_state[1,0] = (scaled_state[1,0] - self.env.min_j_temp)*(80/(self.env.max_j_temp-self.env.min_j_temp))
                scaled_state[2,0] = (scaled_state[2,0])*80
                # print("Update_weights : ",np.array([scaled_state[0,0],scaled_state[1,0],scaled_state[2,0]]))
                prob = self.P(np.reshape(scaled_state, (1, -1)))
                loss = self.calc_loss(prob,action,returns)
                grads = tape.gradient(loss,self.P.trainable_variables)
                self.opt.apply_gradients(zip(grads,self.P.trainable_variables))
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

