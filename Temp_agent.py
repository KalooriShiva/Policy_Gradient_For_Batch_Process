import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense
import tensorflow_probability as tfp

class PGAgentRNN:
    
    def __init__(self, environment, learning_rate=1e-5, discount_factor=1, epsilon=0.05, rnn_units=64):
        super().__init__(environment, learning_rate, discount_factor, epsilon)
        self.rnn_units = rnn_units
        self.P = self._get_model()
        
    def _get_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.rnn_units, return_sequences=True, input_shape=(None, 3)),
            tf.keras.layers.Dense(self.env.num_j_temp, activation='softmax')
        ])
        return model

    def choose_action_training(self, state):
        scaled_state = state.copy()
        scaled_state[1, 0] = (scaled_state[1, 0] - self.env.min_j_temp) * (80 / (self.env.max_j_temp - self.env.min_j_temp))
        scaled_state[2, 0] = (scaled_state[2, 0]) * 80
        prob = self.P.predict(np.expand_dims(np.transpose(scaled_state), axis=0))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        self.action_memory.append(action)
        return int(action.numpy())

    def update_p_weights(self):
        sum_reward = 0
        discounted_rewards = []
        self.reward_memory.reverse()
        for r in self.reward_memory:
            sum_reward = r + self.discount_factor * sum_reward
            discounted_rewards.append(sum_reward)
        discounted_rewards.reverse()

        for state, action, returns in zip(self.state_memory, self.action_memory, discounted_rewards):
            with tf.GradientTape() as tape:
                scaled_state = state.copy()
                scaled_state[1, 0] = (scaled_state[1, 0] - self.env.min_j_temp) * (80 / (self.env.max_j_temp - self.env.min_j_temp))
                scaled_state[2, 0] = (scaled_state[2, 0]) * 80
                prob = self.P(np.expand_dims(np.transpose(scaled_state), axis=0))
                loss = self.calc_loss(prob, action, returns)
                grads = tape.gradient(loss, self.P.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.P.trainable_variables))
        
        # Empty all memories of this episode
        self.reward_memory = []
        self.action_memory = []
        self.state_memory = []

    def train(self, num_episodes):
        iter_num = 0
        episode_versus_reward = np.zeros((num_episodes, 2))
        for episode_index in range(num_episodes):
            # Initialize markov chain with initial state
            state = self.env.reset()
            cumulative_reward = 0
            while not self.env.done:
                action_index = self.choose_action_training(state)
                action = self.env.tj_list[action_index]
                
                # Executing action, observing reward and next state to store experience in tuple
                next_state, reward, done, info = self.env.step(action)
                cumulative_reward += (self.discount_factor ** iter_num) * reward
                
                # Store experience in replay memory
                self.state_memory.append(state)
                self.reward_memory.append(reward)
                iter_num += 1
                state = next_state
            
            # Update policy weights
            self.update_p_weights()
            if episode_index % 10000 == 0:
                self.P.save(f"P_{episode_index}.h5")
            if episode_index % 1 == 0:
                print(f"[episodes]: {episode_index}")
            episode_versus_reward[episode_index] = np.array([episode_index, cumulative_reward])
        
        return episode_versus_reward

