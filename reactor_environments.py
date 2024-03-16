

import numpy as np

class Environment:
    
    def __init__(self, timesteps=10, batch_time=80, min_temp=293, 
                 max_temp=308, min_conc=0, max_conc=1, num_temp=10, 
                 num_conc=10, min_j_temp=273, max_j_temp=318, num_j_temp=10, k0=2.53, testing=False):
        self.curr_state = np.vstack(
            [0, np.random.uniform(min_temp, max_temp), np.random.uniform(min_conc, max_conc)]
        )
        self.dt = batch_time / timesteps
        self.tf = batch_time
        self.n_tf = timesteps
        self.min_temp = min_temp
        self.min_conc = min_conc
        self.max_temp = max_temp
        self.max_conc = max_conc
        self.num_temp = num_temp
        self.num_conc = num_conc
        self.min_j_temp = min_j_temp
        self.max_j_temp = max_j_temp
        self.num_j_temp = num_j_temp
        self.time_list = np.linspace(0, batch_time, timesteps)
        self.Tref = np.zeros_like(self.time_list)
        self.temp_list = np.linspace(min_temp, max_temp, num_temp)
        self.conc_list = np.linspace(min_conc, max_conc, num_conc)
        self.tj_list = np.linspace(min_j_temp, max_j_temp, num_j_temp)
        self.prev_action = None
        self.done = False
        self.info = None
        self.T = 1
        self.Ca = 2
        self._init_ref_temp()
        self.k0 = k0
        self.testing = testing
    
    def _init_ref_temp(self):
        for index, value in enumerate(self.Tref):
            if index <= 10 / self.dt: self.Tref[index] = 298 + 0.5*index*self.dt
            elif index <= 40 / self.dt: self.Tref[index] = 303
            elif index <= 50 / self.dt: self.Tref[index] = 303 + 0.3*(index*self.dt - 40)
            elif index <= 60 / self.dt: self.Tref[index] = 306
            elif index <= 75 / self.dt: self.Tref[index] = 306 - (8/15)*(index*self.dt - 60)
            else: self.Tref[index] = 298
    
    def step(self, action):
        """ Bugs may be there. check it """
        C1 = -0.09 # litre/min
        C2 = 4.1492 * (10 ** 19)
        Ea_R = 13550.0
        k0 = self.k0 * (10**19) * self.testing + (1-self.testing)*np.random.normal(loc=self.k0, scale=1) * (10**19)
        dt = self.dt
        T = 1
        Ca = 2
        Q = np.array([[1, 0],[0, 1]])
        R = 0.01
        Tj = action
        new_T = self.curr_state[T, 0] + \
                dt * (C1 * (self.curr_state[T, 0] - Tj) + \
                C2 * np.exp(-Ea_R/self.curr_state[T, 0]) * \
                (self.curr_state[Ca, 0]**2))
        # Reference expression: Ca(k+1) = Ca(k) - dt*k0*exp(-Ea/RT(k))*Ca(k)^2
        new_Ca = self.curr_state[Ca, 0] - \
                 dt * k0*(self.curr_state[Ca, 0]**2)*np.exp(-1*Ea_R/self.curr_state[T, 0])
        if new_Ca < 0: new_Ca = 0
        next_state = np.vstack([self.curr_state[0, 0] + 1, new_T, new_Ca])
        if self.prev_action is None:
            d_action = 0
        else:
            d_action = action - self.prev_action
        error_term = self.curr_state[1:] - np.vstack([self.Tref[int(self.curr_state[0, 0])], 0])
        reward = error_term.T @ (Q @ error_term) + R * (d_action**2)
        self.prev_action = action
        self.done = self.curr_state[0, 0] + 1 == self.n_tf
        self.curr_state = next_state
        return next_state, -1 * reward[0, 0], self.done, self.info

    def state_index(self, state):
        """ Calculate the index of a state in the state arrays
        Parameters:
            state: a 2 x 1 numpy array"""
        # Gap between two successive temperatures in the temperature list
        temp_delta = 0.5*(self.max_temp - self.min_temp) / (self.num_temp - 1)
        # Gap between two successive concentrations in the concentration list
        conc_delta = 0.5*(self.max_conc - self.min_conc) / (self.num_conc - 1)
        # Finding temperature index and concentration index with builtin numpy function
        if self.min_temp > state[self.T]: temp_index = 0
        elif self.max_temp < state[self.T]: temp_index = self.num_temp - 1
        else: temp_index = np.random.choice(np.where(abs(self.temp_list - state[self.T]) <= temp_delta)[0])
        if self.min_conc > state[self.Ca]: conc_index = 0
        elif self.max_conc < state[self.Ca]: conc_index = self.num_conc - 1
        else: conc_index = np.random.choice(np.where(abs(self.conc_list - state[self.Ca]) <= conc_delta)[0])
        return np.array([np.rint(state[0, 0]).astype(int), temp_index, conc_index])
    
    def reset(self):
        self.curr_state = np.vstack(
            [0, np.random.uniform(self.min_temp, self.max_temp), np.random.uniform(self.min_conc, self.max_conc)]
        )
        self.prev_action = None
        self.done = False
        self.info = None
        return self.curr_state

if __name__ == "__main__":
    import pandas as pd
    import seaborn as sns
    
    env = Environment(timesteps=100)
    state_arr = np.zeros_like(env.time_list)
    i = 0
    while not env.done:
        state_arr[i] = env.curr_state[env.T, 0]
        next_state, reward, done, info = env.step(350)
        print(f"{next_state=}, {reward=}, {done=}, {info=}")
        i += 1
    df = pd.DataFrame({"State": state_arr, "time": env.time_list})
    sns.set_theme()
    sns.lineplot(
        data=df,
        x="time",
        y="State",
    )