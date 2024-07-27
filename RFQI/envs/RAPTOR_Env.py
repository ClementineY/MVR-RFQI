import gymnasium as gym
from gymnasium import spaces
import matlab.engine
import numpy as np
from pathlib import Path
import argparse
from typing import Union


def soft_hat(
    x: Union[float, np.ndarray],
    x_lower: float = 0,
    y_lower: float = 1e-3,
    x_plateau_start: float = 0,
    x_plateau_end: float = 0,
    x_upper: float = 0,
    y_upper: float = 1e-3,
) -> np.ndarray:
    k_lower = -np.log(y_lower) / np.power(x_lower - x_plateau_start, 2)
    k_upper = -np.log(y_upper) / np.power(x_upper - x_plateau_end, 2)
    return np.piecewise(
        x,
        [
            x < x_plateau_start,
            (x >= x_plateau_start) & (x <= x_plateau_end),
            x > x_plateau_end,
        ],
        [
            lambda x: np.exp(-k_lower * np.power(x - x_plateau_start, 2)),
            1,
            lambda x: np.exp(-k_upper * np.power(x - x_plateau_end, 2)),
        ],
    )

def compute_objectives(output_dict: dict) -> np.ndarray:
    rho = np.array(output_dict['rho'])
    q = np.array(output_dict['q'])
    q0 = np.array(output_dict['q0']).item()
    qmin = np.array(output_dict['qmin']).item()
    ROQM = np.argmin(q)
    shear = np.array(output_dict['shear']) # % rho/q dq/drho with rho=rhotorN
    Ip = np.array(output_dict['Ip'][-1]).item()
    Pec = np.array(output_dict['Pec']).item()
    
    # Objective functions
    obj_1 = soft_hat(np.abs(q0 - qmin), x_plateau_start=0, x_plateau_end=0, x_upper=2, y_upper=0.5)
    obj_2 = soft_hat(ROQM / len(rho), x_plateau_start=0, x_plateau_end=0, x_upper=1, y_upper=1e-3)
    obj_3 = soft_hat(np.mean(shear), x_plateau_start=1, x_plateau_end=1, x_upper=2, y_upper=1e-3)
    obj_4 = soft_hat(qmin, x_lower=2.2, y_lower=0.5, x_plateau_start=2.2, x_plateau_end=2.5, x_upper=3, y_upper=0.5)
    obj_5 = soft_hat(np.max(rho[q >= 3] if np.any(q >= 3) else 0), x_lower=0.5, y_lower=0.5, x_plateau_start=0.8, x_plateau_end=1, x_upper=2, y_upper=2)
    obj_6 = soft_hat(np.max(rho[q >= 4] if np.any(q >= 4) else 0), x_lower=0.5, y_lower=0.5, x_plateau_start=0.8, x_plateau_end=1, x_upper=2, y_upper=2)
    obj_7 = soft_hat(np.abs(Ip - 20e6), x_plateau_start=0, x_plateau_end=0, x_upper=20e6, y_upper=1e-3) # Minimisation, 1 if distance is 0, decaying to 1e-3 if the distance is 20e6.
    obj_8 = - soft_hat(np.abs(Pec - 15.4e6), x_lower=0, y_lower=1e-3, x_plateau_start=0, x_plateau_end=5e6) # Penalise excessive usage of pec, for pec lager than 15.4e6 less than 19.4e6, we set the negative reward to 1, decaying to 1e-3 if the distance is 0.
    
    return np.array([obj_1, obj_2, obj_3, obj_4, obj_5, obj_6, obj_7, obj_8])


class RaptorEnv(gym.Env):
    def __init__(self, raptor_directory: Path):
        self.raptor_directory = raptor_directory
        self.eng = matlab.engine.start_matlab()
        self.eng.eval(f"addpath(genpath('{str(self.raptor_directory)}'))", nargout=0)

        # Action space: 9 control signals, but last entry (Greenwald fraction) is fixed
        self.action_space = spaces.Box(low=0, high=1e+08, shape=(8,), dtype=np.float32)

        # Observation space: Ip, sum of U entries, total time, and 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(88,1), dtype=np.float32)

        self.current_step = 0
        self.total_time = 0

        self.eng.configure_scenario(nargout=0)
        self.x0 = np.array(self.eng.workspace['x0'])
        self.U = np.array(self.eng.workspace['U'])
        self.G = np.array(self.eng.workspace['G'])
        # import pdb;pdb.set_trace()
        self.V = np.array(self.eng.workspace['V'])
        self.model = self.eng.workspace['model']
        self.params = self.eng.workspace['params']
        self.reward = 0

    # def configure(self):

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 1
        self.total_time = 1

        self.sim_state_dict = self.eng.raptor_initial_step(
            self.x0, 
            np.array(self.U[:, 0]).reshape(-1, 1), 
            self.G, 
            self.V, 
            self.model, 
            self.params, 
            nargout=1
        )

        self.xk = np.array(self.sim_state_dict['x']).reshape(-1, 1)
        self.gk = np.array(self.G)[:, 0].reshape(-1, 1)
        self.vk = np.array(self.V)[:, 0].reshape(-1, 1)
        self.xdotk = np.array(self.sim_state_dict['xdot']).reshape(-1, 1)
        self.uk = np.array(self.U)[:, 0].reshape(-1, 1)
        # self.it = 1
        self.stap = self.sim_state_dict['stap']
        self.geop = self.sim_state_dict['geop']
        self.trap = self.sim_state_dict['trap']

        self.output_dict = self.get_output()

        return self._get_observation() #? should we also return sim_state_dict and output_dict
    
    def get_output(self):
            
        self.output_dict = self.eng.RAPTOR_out(
            self.xk,
            self.gk, 
            self.vk, 
            self.xdotk, 
            self.uk, 
            self.current_step, 
            self.stap, 
            self.geop, 
            self.trap, 
            self.model, 
            self.params
        )
        
        return self.output_dict 


    def step(self, action):
        self.current_step += 1
        # action = np.append(action, 1.0).reshape(-1, 1)  # Append fixed Greenwald fraction
        self.U[:, self.current_step] = action.flatten()  # Assign values to the selected column
        # print(self.U)
        # print(np.array(self.U[:, self.current_step]).reshape(-1, 1))
        print(self.G)

        self.sim_state_dict = self.eng.raptor_subsequent_step(
            self.current_step,
            np.array(self.U[:, self.current_step]).reshape(-1, 1),
            matlab.double(self.sim_state_dict['x']),
            matlab.double(self.sim_state_dict['xdot']),
            self.sim_state_dict['stap'],
            self.sim_state_dict['geop'],
            self.sim_state_dict['trap'],
            self.G,
            self.V,
            self.model,
            self.params,
            nargout=1
        ) # Key interactive step with RAPTOR
        
        self.xk = self.sim_state_dict['x']
        self.gk = np.array(self.G[:, self.current_step]).reshape(-1, 1)
        self.vk = np.array(self.V[:, self.current_step]).reshape(-1, 1)
        self.xdotk = self.sim_state_dict['xdot']
        self.uk = np.array(self.U[:, self.current_step]).reshape(-1, 1)
        # self.it = 1
        self.stap = self.sim_state_dict['stap']
        self.geop = self.sim_state_dict['geop']
        self.trap = self.sim_state_dict['trap']

        self.output_dict = self.get_output()

        observation = self._get_observation()
        self.reward += self._calculate_reward()
        terminate = self._terminate()
        
        if terminate:
            self.reward += self._calculate_final_reward()
        
        # Store the results
        # state_history = [self.sim_state_dict]
        # output_history = [self.output_dict]
        # state_history.append(self.sim_state_dict)
        # output_history.append(self.output_dict)

        return observation, self.reward, terminate, {}

    def _get_observation(self):
        x = np.array(self.sim_state_dict['x'])
        xdot = np.array(self.sim_state_dict['xdot'])
        return np.array([x, xdot])

    def _calculate_reward(self):
        reward = -1  # Constant negative (penalises for each timestep)
        # reward -= coefficient*Pec   # Negative function of the total EC power at this timestep (penalises actuator use)
        # reward -= coefficient*  # Distance from target Ip

        objectives = compute_objectives(self.output_dict)
        reward += np.sum(objectives)

        return reward

    def _calculate_final_reward(self):
        Ip = np.array(self.output_dict['Ip'][-1]).item()
        if Ip >= 20e6:
            reward = 10000  # Large bonus for reaching target Ip
        else:
            reward = -10000
        
        return reward

    def _terminate(self):
        Ip = np.array(self.output_dict['Ip'][-1]).item()
        return Ip >= 20e6 or self.current_step >= int(self.eng.workspace['t_resolution'] - 1)

    def close(self):
        self.eng.quit() 


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raptor_directory", type=Path, required=True)
    args = parser.parse_args()

    env = RaptorEnv(raptor_directory=args.raptor_directory)
    # env.configure()
    # Reset the environment to get the initial observation
    obs = env.reset()

    total_reward = 0
    for i in range(1, int(env.eng.workspace['t_resolution'] - 1)):
        # action, _states = model.predict(obs)
        action =  np.array(env.U[:, i]).reshape(-1, 1)
        # print(action)
        obs, rewards, terminate, info = env.step(action)

        # Print step details
        print(f"Action: {action}")
        print(f"Observation: {obs}")
        print(f"Rewards: {rewards}")
        print(f"Terminate: {terminate}")

        total_reward += rewards
        if terminate:
            break

    print(f"Total Reward: {total_reward}")

    env.close()
