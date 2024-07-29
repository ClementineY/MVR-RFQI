import gymnasium as gym
import numpy as np
from gymnasium import spaces
import RFQI.envs
from stable_baselines3 import PPO
import os
import argparse
import libtmux

def get_action_type(action_space):
    if isinstance(action_space, spaces.Box):
        shape = action_space.shape
        assert len(shape) == 1
        if shape[0] == 1:
            return 'continuous'
        else:
            return 'multi_continuous'
    elif isinstance(action_space, spaces.Discrete):
        return 'discrete'
    elif isinstance(action_space, spaces.MultiDiscrete):
        return 'multi_discrete'
    elif isinstance(action_space, spaces.MultiBinary):
        return 'multi_binary'
    else:
        raise NotImplementedError

def train_ppo_model(env_name, total_timesteps=25000):
    env = gym.make(env_name)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model_path = f"./models/ppo_{env_name}"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # loaded_model = PPO.load(model_path)
    # print("Model loaded successfully")

def start_tmux_session(session_name, env_name, total_timesteps):
    server = libtmux.Server()
    
    # Create a new session or attach to an existing one
    try:
        session = server.new_session(session_name=session_name)
    except libtmux.exc.TmuxSessionExists:
        session = server.find_where({"session_name": session_name})

    # Create a new window
    window = session.new_window(attach=False, window_name="ppo_training")

    # Send the training command
    cmd = f"python -c \"from __main__ import train_ppo_model; train_ppo_model('{env_name}', {total_timesteps})\""
    window.panes[0].send_keys(cmd)

    print(f"Training started in tmux session '{session_name}', window 'ppo_training'")
    print(f"To attach to the session, run: tmux attach-session -t {session_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='raptor-v0', type=str)
    parser.add_argument('--total_timesteps', default=25000, type=int)
    parser.add_argument('--tmux_session', default='ppo_training', type=str)
    args = parser.parse_args()

    # Create models directory if it doesn't exist
    if not os.path.exists("./models"):
        os.makedirs("./models")

    # Start the training in a tmux session
    start_tmux_session(args.tmux_session, args.env, args.total_timesteps)
