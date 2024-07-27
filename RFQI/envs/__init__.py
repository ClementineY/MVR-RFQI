from envs.RAPTOR_Env import RaptorEnv
import gymnasium as gym
from gym.envs.registration import register


register(
    id='raptor-v0',
    entry_point=RaptorEnv,
)
