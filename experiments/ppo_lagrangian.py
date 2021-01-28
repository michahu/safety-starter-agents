from tensorflow.python.ops.gen_io_ops import save

from safe_rl import ppo_lagrangian
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork

from safety_gym.envs.engine import Engine
import gym, safety_gym
from gym import Env
from copy import deepcopy
import random
import click

class HazardWorld3D(Env):
	def __init__(self) -> None:
		super().__init__()

		config = {
		'robot_base': 'xmls/point.xml',
		'use_language': True,
		'task': 'goal',
		'observe_goal_lidar': True,
		'observe_box_lidar': True,
		'lidar_max_dist': 3,
		'lidar_num_bins': 16,
		'placements_extents': [-2,-2,2,2],
		'task': 'goal',
		'goal_size': 0.3,
		'goal_keepout': 0.305,
		# puddles
		'hazards_size': 0.2,
		'hazards_keepout': 0.18,
		'observe_hazards': True,
		'hazards_num': 4,
		# vases
		'observe_vases': True,
		'vases_num': 4,  
		# gremlins
		'gremlins_travel': 0.35,
		'gremlins_keepout': 0.4,
		'observe_gremlins': True,
		'gremlins_num': 4,
		# button
		# 'buttons_num': 4,
		# 'buttons_size': 0.1,
		# 'buttons_keepout': 0.2,
		# 'observe_buttons': True,
		# pillars
		'observe_pillars': True,
		'pillars_num': 4
		}

		pillar_env_config = deepcopy(config)
		pillar_env_config['constrain_pillars'] = True 
		pillar_env_config['constrained_object'] = 'pillars'

		hazard_env_config = deepcopy(config)
		hazard_env_config['constrain_hazards'] = True
		hazard_env_config['constrained_object'] = 'hazards' 

		vases_env_config = deepcopy(config)
		vases_env_config['constrain_vases'] = True 
		vases_env_config['constrained_object'] = 'vases'

		gremlins_env_config = deepcopy(config)
		gremlins_env_config['constrain_gremlins'] = True 
		gremlins_env_config['constrained_object'] = 'gremlins'

		buttons_env_config = deepcopy(config)
		buttons_env_config['constrain_buttons'] = True
		buttons_env_config['constrained_object'] = 'buttons'

		self.envs = [
			Engine(pillar_env_config),
			Engine(hazard_env_config),
			Engine(vases_env_config),
			Engine(gremlins_env_config),
			Engine(buttons_env_config)
		]
		self.curr_env = None
		self.observation_space = self.envs[0].observation_space
		self.action_space = self.envs[0].action_space

	def reset(self):
		self.curr_env = random.choice(self.envs)
		return self.curr_env.reset()
	
	def render(self, **kwargs):
		return self.curr_env.render(**kwargs)

	def __str__(self):
		return self.envs[0].__str__()

	def step(self, action):
		return self.curr_env.step(action)


@click.command()
@click.option('--seed', default=42)
@click.option('--num_steps', default=1e7)
@click.option('--steps_per_epoch', default=30000)
@click.option('--save_freq', default=50)
@click.option('--target_kl', default=0.01)
@click.option('--cost_lim', default=25)
@click.option('--cpu', default=1)
@click.option('--render', default=False)
@click.option('--exp_name')
def ppo_lagrangian_hw3d(exp_name, cpu, seed, num_steps, steps_per_epoch, save_freq, target_kl, cost_lim, render):

	mpi_fork(cpu)

	epochs = int(num_steps / steps_per_epoch)
	logger_kwargs = setup_logger_kwargs(exp_name, seed)

	ppo_lagrangian(
		env_fn = lambda : HazardWorld3D(),
		ac_kwargs = dict(hidden_sizes=(256,256)),
		epochs = epochs, 
		steps_per_epoch = steps_per_epoch,
		save_freq=save_freq,
		target_kl=target_kl,
		cost_lim=cost_lim,
		seed=seed,
		logger_kwargs=logger_kwargs,
		render=render
		)


ppo_lagrangian_hw3d()

'''
Design idea: How to extend Safety Gym to our setting?
1st attempt: Create 5 different environments, each with a constrained object.
For the env function, create a new environment. 
This environment amalgamates the 5 different environments. Env reset switches between them.
In multitask env, keep an instance variable that points to the current env.
'''