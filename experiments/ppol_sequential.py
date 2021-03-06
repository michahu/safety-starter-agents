from tensorflow.python.ops.gen_io_ops import save

from safe_rl import ppo_lagrangian
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork

from garage.envs import GymEnv

from safety_gym.envs.engine import Engine
from hazard_world_envs import HW3DS, HW3DWrapper
from gym import Env
from copy import deepcopy
import click


@click.command()
@click.option("--seed", default=42)
@click.option("--num_steps", default=1e7)
@click.option("--steps_per_epoch", default=30000)
@click.option("--save_freq", default=50)
@click.option("--target_kl", default=0.01)
@click.option("--cost_lim", default=25)
@click.option("--cpu", default=1)
@click.option("--render", default=False)
@click.option("--exp_name")
def ppo_lagrangian_hw3d(
    exp_name,
    cpu,
    seed,
    num_steps,
    steps_per_epoch,
    save_freq,
    target_kl,
    cost_lim,
    render,
):

    mpi_fork(cpu)

    epochs = int(num_steps / steps_per_epoch)
    logger_kwargs = setup_logger_kwargs(exp_name, seed)
    env = HW3DS()

    ppo_lagrangian(
        env_fn=lambda: HW3DWrapper(env),
        ac_kwargs=dict(hidden_sizes=(256, 256)),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        save_freq=save_freq,
        target_kl=target_kl,
        cost_lim=env.cost_lim,
        seed=seed,
        logger_kwargs=logger_kwargs,
        render=render,
    )


ppo_lagrangian_hw3d()