# Q-Score Matching RL (QSM)

## Overview

This repository provides the official implementation of the Q-Score Matching algorithm (QSM) for the paper ["Learning a Diffusion Model Policy from Rewards via Q-Score Matching"](https://arxiv.org/abs/2312.11752) by Michael Psenka, Alejandro Escontrela, Pieter Abbeel, and Yi Ma. The setting for QSM is off-policy reinforcement learning in continuous state/action spaces, where the agent's policy is represented as a diffusion model. **The core of the QSM algorithm**: by iteratively aligning the denoising model of the policy with the action gradient of the critic $\nabla_a Q$, we can learn an optimal diffusion model policy by *only training against the denoiser model*, akin to standard diffusion model training, rather than having to backpropagate through the entire diffusion model evaluation. A straightforward idea, but one with interesting math behind it and surprising practical benefits (e.g. *naturally learns explorative agents*). Please see [the original paper](https://www.michaelpsenka.io/qsm/) for the theory and practical benefits of QSM.

Diffusion models have gained popularity in generative tasks due to their ability to represent complex distributions over continuous spaces. In the context of reinforcement learning, they offer both expressiveness and ease of sampling, making them a promising choice for policy representation. While many works have been done in the offline setting, the online/off-policy setting for diffusion model policies is still relatively underexplored.

The code is built on top of a re-implementation of the [jaxrl](https://github.com/ikostrikov/jaxrl) framework.

## Installation

To get started, you need to install the required dependencies. Ensure you have Python 3.8+ and a suitable GPU setup.

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For other versions of CUDA, follow the [JAX installation instructions](https://github.com/google/jax#pip-installation-gpu-cuda).

## Getting Started

To reproduce the results or to run the provided training scripts, navigate to the respective example directories and execute the provided training script:

```bash
cd examples/states
python3 train_score_matching_online.py
```

Script options can be found within the [training script file](examples/states/train_score_matching_online.py). For example, if you want to train on a different environment:

```bash
python3 train_score_matching_online.py --env_name walker_walk
```

## Important Files and Scripts

- **[Main Training Script](examples/states/train_score_matching_online.py)**: The main training script to train a diffusion model agent using QSM. Includes options for the environment and training scenario.

- **[QSM Learner](jaxrl5/agents/score_matching/score_matching_learner.py)**: The core implementation of the QSM algorithm, including methods for creating the learner, updating critic and actor networks, and sampling actions. **Note** that if you want to make any changes to the learner after installation, you will need to reinstall jaxrl5 locally, by running the following from the root directory of the repository:
```bash
pip install ./
```

- **[Training Configuration for Score Matching](examples/states/configs/score_matching_config.py)**: Configuration file for setting hyperparameters and model configurations for the QSM learner.

- **[DDPM Implementation](jaxrl5/networks/diffusion.py)**: Contains the implementation of Denoising Diffusion Probabilistic Models (DDPM), Fourier Features, and various beta schedules essential for the score matching process.

## Tuning QSM

Hyperparameters can be found and modified in [examples/states/configs/score_matching_config.py](examples/states/configs/score_matching_config.py). An example config looks like the following:

```python
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "ScoreMatchingLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.discount = 0.99
    config.tau = 0.005
    config.T = 5
    config.M_q = 50
    config.critic_hidden_dims=(512,512)
    config.actor_hidden_dims=(512,512)
    return config
```

For QSM, `M_q` is probably the most important parameter to tune. This corresponds to how much we scale the Q-score when matching the DDPM noise model to it ($\alpha$ in Alg. 1 of the main paper https://arxiv.org/pdf/2312.11752):
    
$$min_\theta \lVert\epsilon_\theta(a, s, t) - M_q \nabla_a Q(s, a)\rVert_2^2$$

We us "M_q" in code to not confuse with $\alpha$'s from DDPM noise scheduling.

`M_q` corresponds to something akin to how aggresive the learning process is: lower M_q will take more conservative steps through the denoising process and remain higher entropy, and higher M_q will take more aggressive steps towards optimal actions.

This is also analogous to a sort of explore/expoit tradeoff, but note that QSM will typically learn explorative policies on convergence regardless, see for example Fig. 4 & 5 of the main paper.

For more difficult tasks like `quadruped_walk` and `humanoid_walk`, a more aggressive `M_q` (e.g. `M_q = 120`) may perform better.


## Usage Example

The [main training script](examples/states/train_score_matching_online.py) gives a minimal example for launching a QSM learning agent in an environment. Below is a slightly stripped down version to illustrate the usage of jaxrl5 and the QSM learner:

```python
import os
import jax
import gym
import tqdm
from absl import app, flags
from ml_collections import config_flags

from jaxrl5.agents import ScoreMatchingLearner
from jaxrl5.data import ReplayBuffer
from jaxrl5.evaluation import evaluate
from jaxrl5.wrappers import wrap_gym

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "CartPole-v1", "Environment name.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 1, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("start_training", int(1e4), "Number of steps to start training.")
config_flags.DEFINE_config_file("config", "examples/states/configs/score_matching_config.py", "Training configuration file.")

def main(_):
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env.seed(FLAGS.seed)
    
    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    config = FLAGS.config
    kwargs = dict(config)

    agent = ScoreMatchingLearner.create(FLAGS.seed, env.observation_space, env.action_space, **kwargs)
    replay_buffer = ReplayBuffer(env.observation_space, env.action_space, FLAGS.max_steps)
    replay_buffer.seed(FLAGS.seed)

    observation, done = env.reset(), False

    for step in tqdm.tqdm(range(1, FLAGS.max_steps + 1), smoothing=0.1):
        if step < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action, agent = agent.sample_actions(observation)
        
        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert({
            "observations": observation,
            "actions": action,
            "rewards": reward,
            "masks": mask,
            "dones": done,
            "next_observations": next_observation
        })
        observation = next_observation

        if done:
            observation, done = env.reset(), False

        if step >= FLAGS.start_training:
            batch = replay_buffer.sample(FLAGS.batch_size)
            agent, _ = agent.update(batch)

        if step % FLAGS.log_interval == 0:
            print(f"Step: {step}")

        if step % FLAGS.eval_interval == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=FLAGS.eval_episodes)
            print(f"Evaluation at step {step}: {eval_info}")

if __name__ == "__main__":
    app.run(main)
```

## Contributing

We welcome contributions to enhance the repository. If you encounter any issues or have suggestions, feel free to open an issue or a pull request.

## Citation

If you use this code or our QSM algorithm in your research, please cite our paper:

```
@inproceedings{psenka2024qsm,
  title={Learning a Diffusion Model Policy from Rewards via Q-Score Matching},
  author={Psenka, Michael and Escontrela, Alejandro and Abbeel, Pieter and Ma, Yi},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  year={2024}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
