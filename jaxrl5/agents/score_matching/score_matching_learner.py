"""Implementations of algorithms for continuous control."""
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flax import struct
import numpy as np

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, subsample_ensemble, DDPM, FourierFeatures, cosine_beta_schedule, ddpm_sampler, vp_beta_schedule

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def tensorstats(tensor, prefix=None):
  assert tensor.size > 0, tensor.shape
  metrics = {
      'mean': tensor.mean(),
      'std': tensor.std(),
      'mag': jnp.abs(tensor).max(),
      'min': tensor.min(),
      'max': tensor.max(),
  }
  if prefix:
    metrics = {f'{prefix}_{k}': v for k, v in metrics.items()}
  return metrics


class ScoreMatchingLearner(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    act_dim: int = struct.field(pytree_node=False)
    min_T: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    num_qs: int = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    policy_temperature: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.7,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_tau: float = 0.001,
        actor_layer_norm: bool = False,
        policy_temperature: float = 3.0,
        min_T: int = 3, 
        T: int = 5,
        time_dim: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[-1]

        # Time embedding network.
        preprocess_time_cls = partial(
            FourierFeatures, output_size=time_dim, learnable=True)

        cond_model_cls = partial(
            MLP, hidden_dims=(128, 128), activations=mish,
            activate_final=False)
        
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                activations=mish, use_layer_norm=actor_layer_norm,
                activate_final=False)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)
        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']

        score_model = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params,
            tx=optax.adam(learning_rate=actor_lr))
        target_score_model = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None))

        critic_base_cls = partial(
            MLP,
            hidden_dims=critic_hidden_dims,
            activate_final=True,
        )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.adam(learning_rate=critic_lr),
        )
        target_critic_def = Ensemble(critic_cls, num=num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')

        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None,
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            min_T=min_T,
            T=T,
            M=M,
            alphas=alphas,
            num_qs=num_qs,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            critic_hyperparam=critic_hyperparam,
            clip_sampler=clip_sampler,
            policy_temperature=policy_temperature,
        )

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:

        # Sample actions for next state.
        key, rng = jax.random.split(agent.rng)
        next_actions, rng = ddpm_sampler(
            agent.score_model.apply_fn,
            agent.target_score_model.params,
            agent.T, rng, agent.act_dim,
            batch['next_observations'],
            agent.alphas, agent.alpha_hats,
            agent.betas, agent.ddpm_temperature,
            agent.M, agent.clip_sampler)
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, agent.target_critic.params, agent.num_qs, agent.num_qs)
        key, rng = jax.random.split(rng)
        next_qs = agent.target_critic.apply_fn(
            {"params": target_params}, batch["next_observations"],
            next_actions, False, rngs={"dropout": key})
        key, rng = jax.random.split(rng)
        next_V = next_qs.min(axis=0)
        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_V
        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params}, batch["observations"], batch["actions"]
            )
            critic_loss = ((qs - sg(target_q)) ** 2)
            metrics = tensorstats(critic_loss, 'critic_loss')
            metrics.update(tensorstats(qs, 'qs'))
            metrics.update(tensorstats(target_q, 'target_qs'))
            return critic_loss.mean(), metrics

        grads, metrics = jax.grad(critic_loss_fn, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau)
        target_critic = agent.target_critic.replace(params=target_critic_params)
        new_agent = agent.replace(critic=critic, target_critic=target_critic, rng=rng)
        return new_agent, metrics

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        B, A = batch['actions'].shape

        # Forward process with RB actions.
        key, rng = jax.random.split(agent.rng, 2)
        time = jax.random.randint(key, (B,), 0, agent.T)
        key, rng = jax.random.split(rng, 2)
        noise_sample = jax.random.normal(
            key, (B, agent.act_dim))
        key, rng = jax.random.split(rng, 2)
        alpha_hats = agent.alpha_hats[time]
        time = jnp.expand_dims(time, axis=1)
        alpha_1 = jnp.expand_dims(jnp.sqrt(alpha_hats), axis=1)
        alpha_2 = jnp.expand_dims(jnp.sqrt(1 - alpha_hats), axis=1)
        noisy_actions = alpha_1 * batch['actions'] + alpha_2 * noise_sample
        key, rng = jax.random.split(rng, 2)

        # Compute dQ/da. # TODO(alescontrela): Is this correct?
        critic_jacobian = jax.grad(lambda actions: agent.critic.apply_fn(
            {"params": agent.critic.params}, batch['observations'], actions).sum(),
            has_aux=False)(noisy_actions)
        assert critic_jacobian.shape == (B, A), (critic_jacobian.shape, (B, A))

        def actor_loss_fn(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            eps_pred = agent.score_model.apply_fn(
                {'params': score_model_params}, batch['observations'],
                noisy_actions, time, rngs={'dropout': key}, training=True)
            assert eps_pred.shape == (B, A)
            actor_loss = -1 * jnp.multiply(eps_pred, sg(critic_jacobian)).sum(-1)
            assert actor_loss.shape == (B,)
            metrics = tensorstats(actor_loss, 'actor_loss')
            metrics.update(tensorstats(eps_pred, 'eps_pred'))
            metrics.update(tensorstats(critic_jacobian, 'critic_jacobian'))
            return actor_loss.mean(0), metrics

        key, rng = jax.random.split(rng, 2)
        grads, metrics = jax.grad(actor_loss_fn, has_aux=True)(
            agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params,
            agent.actor_tau)
        target_score_model = agent.target_score_model.replace(
            params=target_score_params)
        new_agent = agent.replace(
            score_model=score_model,
            target_score_model=target_score_model, rng=rng)
        return new_agent, metrics

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        actions, new_agent = self.eval_actions(observations)
        key, rng = jax.random.split(new_agent.rng, 2)
        noise = jax.random.normal(key, shape=actions.shape) * 0.1
        actions = actions + noise
        actions = jnp.clip(actions, -1.0, 1.0)
        key, rng = jax.random.split(rng, 2)
        return actions, new_agent.replace(rng=rng)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            self.target_score_model.params,
            self.T, rng, self.act_dim, observations,
            self.alphas, self.alpha_hats,
            self.betas, self.ddpm_temperature,
            self.M, self.clip_sampler)
        assert actions.shape == (1, self.act_dim)
        _, rng = jax.random.split(rng, 2)
        return jnp.squeeze(actions), self.replace(rng=rng)

    
    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, actor_info = new_agent.update_actor(batch)
        new_agent, critic_info = new_agent.update_q(batch)
        return new_agent, {**actor_info, **critic_info}
