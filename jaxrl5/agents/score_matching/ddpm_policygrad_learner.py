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

from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, StateActionValue, DDPM, FourierFeatures
from jaxrl5.networks import cosine_beta_schedule, ddpm_sampler, ddpm_sampler_keepinner, vp_beta_schedule

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


class DiffusionPolicygradLearner(Agent):
    score_model: TrainState
    critic_1: TrainState
    critic_2: TrainState
    target_critic_1: TrainState
    target_critic_2: TrainState

    discount: float
    tau: float
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
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
        ddpm_temperature: float = 1.0,
        actor_layer_norm: bool = False,
        T: int = 5,
        time_dim: int = 64,
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
        actor_params = actor_def.init(
            actor_key, observations, actions, time)['params']

        score_model = TrainState.create(
            apply_fn=actor_def.apply, params=actor_params,
            tx=optax.adam(learning_rate=actor_lr))

        # Initialize critics.
        critic_base_cls = partial(
            MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_def = StateActionValue(critic_base_cls)
        critic_key_1, critic_key_2 = jax.random.split(critic_key, 2)
        critic_params_1 = critic_def.init(critic_key_1, observations, actions)["params"]
        critic_params_2 = critic_def.init(critic_key_2, observations, actions)["params"]
        critic_1 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_1,
            tx=optax.adam(learning_rate=critic_lr))
        critic_2 = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params_2,
            tx=optax.adam(learning_rate=critic_lr))

        target_critic_def = StateActionValue(critic_base_cls)
        target_critic_1 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_1,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),)
        target_critic_2 = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params_2,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),)

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
            critic_1=critic_1,
            critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            clip_sampler=clip_sampler,
        )

    def update_q(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:

        (B, _) = batch['observations'].shape
        A = agent.act_dim 

        # Sample actions for next state.
        key, rng = jax.random.split(agent.rng)
        next_actions, rng = ddpm_sampler(
            agent.score_model.apply_fn,
            agent.score_model.params,
            agent.T, rng, agent.act_dim,
            batch['next_observations'],
            agent.alphas, agent.alpha_hats,
            agent.betas, agent.ddpm_temperature,
            agent.clip_sampler)
        key, rng = jax.random.split(rng, 2)
        noise = jax.random.normal(key, shape=next_actions.shape) * 0.1
        next_actions = next_actions + noise
        next_actions = jnp.clip(next_actions, -1.0, 1.0)
        key, rng = jax.random.split(rng, 2)
        assert next_actions.shape == (B, A)

        # Compute target q.
        key, rng = jax.random.split(rng)
        next_q_1 = agent.target_critic_1.apply_fn(
            {"params": agent.target_critic_1.params}, batch["next_observations"],
            next_actions, True, rngs={"dropout": key})
        key, rng = jax.random.split(rng)
        next_q_2 = agent.target_critic_2.apply_fn(
            {"params": agent.target_critic_2.params}, batch["next_observations"],
            next_actions, True, rngs={"dropout": key})
        key, rng = jax.random.split(rng)
        next_v = jnp.stack([next_q_1, next_q_2], 0).min(0)
        target_q = batch["rewards"] + agent.discount * batch["masks"] * next_v
        metrics = tensorstats(target_q, 'target_q')
        assert target_q.shape == (B,)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            q = agent.critic_1.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch['actions'], training=True)
            loss = ((q - sg(target_q)) ** 2)
            assert loss.shape == (B,)
            metrics = {**tensorstats(loss, 'c_loss'), **tensorstats(q, 'q')}
            return loss.mean(), metrics

        grads_c_1, metrics_c_1 = jax.grad(critic_loss_fn, has_aux=True)(agent.critic_1.params)
        metrics.update({f'{k}_1': v for k, v in metrics_c_1.items()})
        critic_1 = agent.critic_1.apply_gradients(grads=grads_c_1)

        grads_c_2, metrics_c_2 = jax.grad(critic_loss_fn, has_aux=True)(agent.critic_2.params)
        metrics.update({f'{k}_2': v for k, v in metrics_c_2.items()})
        critic_2 = agent.critic_2.apply_gradients(grads=grads_c_2)

        target_critic_1_params = optax.incremental_update(
            critic_1.params, agent.target_critic_1.params, agent.tau)
        target_critic_2_params = optax.incremental_update(
            critic_2.params, agent.target_critic_2.params, agent.tau)
        target_critic_1 = agent.target_critic_1.replace(params=target_critic_1_params)
        target_critic_2 = agent.target_critic_2.replace(params=target_critic_2_params)
        new_agent = agent.replace(
            critic_1=critic_1, critic_2=critic_2,
            target_critic_1=target_critic_1,
            target_critic_2=target_critic_2,
            rng=rng)
        return new_agent, metrics

    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        B, _ = batch['actions'].shape
        A = agent.act_dim 

        # applying diffusion model policygrad formula
        def actor_loss_fn_policygrad(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # compute new action samples using params so we can autodiff
            # through
            # note we need to convert to batch operation
            
            # first time we dont do with respect to input, to ignore gradient
            key, rng_main = jax.random.split(agent.rng)
            actions_actorloss, _, rng = ddpm_sampler_keepinner(
                agent.score_model.apply_fn,
                agent.score_model.params,
                agent.T, rng_main, agent.act_dim,
                batch["observations"],
                agent.alphas, agent.alpha_hats,
                agent.betas, agent.ddpm_temperature,
                agent.clip_sampler)
            
            # evaluate target critic on batch
            key, rng = jax.random.split(rng)
            target_q_1 = agent.target_critic_1.apply_fn(
                {"params": agent.target_critic_1.params},
                batch["observations"], actions_actorloss, True, rngs={"dropout": key})
            key, rng = jax.random.split(rng)
            target_q_2 = agent.target_critic_2.apply_fn(
                {"params": agent.target_critic_2.params},
                batch["observations"], actions_actorloss, True, rngs={"dropout": key})
            target_q = jnp.stack([target_q_1, target_q_2], 0).min(0)

            # now compute with respect to input for logprobs
            _, logprobs, rng = ddpm_sampler_keepinner(
                agent.score_model.apply_fn,
                score_model_params,
                agent.T, rng_main, agent.act_dim,
                batch["observations"],
                agent.alphas, agent.alpha_hats,
                agent.betas, agent.ddpm_temperature,
                agent.clip_sampler)
            

            # extract logprobs
            # logprobs = actions_actorloss_logprob[:, -1]
            
            # compute diffusion-QL loss
            actor_loss = -target_q*logprobs
            
            assert actor_loss.shape == (B,)
            metrics = tensorstats(actor_loss, 'actor_loss')
            return actor_loss.mean(0), metrics

        # Diffusion-QL Q function, basically just autodiffing the
        # computation all the way from action sampling to Q computation
        def actor_loss_fn_diffQ(
                score_model_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            # compute new action samples using params so we can autodiff
            # through
            # note we need to convert to batch operation
            
            key, rng = jax.random.split(agent.rng)
            actions_actorloss, rng = ddpm_sampler(
                agent.score_model.apply_fn,
                score_model_params,
                agent.T, rng, agent.act_dim,
                batch["observations"],
                agent.alphas, agent.alpha_hats,
                agent.betas, agent.ddpm_temperature,
                agent.clip_sampler)

            # evaluate target critic on batch
            key, rng = jax.random.split(rng)
            target_q_1 = agent.target_critic_1.apply_fn(
                {"params": agent.target_critic_1.params},
                batch["observations"], actions_actorloss, True, rngs={"dropout": key})
            key, rng = jax.random.split(rng)
            target_q_2 = agent.target_critic_2.apply_fn(
                {"params": agent.target_critic_2.params},
                batch["observations"], actions_actorloss, True, rngs={"dropout": key})
            target_q = jnp.stack([target_q_1, target_q_2], 0).min(0)
            
            # compute diffusion-QL loss
            actor_loss = -target_q
            
            assert actor_loss.shape == (B,)
            metrics = tensorstats(actor_loss, 'actor_loss')
            return actor_loss.mean(0), metrics

        key, rng = jax.random.split(agent.rng, 2)
        grads, metrics = jax.grad(actor_loss_fn_policygrad, has_aux=True)(
            agent.score_model.params)
        # print 2-norm of gradients
        # @jax.jit
        # def my_function(x):
        #     # Some operations
        #     return jnp.linalg.norm(x.flatten())
        
        # value = jax.device_get(my_function(grads['MLP_0']['Dense_0']['kernel']))
        # print(value.item())
        score_model = agent.score_model.apply_gradients(grads=grads)
        new_agent = agent.replace(
            score_model=score_model,
            rng=rng)
        return new_agent, metrics

    @jax.jit
    def sample_actions(self, observations: jnp.ndarray):
        actions, new_agent = self.eval_actions(observations)
        key, rng = jax.random.split(new_agent.rng, 2)
        # no noising for policygrad, need to keep correct logprobs
        # noise = jax.random.normal(key, shape=actions.shape) * 0.1
        # actions = actions + noise
        # actions = jnp.clip(actions, -1.0, 1.0)
        key, rng = jax.random.split(rng, 2)
        return actions, new_agent.replace(rng=rng)

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = observations[None]

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            self.score_model.params,
            self.T, rng, self.act_dim, observations,
            self.alphas, self.alpha_hats,
            self.betas, self.ddpm_temperature,
            self.clip_sampler)
        assert actions.shape == (1, self.act_dim)
        _, rng = jax.random.split(rng, 2)
        return jnp.squeeze(actions), self.replace(rng=rng)

    @jax.jit
    def eval_actions_with_params(self, observations: jnp.ndarray, params):
        rng = self.rng
        assert len(observations.shape) == 1
        print(observations.shape)
        observations = observations[None]

        actions, rng = ddpm_sampler(
            self.score_model.apply_fn,
            params,
            self.T, rng, self.act_dim, observations,
            self.alphas, self.alpha_hats,
            self.betas, self.ddpm_temperature,
            self.clip_sampler)
        assert actions.shape == (1, self.act_dim)
        _, rng = jax.random.split(rng, 2)
        return jnp.squeeze(actions), self.replace(rng=rng)

    
    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        new_agent, critic_info = new_agent.update_q(batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        return new_agent, {**actor_info, **critic_info}
