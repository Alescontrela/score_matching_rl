import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "DiffusionPolicygradLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.discount = 0.99
    config.tau = 0.005  # For soft target updates.
    config.T = 15
    return config
