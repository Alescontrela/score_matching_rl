import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "ScoreMatchingLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.discount = 0.99
    config.num_qs = 2
    config.tau = 0.005  # For soft target updates.
    return config
