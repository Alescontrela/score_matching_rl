import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    config.model_cls = "ScoreMatchingLearner"
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.discount = 0.99
    config.tau = 0.005  # For soft target updates.
    config.T = 5
    '''
    M_q is probably the most important parameter to tune. Corresponds to how
    much we scale the Q-score when matching the DDPM noise model to it
    (\alpha in Alg. 1 of the main paper https://arxiv.org/pdf/2312.11752):
    
    min_\theta \|\epsilon_\theta(a, s, t) - M_q \nabla_a Q(s, a)\|_2^2

    Corresponds to something akin to how aggresive the learning process
    is: lower M_q will take more conservative steps and remain higher
    entropy, and higher M_q will take more aggressive steps towards
    optimal actions.

    Also analogous to an explore/expoit tradeoff, but note that
    QSM will typically learn explorative policies on convergence,
    see Fig. 4 & 5 of the main paper.

    For more difficult tasks like quadruped_walk and humanoid_walk,
    a more aggressive M_q = 120 may perform better.
    '''
    config.M_q = 50
    config.critic_hidden_dims=(512,512)
    config.actor_hidden_dims=(512,512)
    return config
