# Score Matching RL

# Reproducing Results

[Script location](examples/states/train_score_matching_online.py)

Run 
```
cd examples/states
python3 train_score_matching_online.py
```

# Important File Locations

[Main run script were variant dictionary is passed.](/examples/states/train_diffusion_offline.py)

[DDPM Implementation.](/jaxrl5/networks/diffusion.py)

[Score matching RL learner.](/jaxrl5/agents/score_matching/score_matching_learner.py)

# Installation

Run
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).

Based from a re-implementation of https://github.com/ikostrikov/jaxrl 
