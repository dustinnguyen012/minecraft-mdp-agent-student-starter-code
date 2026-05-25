"""
student/mdp_definition.py — Define Your MDP Here
=================================================

Define the components of your Markov Decision Process:

    MDP = {S, A, P(s'|s,a), R(s,a,s'), γ}

You must implement:
    1. state_fn(raw)          — Convert observation → hashable state tuple
    2. reward_fn(s, a, s')    — Immediate reward for a transition
    3. terminal_fn(s, step)   — Whether the episode should end
    4. prior_transitions(s,a) — Starting transition matrix T(s,a,s')

The transition matrix is your model of the world. You start with a prior
that encodes your initial beliefs, then your agent updates it from real
observations using count-based estimation:

    T̂(s,a,s') = count(s,a,s') / count(s,a)

Your agents use BOTH Policy Iteration AND Value Iteration on this matrix to
compute an optimal policy via Bellman's equation:

    V^π(s) = R(s) + γ Σ_{s'} P(s'|s,π(s)) V^π(s')       (Bellman equation)
    V*(s)  = max_a [R(s) + γ Σ_{s'} P(s'|s,a) V*(s')]    (Bellman optimality)

Raw observation fields are documented in the README "Raw State Fields"
section. The action list is fetched at runtime — see `env.ACTION_NAMES`
and `env.num_actions`.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# MDP PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

GAMMA = 0.99          # Discount factor
MAX_STEPS = 500       # Max steps per episode before truncation
NUM_EPISODES = 100    # Training episodes
BUCKET_SIZE = 10      # Grid bucketing (blocks per cell) for position discretization


# ═══════════════════════════════════════════════════════════════════════════════
# 1. STATE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def state_fn(raw: dict) -> tuple:
    """
    Convert the raw observation into a hashable state tuple.

    Parameters
    ----------
    raw : dict
        Observation from the environment. See the README "Raw State Fields"
        section for the full list of keys. You can also call
        `env.get_raw_state()` to inspect a live dict.

    Returns
    -------
    tuple
        Hashable state representation. Must have the SAME shape every call.
    """
    raise NotImplementedError("Implement state_fn: convert raw observation to a state tuple")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. REWARD FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def reward_fn(old_state: tuple, action: int, new_state: tuple) -> float:
    """
    Compute immediate reward R(s, a, s') for a transition.

    Parameters
    ----------
    old_state : tuple — State before the action
    action    : int   — Action index (0 through env.num_actions-1)
    new_state : tuple — State after the action

    Returns
    -------
    float : Scalar reward.
    """
    raise NotImplementedError("Implement reward_fn: define rewards for transitions")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TERMINAL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def terminal_fn(state: tuple, step_count: int) -> bool:
    """
    Return True if the episode should end.

    Parameters
    ----------
    state      : tuple — Current state
    step_count : int   — Steps taken this episode
    """
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PRIOR TRANSITION MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def prior_transitions(state, action):
    """
    Prior transition model T(s, a) = [(probability, next_state), ...].

    Encodes your initial beliefs about the dynamics before any observation.
    The agent updates from real experience via T̂(s,a,s') = count(s,a,s') / count(s,a).

    Returns
    -------
    list of (float, tuple)
        Probabilities must sum to 1.0.
    """
    # Default: self-loop (nothing changes). Students override.
    return [(1.0, state)]
