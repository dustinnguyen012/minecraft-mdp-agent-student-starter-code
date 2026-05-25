"""
student/mdp_agent.py — The MDP Agent
=====================================

ONE agent. It runs both **Policy Iteration** and **Value Iteration** to solve
its MDP, {S, A, P(s'|s,a), R(s,a,s'), γ}, and funnels itself toward the
optimal policy π*.

This is Planning in MDPs (see lecture: "Planning in MDPs and Policy Iteration"):

    Agent experiences the world  ─►  learns T̂(s'|s,a) from counts
                ▼
    Replan periodically:
        π_PI ← policy_iteration(T̂, R, γ)      ← finite-step convergence
        π_VI ← value_iteration(T̂, R, γ)       ← asymptotic convergence
                ▼
        deploy π (pick either; both converge to π*)

The two algorithms complement each other:
  • PI searches the combinatorial space of policies.  Finite-step convergence.
  • VI searches the continuous space of value functions. Asymptotic convergence.
Your writeup should compare them on the same T̂.

TODO blocks you must implement (search for "TODO" in this file):
  • policy_iteration()   — the function itself, plus _evaluate_policy() and
                           _improve_policy() helpers.
  • value_iteration()    — the Bellman-optimality fixed-point iteration.

The TransitionMatrix class is provided in student/agent.py and handles
count-based estimation of T̂ — do not modify it.

Usage:
    export MDP_API_KEY="your-api-key-here"
    export MDP_SERVER_URL="https://your-server-url"
    python -m student.mdp_agent
"""

import os
import re
import sys
import time
import pickle
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from engine.minecraft_env import MinecraftMDPEnv
from student.mdp_definition import (
    state_fn, reward_fn, terminal_fn, prior_transitions,
    GAMMA, MAX_STEPS, NUM_EPISODES,
)
from student.agent import TransitionMatrix   # provided infra; do not modify


# ── Configuration ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("MDP_API_KEY", "YOUR_API_KEY_HERE")
SERVER_URL = os.environ.get("MDP_SERVER_URL", "https://localhost")
BOT_NAME = os.environ.get("BOT_NAME", "my_bot")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SAVE_EVERY = 5   # write a pkl snapshot every N episodes


# ═══════════════════════════════════════════════════════════════════════════════
# Pkl save / load (provided — you don't need to modify these)
# ═══════════════════════════════════════════════════════════════════════════════
# `save_checkpoint` writes a pickle snapshot of your learned T̂ + policy under
# results/<BOT_NAME>_ep<N>.pkl. The write is atomic (tmp + os.replace) so a
# mid-write SIGKILL can't leave a truncated file. `load_latest_checkpoint`
# scans results/ for the highest-episode snapshot for BOT_NAME and returns a
# hydrated TransitionMatrix + policy dict + episode counter, or empty state if
# nothing found.
#
# These feed scripts/analyze.py, which expects files matching
# results/<BOT_NAME>_ep<N>.pkl with keys T_counts, T_totals, policy, episode.
# Resume-on-restart also works: every run() call loads the latest checkpoint
# if present, so killing a bot mid-training and re-launching picks up where
# you left off.

KEEP_LATEST_PKLS = 10  # rolling retention: how many ep*.pkl per bot to keep


def save_checkpoint(T, policy, episode, epsilon=None):
    """Atomically save T̂, policy, and episode to results/<BOT_NAME>_ep<N>.pkl.

    After writing, prunes older ep*.pkl for this bot down to KEEP_LATEST_PKLS
    so results/ doesn't balloon across a multi-hour training run. The file we
    just wrote is always in the keep set (it has the highest episode number).
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"{BOT_NAME}_ep{episode}.pkl")
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({
            "bot_name": BOT_NAME,
            "episode": episode,
            "epsilon": epsilon,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "num_states": len(T.observed_states),
            "num_transitions": T.num_transitions,
            "T_counts": dict(T._counts),
            "T_totals": dict(T._sa_totals),
            "policy": dict(policy) if policy else {},
        }, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)
    _prune_old_checkpoints(KEEP_LATEST_PKLS)


def _prune_old_checkpoints(keep: int) -> None:
    """Delete all but the `keep` highest-episode ep*.pkl for BOT_NAME."""
    try:
        pattern = re.compile(rf"^{re.escape(BOT_NAME)}_ep(\d+)\.pkl$")
        candidates = []
        with os.scandir(SAVE_DIR) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                m = pattern.match(entry.name)
                if m:
                    candidates.append((int(m.group(1)), entry.path))
        candidates.sort(reverse=True)
        for _, stale in candidates[keep:]:
            try:
                os.remove(stale)
            except OSError:
                pass  # concurrent delete is fine
    except OSError:
        pass


def load_latest_checkpoint(current_state_arity=None):
    """Load the highest-episode pkl for BOT_NAME.

    Returns (T, policy, episode, epsilon) — empty/default if nothing found
    or if the saved state-tuple arity doesn't match the caller's state_fn
    (which would crash planning).

    Pass `current_state_arity = len(state_fn({}))` to enable arity filtering;
    any saved state tuple whose length differs is silently dropped.
    """
    if not os.path.isdir(SAVE_DIR):
        return TransitionMatrix(prior_fn=prior_transitions), {}, 0, None
    pattern = re.compile(rf"^{re.escape(BOT_NAME)}_ep(\d+)\.pkl$")
    candidates = []
    for name in os.listdir(SAVE_DIR):
        m = pattern.match(name)
        if m and os.path.getsize(os.path.join(SAVE_DIR, name)) > 0:
            candidates.append((int(m.group(1)), name))
    if not candidates:
        return TransitionMatrix(prior_fn=prior_transitions), {}, 0, None
    candidates.sort()
    _, latest = candidates[-1]
    with open(os.path.join(SAVE_DIR, latest), "rb") as f:
        data = pickle.load(f)

    def _arity_ok(s):
        if current_state_arity is None:
            return True
        return isinstance(s, tuple) and len(s) == current_state_arity

    T = TransitionMatrix(prior_fn=prior_transitions)
    # Rebuild T̂ from saved counts, filtering out stale-arity keys.
    for s, a_map in (data.get("T_counts") or {}).items():
        if not _arity_ok(s):
            continue
        clean_a_map = {}
        for a, sp_map in a_map.items():
            clean_sp = {sp: c for sp, c in sp_map.items() if _arity_ok(sp)}
            if clean_sp:
                clean_a_map[a] = clean_sp
        if clean_a_map:
            T._counts[s] = clean_a_map
            T._sa_totals[s] = {a: sum(sp.values()) for a, sp in clean_a_map.items()}
            T._observed_states.add(s)
            for a in clean_a_map:
                for sp in clean_a_map[a]:
                    T._observed_states.add(sp)
    policy = {s: a for s, a in (data.get("policy") or {}).items() if _arity_ok(s)}
    episode = int(data.get("episode") or 0)
    epsilon = data.get("epsilon")
    print(f"  [resume] loaded {latest}: {len(T.observed_states)} states, "
          f"{T.num_transitions} transitions, episode {episode}")
    return T, policy, episode, epsilon


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY ITERATION (you implement this)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Pseudocode from lecture (slide "Policy iteration"):
#
#   1. Choose an initial policy π : S → A.
#   2. Repeat until convergence:
#        V^π ← evaluate(π)          (Bellman expectation equation)
#        π'  ← greedy w.r.t. V^π    (π'(s) = argmax_a Q^π(s,a))
#        if π' == π:  return π      (policy stable → optimal)
#        π ← π'
# ═══════════════════════════════════════════════════════════════════════════════

def _evaluate_policy(policy, T, R, states, gamma, eval_sweeps=50, eval_theta=1e-4):
    """
    Policy evaluation — compute V^π for a FIXED policy π.

    Iteratively applies the Bellman expectation equation:
        V^π(s) ← Σ_{s'} T(s, π(s), s') · [ R(s, π(s), s') + γ · V^π(s') ]

    Sweep all states until max |ΔV| < eval_theta, or eval_sweeps hits.

    Parameters
    ----------
    policy : dict {state: action}
    T : TransitionMatrix
    R : callable(s, a, s') -> float
    states : iterable of state tuples
    gamma : discount factor
    eval_sweeps, eval_theta : convergence controls

    Returns
    -------
    V : dict {state: float}
    """
    V = {s: 0.0 for s in states}

    for sweep in range(eval_sweeps):
        delta = 0.0
        for s in states:
            v_old = V[s]
            a = policy[s]

            # ── TODO ─────────────────────────────────────────────
            # Update V[s] using the Bellman expectation equation under
            # action a = policy[s]:
            #
            #     V^π(s) = Σ_{s'} T(s, a, s') · [ R(s, a, s') + γ · V^π(s') ]
            #
            # API available:
            #   T.get_transitions(s, a)   # successor distribution
            #   R(s, a, s_prime)          # immediate reward
            #   V.get(s_prime, 0.0)       # current estimate; default 0 for unseen
            # ─────────────────────────────────────────────────────

            pass  # remove when you add your code

            delta = max(delta, abs(V[s] - v_old))
        if delta < eval_theta:
            break
    return V


def _improve_policy(policy, V, T, R, states, num_actions, gamma):
    """
    Policy improvement — greedy update.

    For each state, set:
        π'(s) = argmax_a Σ_{s'} T(s,a,s') · [ R(s,a,s') + γ · V(s') ]

    Returns the new policy and a flag indicating whether it differs from the
    old one (unchanged → PI has converged).
    """
    new_policy = dict(policy)
    changed = False
    for s in states:
        old_action = policy[s]

        # ── TODO ─────────────────────────────────────────────
        # Pick the greedy action for state s given the current V, then
        # assign it to new_policy[s]. Set `changed = True` if the chosen
        # action differs from old_action.
        #
        #     π'(s) = argmax_a Σ_{s'} T(s, a, s') · [ R(s, a, s') + γ · V(s') ]
        #
        # You will need to iterate over a ∈ {0, …, num_actions-1} and
        # compute the Q-value of each candidate action.
        # ─────────────────────────────────────────────────────

        pass  # remove when you add your code

    return new_policy, changed


def policy_iteration(T, R, states, num_actions, gamma=GAMMA, max_iterations=100):
    """
    Full Policy Iteration. Returns (policy, V).

    Initialise π with a random action per state, then alternate
    _evaluate_policy + _improve_policy until the policy stops changing
    or max_iterations is reached.
    """
    policy = {s: np.random.randint(num_actions) for s in states}
    V = {s: 0.0 for s in states}

    # ── TODO ─────────────────────────────────────────────────────
    # Alternate policy evaluation (_evaluate_policy) and policy improvement
    # (_improve_policy) until the policy is stable, or max_iterations is
    # reached. Return (policy, V) once converged.
    # ─────────────────────────────────────────────────────────────

    return policy, V  # placeholder — replace with your loop


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE ITERATION (you implement this)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Pseudocode from lecture (slide "Algorithm for value iteration"):
#
#   1. Initialise V₀(s) = 0 for all s ∈ S.
#   2. Iterate until convergence:
#        V_{k+1}(s) = max_a [ Σ_{s'} T(s,a,s') · (R(s,a,s') + γ · V_k(s')) ]
#   3. Derive the policy:
#        π*(s) = argmax_a Σ_{s'} T(s,a,s') · (R(s,a,s') + γ · V(s'))
# ═══════════════════════════════════════════════════════════════════════════════

def value_iteration(T, R, states, num_actions, gamma=GAMMA,
                    max_iterations=1000, theta=1e-6):
    """
    Full Value Iteration. Returns (policy, V).

    Loops the Bellman optimality update until max |ΔV| < theta, then
    extracts the greedy policy from the converged V.
    """
    V = {s: 0.0 for s in states}
    policy = {s: 0 for s in states}

    # ── TODO ─────────────────────────────────────────────────────
    # Apply the Bellman-optimality update until the value function stops
    # changing (max |ΔV| < theta) or max_iterations is reached, then
    # extract the greedy policy from V.
    #
    #     V_{k+1}(s) = max_a Σ_{s'} T(s, a, s') · [ R(s, a, s') + γ · V_k(s') ]
    #
    # Return (policy, V).
    # ─────────────────────────────────────────────────────────────

    return policy, V  # placeholder — replace with your loop


# ═══════════════════════════════════════════════════════════════════════════════
# Bridge plumbing (provided — no TODOs below here)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_config():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Set your API key. Either:")
        print("  export MDP_API_KEY='your-key'")
        print("  or edit API_KEY at the top of mdp_agent.py")
        sys.exit(1)


def _wait_for_server(url: str, timeout: int = 30):
    """Poll /health until the bridge responds. /health is auth-gated, so we
    include the X-API-Key header like every other request."""
    import requests
    print(f"Connecting to {url} ...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=5, verify=False,
                             headers={"X-API-Key": API_KEY})
            if r.ok and r.json().get("ok"):
                print(" connected!")
                return
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)
    print(f"\nERROR: Server not reachable at {url}")
    sys.exit(1)


def _make_env():
    _check_config()
    _wait_for_server(SERVER_URL)
    return MinecraftMDPEnv(
        server_url=SERVER_URL,
        max_steps=MAX_STEPS,
        api_key=API_KEY,
        reward_fn=reward_fn,
        state_fn=state_fn,
        terminal_fn=terminal_fn,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP: Explore → Learn T̂ → Plan (PI + VI) → Deploy
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    env = _make_env()
    NUM_ACTIONS = env.num_actions
    print(f"Loaded {NUM_ACTIONS} actions from bridge: {env.ACTION_NAMES}")

    # Resume-from-pkl if a checkpoint exists for this BOT_NAME. Drops state
    # keys whose arity no longer matches your current state_fn (useful if
    # you rewrite state_fn mid-project). Returns fresh T + empty policy +
    # episode=0 when no checkpoint is found.
    current_arity = len(state_fn(env.get_raw_state()))
    T, policy, resume_episode, resume_epsilon = load_latest_checkpoint(current_arity)
    all_states = set(T.observed_states)
    epsilon = resume_epsilon if resume_epsilon is not None else 0.5
    replan_every = 25            # re-solve every N episodes

    episode_rewards = []

    for episode in range(resume_episode, resume_episode + NUM_EPISODES):
        state, info = env.reset()
        available = info.get("available_actions", frozenset(range(NUM_ACTIONS)))
        all_states.add(state)
        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy action selection with action-masking
            available_list = list(available) if available else list(range(NUM_ACTIONS))
            if np.random.random() < epsilon or state not in policy or policy[state] not in available:
                action = np.random.choice(available_list)
            else:
                action = policy[state]

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_available = info.get("available_actions", frozenset(range(NUM_ACTIONS)))

            T.record(state, action, next_state)
            all_states.add(next_state)

            total_reward += reward
            state = next_state
            available = next_available

        epsilon = max(0.05, epsilon * 0.995)
        episode_rewards.append(total_reward)

        # ── Planning: run BOTH PI and VI on the current T̂ ────────────
        # Both converge to π*. You can deploy either or compare.
        # Here we run PI and use VI as a check; swap freely.
        if (episode + 1) % replan_every == 0 and len(all_states) > 5:
            states_list = list(all_states)
            print(f"\n  Re-planning on {len(states_list)} states, "
                  f"{T.num_transitions} transitions:")
            try:
                pi_policy, pi_V = policy_iteration(
                    T, reward_fn, states_list, NUM_ACTIONS, gamma=GAMMA)
                vi_policy, vi_V = value_iteration(
                    T, reward_fn, states_list, NUM_ACTIONS, gamma=GAMMA,
                    max_iterations=200, theta=1e-4)

                # Compare: how often do PI and VI pick the same action?
                agree = sum(1 for s in states_list if pi_policy[s] == vi_policy[s])
                print(f"  PI/VI agreement: {agree}/{len(states_list)} states")

                # Deploy VI's policy (or PI's — student's choice).
                policy = vi_policy
            except Exception as e:
                print(f"  Planning failed: {e}")

        # Logging (format matches what scripts/analyze.py parses)
        avg = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        print(
            f"[Bot {BOT_NAME}] Ep {episode + 1:5d} | "
            f"R: {total_reward:7.1f} | Avg(10): {avg:7.1f} | "
            f"ε: {epsilon:.4f} | States: {len(all_states):5d} | "
            f"Transitions: {T.num_transitions}"
        )

        # Snapshot every SAVE_EVERY episodes so scripts/analyze.py can read
        # progress, and so a crash / Ctrl-C keeps a recoverable checkpoint.
        if (episode + 1) % SAVE_EVERY == 0:
            save_checkpoint(T, policy, episode + 1, epsilon=epsilon)

    # Final save at end-of-run.
    save_checkpoint(T, policy, resume_episode + NUM_EPISODES, epsilon=epsilon)

    # ── Results ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes:       {NUM_EPISODES}")
    print(f"Final avg reward:     {np.mean(episode_rewards[-10:]):.1f}")
    print(f"Best episode reward:  {max(episode_rewards):.1f}")
    print(f"Unique states:        {len(all_states)}")
    print(f"Transitions learned:  {T.num_transitions}")
    print(f"Final epsilon:        {epsilon:.4f}\n")

    action_names = env.get_action_names()
    print("Learned Policy (sample states):")
    print("-" * 50)
    for s in sorted(policy.keys())[:20]:
        print(f"  State {s} → {action_names[policy[s]]}")

    return episode_rewards


if __name__ == "__main__":
    run()
