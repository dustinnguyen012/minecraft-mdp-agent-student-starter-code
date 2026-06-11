"""
student/mdp_agent.py
--------------------
MDP agent: Value Iteration, Policy Iteration, episode loop.

Run:
    export MDP_API_KEY="313c038f6ec1c2f1a09a383e06df99cb"
    export MDP_SERVER_URL="https://tile.ucsd.edu"
    mkdir -p results && python -m student.mdp_agent
"""

import os, sys, glob, pickle, logging, random, time, warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from student.agent import TransitionMatrix
from student.mdp_definition import state_fn, reward_fn, terminal_fn, prior_transitions, MAX_STEPS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
log = logging.getLogger(__name__)

# ── Hyperparameters ───────────────────────────────────────────────────────────
GAMMA         = 0.97
EPSILON_START = 0.90
EPSILON_DECAY = 0.994
EPSILON_MIN   = 0.10
THETA         = 1e-4
REPLAN_EVERY  = 5

BOT_NAME         = os.environ.get("BOT_NAME", "my_bot")
SAVE_EVERY       = 5
KEEP_LATEST_PKLS = 10
RESULTS_DIR      = "results"


# ── Reward table (TransitionMatrix doesn't store rewards) ─────────────────────
class RewardTable:
    def __init__(self, data=None):
        self._t = data if data is not None else {}

    def record(self, s, a, r):
        self._t.setdefault(s, {}).setdefault(a, [0.0, 0])
        self._t[s][a][0] += r
        self._t[s][a][1] += 1

    def get(self, s, a, default=0.0):
        e = self._t.get(s, {}).get(a)
        return default if e is None else e[0] / e[1]

    def to_dict(self):
        return {s: {a: list(v) for a, v in ad.items()} for s, ad in self._t.items()}


# ── Build plain dicts for VI / PI ─────────────────────────────────────────────
def _build_tables(T: TransitionMatrix, R: RewardTable, states):
    t_dict, r_dict = {}, {}
    for s in states:
        tried = T._sa_totals.get(s, {}).keys()
        if not tried:
            continue
        t_dict[s], r_dict[s] = {}, {}
        for a in tried:
            t_dict[s][a] = {sp: p for p, sp in T.get_transitions(s, a)}
            r_dict[s][a] = R.get(s, a)
    return t_dict, r_dict


# ── Planning helpers ──────────────────────────────────────────────────────────
def _evaluate_policy(policy, T, R, states, gamma):
    V = {s: 0.0 for s in states}
    while True:
        delta = 0.0
        for s in states:
            a = policy.get(s)
            if a is None or s not in T or a not in T[s]:
                continue
            r     = R.get(s, {}).get(a, 0.0)
            v_new = r + gamma * sum(p * V.get(sp, 0.0) for sp, p in T[s][a].items())
            delta = max(delta, abs(V[s] - v_new))
            V[s]  = v_new
        if delta < THETA:
            break
    return V


def _greedy_policy(V, T, R, states, num_actions, gamma):
    policy = {}
    for s in states:
        if s not in T:
            continue
        best_a, best_q = None, float("-inf")
        for a, sp_probs in T[s].items():
            r = R.get(s, {}).get(a, 0.0)
            q = r + gamma * sum(p * V.get(sp, 0.0) for sp, p in sp_probs.items())
            if q > best_q:
                best_q, best_a = q, a
        if best_a is not None:
            policy[s] = best_a
    return policy


# ── Policy Iteration ──────────────────────────────────────────────────────────
def policy_iteration(T, R, states, num_actions, gamma=GAMMA):
    """
    Alternate policy evaluation and greedy improvement until stable.
    Returns (policy, V).
    """
    policy = {}
    for s in states:
        tried = list(T.get(s, {}).keys())
        if tried:
            policy[s] = random.choice(tried)

    for i in range(500):
        V          = _evaluate_policy(policy, T, R, states, gamma)
        new_policy = _greedy_policy(V, T, R, states, num_actions, gamma)
        if new_policy == policy:
            log.debug(f"PI converged in {i+1} iters")
            return policy, V
        policy = new_policy

    log.warning("PI hit 500-iter cap")
    return policy, _evaluate_policy(policy, T, R, states, gamma)


# ── Value Iteration ───────────────────────────────────────────────────────────
def value_iteration(T, R, states, num_actions, gamma=GAMMA):
    """
    Apply Bellman optimality update until max|ΔV| < THETA, then extract policy.
    V_{k+1}(s) = max_a [ R(s,a) + γ · Σ T(s,a,s') · V_k(s') ]
    Returns (policy, V).
    """
    V = {s: 0.0 for s in states}
    for i in range(10_000):
        delta = 0.0
        for s in states:
            if s not in T:
                continue
            best_v = float("-inf")
            for a, sp_probs in T[s].items():
                r = R.get(s, {}).get(a, 0.0)
                q = r + gamma * sum(p * V.get(sp, 0.0) for sp, p in sp_probs.items())
                if q > best_v:
                    best_v = q
            if best_v > float("-inf"):
                delta = max(delta, abs(V[s] - best_v))
                V[s]  = best_v
        if delta < THETA:
            log.debug(f"VI converged in {i+1} iters (δ={delta:.2e})")
            break
    else:
        log.warning("VI hit 10000-iter cap")
    return _greedy_policy(V, T, R, states, num_actions, gamma), V


# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(T, R, policy, episode, epsilon=None, extra=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    target  = os.path.join(RESULTS_DIR, f"{BOT_NAME}_ep{episode}.pkl")
    payload = {"T_counts": T._counts, "T_totals": T._sa_totals,
               "R_table": R.to_dict(), "policy": policy,
               "episode": episode, "epsilon": epsilon, "timestamp": time.time(),
               "n_states": len(T.observed_states),
               "n_transitions": T.num_transitions}
    if extra:
        payload.update(extra)
    tmp = target + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, target)
    log.info(f"Checkpoint → {target}  "
             f"(states={payload['n_states']} trans={payload['n_transitions']})")
    pkls = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{BOT_NAME}_ep*.pkl")),
                  key=lambda p: int(p.rsplit("ep",1)[1].split(".")[0]))
    for old in pkls[:-KEEP_LATEST_PKLS]:
        try: os.remove(old)
        except OSError: pass


def load_latest_checkpoint(current_state_arity=None):
    pkls = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{BOT_NAME}_ep*.pkl")),
                  key=lambda p: int(p.rsplit("ep",1)[1].split(".")[0]))
    if not pkls:
        log.info("No checkpoint — starting fresh.")
        return TransitionMatrix(prior_fn=prior_transitions), RewardTable(), {}, 0, EPSILON_START
    with open(pkls[-1], "rb") as f:
        data = pickle.load(f)
    T = TransitionMatrix(prior_fn=prior_transitions)
    T._counts    = data.get("T_counts", {})
    T._sa_totals = data.get("T_totals", {})
    T._observed_states = set()
    for s in T._counts:
        T._observed_states.add(s)
        for sp_dict in T._counts[s].values():
            T._observed_states.update(sp_dict.keys())
    R       = RewardTable(data.get("R_table"))
    policy  = data.get("policy", {})
    episode = data.get("episode", 0)
    epsilon = data.get("epsilon", EPSILON_START)
    if current_state_arity is not None:
        policy = {s: a for s, a in policy.items() if len(s) == current_state_arity}
    log.info(f"Resumed ep={episode} | states={len(T.observed_states)} | "
             f"trans={T.num_transitions} | ε={epsilon:.3f}")
    return T, R, policy, episode, epsilon


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    try:
        from engine.minecraft_env import MinecraftMDPEnv
    except ImportError as e:
        log.error(f"Cannot import MinecraftMDPEnv: {e}"); sys.exit(1)

    api_key    = os.environ.get("MDP_API_KEY", "")
    server_url = os.environ.get("MDP_SERVER_URL", "https://tile.ucsd.edu")
    if not api_key:
        log.error("Set MDP_API_KEY before running."); sys.exit(1)

    env = MinecraftMDPEnv(server_url=server_url, api_key=api_key,
                          state_fn=state_fn, reward_fn=reward_fn,
                          terminal_fn=terminal_fn)
    num_actions = env.num_actions
    log.info(f"Connected | {num_actions} actions | BOT={BOT_NAME}")

    state_arity              = len(state_fn({}))
    T, R, policy, start_ep, eps = load_latest_checkpoint(state_arity)

    episode_rewards, episode_lengths, pi_vi_agreements = [], [], []

    log.info(f"γ={GAMMA}  ε={eps:.2f}→{EPSILON_MIN}  replan_every={REPLAN_EVERY}")

    try:
        episode = start_ep
        while True:
            episode     += 1
            goal_reached = False

            state, info = env.reset()
            ep_reward   = 0.0
            step        = 0

            while not terminal_fn(state, step):
                available = list(info.get("available_actions", range(num_actions)))

                if random.random() < eps:
                    action = random.choice(available)
                else:
                    action = policy.get(state)
                    if action is None or action not in available:
                        action = random.choice(available)

                next_state, reward, terminated, truncated, info = env.step(action)

                raw = env.get_raw_state()
                if action == 77 and raw.get("inventory", {}).get("iron_pickaxe", 0) > 0:
                    reward      += 30.0
                    goal_reached = True
                    log.info(f"  🏆 Iron pickaxe! ep={episode} step={step}")

                ep_reward += reward
                step      += 1
                T.record(state, action, next_state)
                R.record(state, action, reward)
                state = next_state

                if terminated or truncated or goal_reached:
                    break

            eps = max(EPSILON_MIN, eps * EPSILON_DECAY)
            episode_rewards.append(ep_reward)
            episode_lengths.append(step)

            if episode % REPLAN_EVERY == 0:
                known = list(T.observed_states)
                if len(known) >= 3:
                    t_dict, r_dict = _build_tables(T, R, known)
                    t0 = time.time()
                    pi_pol, _ = policy_iteration(t_dict, r_dict, known, num_actions)
                    t_pi = time.time() - t0
                    t0 = time.time()
                    vi_pol, _ = value_iteration(t_dict, r_dict, known, num_actions)
                    t_vi = time.time() - t0
                    common = [s for s in known if s in pi_pol and s in vi_pol]
                    if common:
                        pct = sum(pi_pol[s] == vi_pol[s] for s in common) / len(common) * 100
                        pi_vi_agreements.append(pct)
                        log.info(f"  Replan ep{episode}: VI {t_vi:.2f}s | PI {t_pi:.2f}s | "
                                 f"agreement {pct:.1f}% ({len(common)} states)")
                    policy = vi_pol

            avg10 = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            log.info(f"Ep {episode:4d} | r={ep_reward:7.2f} | avg10={avg10:6.2f} | "
                     f"steps={step:4d} | states={len(T.observed_states):4d} | "
                     f"trans={T.num_transitions:7d} | ε={eps:.3f}"
                     + (" ✓IRON" if goal_reached else ""))

            if episode % SAVE_EVERY == 0:
                save_checkpoint(T, R, policy, episode, epsilon=eps,
                                extra={"episode_rewards": episode_rewards,
                                       "episode_lengths": episode_lengths,
                                       "pi_vi_agreements": pi_vi_agreements})

    except KeyboardInterrupt:
        log.info("Interrupted — saving...")
    except Exception as e:
        log.error(f"Crashed: {e}", exc_info=True)
        time.sleep(5)
    finally:
        save_checkpoint(T, R, policy, episode, epsilon=eps,
                        extra={"episode_rewards": episode_rewards,
                               "episode_lengths": episode_lengths,
                               "pi_vi_agreements": pi_vi_agreements})
        n = len(episode_rewards)
        if n:
            log.info("=" * 55)
            log.info(f"Episodes: {n} | States: {len(T.observed_states)} | "
                     f"Trans: {T.num_transitions}")
            log.info(f"Avg reward first 10: {sum(episode_rewards[:10])/min(10,n):.2f}")
            log.info(f"Avg reward last  10: {sum(episode_rewards[-10:])/min(10,n):.2f}")
            if pi_vi_agreements:
                log.info(f"Avg PI/VI agreement: "
                         f"{sum(pi_vi_agreements)/len(pi_vi_agreements):.1f}%")
            log.info("=" * 55)


if __name__ == "__main__":
    run()