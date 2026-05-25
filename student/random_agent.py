"""
student/random_agent.py — Reference random agent (demo, not the assignment).
=============================================================================

Run this BEFORE you start writing mdp_agent.py. It is the simplest possible
Minecraft MDP agent — it connects to the bridge, discovers the action list,
polls the current state, picks a random available action, and loops.

Purpose:
  1. Verify your API key and server URL work.
  2. See what `env.step()` returns — observation tuple, reward, done flag,
     info dict (with raw_state + available_actions + wait_ms hint).
  3. Watch the bot act in-game.
  4. Understand the control flow your mdp_agent.py has to implement:
         reset → observe → pick-action → step → record → repeat.

This file is intentionally tiny and has no TODOs. Do NOT reuse it as the
basis for your assignment — mdp_agent.py is where you implement PI and VI.

Usage:
    export MDP_API_KEY="your-api-key"
    export MDP_SERVER_URL="https://your-server-url"
    python -m student.random_agent
"""

import os
import sys
import time
import warnings
import random

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

from engine.minecraft_env import MinecraftMDPEnv


API_KEY    = os.environ.get("MDP_API_KEY", "YOUR_API_KEY_HERE")
SERVER_URL = os.environ.get("MDP_SERVER_URL", "https://localhost")
MAX_STEPS  = 100   # steps per episode (keep small; this is just a demo)
EPISODES   = 3     # episodes before exit


def _trivial_state_fn(raw):
    """
    Turn the raw observation dict into a tuple you can print / hash.

    For the assignment you'll write a much smarter state_fn in
    mdp_definition.py. For a random agent the state isn't used, so we just
    return a minimal tuple of the fields most useful for debugging.
    """
    return (
        raw.get("grid_x", 0),
        raw.get("grid_z", 0),
        raw.get("health_bin", 3),
        bool(raw.get("has_wood", False)),
        bool(raw.get("has_planks", False)),
    )


def _trivial_reward_fn(old_state, action, new_state):
    """Reward doesn't matter for a random agent. Return 0 for every step."""
    return 0.0


def _never_terminal(state, step_count):
    """Let MAX_STEPS truncate; no early termination."""
    return False


def main():
    if API_KEY == "YOUR_API_KEY_HERE":
        print("ERROR: Set MDP_API_KEY in your environment.")
        sys.exit(1)

    print(f"Connecting to {SERVER_URL} …")
    env = MinecraftMDPEnv(
        server_url=SERVER_URL,
        max_steps=MAX_STEPS,
        api_key=API_KEY,
        state_fn=_trivial_state_fn,
        reward_fn=_trivial_reward_fn,
        terminal_fn=_never_terminal,
    )

    # The env fetches the action list from the bridge on init.
    # No hardcoding — env.num_actions and env.ACTION_NAMES are authoritative.
    print(f"Loaded {env.num_actions} actions: {env.ACTION_NAMES[:8]} …")
    print()

    for episode in range(EPISODES):
        print(f"═══ Episode {episode + 1}/{EPISODES} ═══")

        # reset teleports the bot back to its assigned grid cell if needed,
        # and returns (initial_state_tuple, info).
        state, info = env.reset()
        print(f"  reset → state={state}  available_actions={len(info['available_actions'])}")

        # Inspect the full raw observation. This is what state_fn gets called on.
        raw = env.get_raw_state()
        inv_preview = ", ".join(
            f"{name}×{count}"
            for name, count in sorted(
                raw.get("inventory", {}).items(), key=lambda kv: -kv[1]
            )[:5]
        ) or "(empty)"
        print(f"  position: ({raw.get('x')}, {raw.get('y')}, {raw.get('z')})")
        print(f"  health:   {raw.get('health_raw')} (bin={raw.get('health_bin')})")
        print(f"  inventory top 5: {inv_preview}")
        print(f"  flags: water_adjacent={raw.get('water_adjacent')} "
              f"has_table_nearby={raw.get('has_table_nearby')} "
              f"on_grass={raw.get('on_grass')}")
        print()

        for step in range(MAX_STEPS):
            # Pick a uniformly-random action from the currently-available set.
            # The bridge masks out actions that can't execute right now
            # (missing materials, no nearby mob, wrong tool tier, etc.), so
            # available_actions is always a safe choice pool.
            available = list(info["available_actions"])
            if not available:
                print("  (no actions available — bridge returned empty set)")
                break
            action = random.choice(available)
            action_name = env.ACTION_NAMES[action]

            # env.step hands the action to the bridge, waits for confirmation,
            # and returns (next_state, reward, terminated, truncated, info).
            next_state, reward, terminated, truncated, info = env.step(action)

            if step < 3 or step % 20 == 0:
                # Print a handful of steps — not every one, to keep output readable.
                print(f"  step {step:3d}: {action_name:22s} "
                      f"→ reward={reward:+.2f}  state={next_state}")

            state = next_state
            if terminated or truncated:
                break

            # wait_ms is embedded in the bridge response; env.step already
            # respects it by blocking on the action-lock. No explicit sleep
            # needed here — the loop naturally paces itself.

        print()

    print("Done. This random agent has no policy — to do better, implement")
    print("Value Iteration and Policy Iteration in student/mdp_agent.py.")


if __name__ == "__main__":
    main()
