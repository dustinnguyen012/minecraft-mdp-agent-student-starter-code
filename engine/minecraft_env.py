"""
engine/minecraft_env.py — Gymnasium wrapper around the Mineflayer bridge.

This environment follows the Gymnasium API so students can use standard
RL tooling. It communicates with the Node.js bridge via HTTP.
"""

import time
import requests
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class MinecraftMDPEnv(gym.Env):
    """
    A discrete Gymnasium environment backed by a live Minecraft server.

    Observation: A tuple representing the discretized world state.
    Actions:     Integers 0–14 mapped to movement, mining, crafting, combat, etc.

    Not all actions are available at all times. Check `available_actions` in
    the info dict or call `get_available_actions()` to get the current set.
    """

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        server_url: str = "https://localhost",
        max_steps: int = 200,
        api_key: str = None,
        reward_fn=None,
        state_fn=None,
        terminal_fn=None,
    ):
        """
        Parameters
        ----------
        server_url : str
            URL provided by your instructor.
        max_steps : int
            Maximum steps per episode before forced termination.
        api_key : str
            Your API key (provided by your instructor).
        reward_fn : callable(old_state, action, new_state) -> float
            Custom reward function. If None, returns -1 per step.
        state_fn : callable(raw_state_dict) -> tuple
            Custom state discretizer. If None, uses default tuple.
        terminal_fn : callable(state, step_count) -> bool
            Custom termination check. If None, only max_steps triggers done.
        """
        super().__init__()

        self._server_url = server_url.rstrip("/")
        self.max_steps = max_steps
        self._api_key = api_key

        # User-overridable functions
        self._reward_fn = reward_fn or self._default_reward
        self._state_fn = state_fn or self._default_state
        self._terminal_fn = terminal_fn or self._default_terminal

        # Fetch action list from the bridge — indices and names are defined server-side.
        self.ACTION_NAMES = self._fetch_actions()
        self.num_actions = len(self.ACTION_NAMES)

        # Spaces
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict({})  # placeholder

        # Episode tracking
        self._step_count = 0
        self._current_raw_state = None
        self._available_actions = frozenset(range(self.num_actions))

    def _fetch_actions(self) -> list:
        """Query the bridge's /actions endpoint to get the canonical action list."""
        resp = requests.get(
            f"{self._server_url}/actions", headers=self._headers,
            timeout=10, verify=False,
        )
        resp.raise_for_status()
        return list(resp.json()["actions"])

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Start a new episode. Returns (obs, info) with the bot's current state.

        The bot does NOT move. Use this at episode boundaries — position,
        inventory, and effects are all preserved so the bot's exploration
        progress carries across episodes.

        If you want the bot to drift back to its spawn cell, use
        request_reset() instead (explicit, costs an action cycle).
        """
        super().reset(seed=seed)
        raw = self._get("/raw_state")
        self._current_raw_state = raw
        self._update_available(raw)
        self._step_count = 0
        obs = self._state_fn(raw)
        info = {
            "raw_state": raw,
            "available_actions": self._available_actions,
        }
        return obs, info

    def peek_state(self):
        """Get current state without moving the bot. Same as reset() but
        doesn't touch the Gymnasium episode counter.

        Returns (obs, info) with the same structure as reset().
        """
        raw = self._get("/raw_state")
        self._current_raw_state = raw
        self._update_available(raw)
        obs = self._state_fn(raw)
        info = {
            "raw_state": raw,
            "available_actions": self._available_actions,
        }
        return obs, info

    def request_reset(self):
        """Explicitly request a drift back to the bot's assigned spawn cell.

        Unlike reset(), this DOES move the bot — the server teleports it
        to its grid cell via spreadplayers. Use this when the bot is stuck
        (e.g., trapped in a cave, surrounded by water) and the policy
        decides it needs a fresh start from its home area.

        This is NOT called automatically per episode. The reference agent
        never calls it — drift only happens on death (server-controlled)
        or if you explicitly invoke this method.

        Returns (obs, info) from the new position after the drift.
        """
        resp = self._post("/reset")
        self._current_raw_state = resp.get("state", {})
        self._update_available(self._current_raw_state)
        self._step_count = 0
        obs = self._state_fn(self._current_raw_state)
        info = {
            "raw_state": self._current_raw_state,
            "available_actions": self._available_actions,
        }
        return obs, info

    def step(self, action: int):
        """Execute one discrete action and return (obs, reward, terminated, truncated, info)."""
        action_name = self.ACTION_NAMES[action]
        old_raw = self._current_raw_state

        resp = self._post("/action", json={"action": action_name})
        new_raw = resp.get("state", {})
        self._current_raw_state = new_raw
        self._update_available(new_raw)
        self._step_count += 1

        old_state = self._state_fn(old_raw)
        new_state = self._state_fn(new_raw)

        reward = self._reward_fn(old_state, action, new_state)
        terminated = self._terminal_fn(new_state, self._step_count)
        truncated = self._step_count >= self.max_steps

        info = {
            "raw_state": new_raw,
            "action_name": action_name,
            "action_succeeded": resp.get("success", True),
            "step": self._step_count,
            "available_actions": self._available_actions,
        }
        return new_state, reward, terminated, truncated, info

    def get_state(self):
        """Fetch current state without taking an action."""
        raw = self._get("/state")
        self._current_raw_state = raw
        self._update_available(raw)
        return self._state_fn(raw)

    def get_raw_state(self):
        """Fetch the full raw observation dictionary."""
        return self._get("/raw_state")

    def get_action_names(self):
        """Return ordered list of all action names."""
        return list(self.ACTION_NAMES)

    def get_available_actions(self) -> frozenset:
        """
        Return the frozenset of action indices currently available.

        Use this to mask invalid actions in your agent:
            available = env.get_available_actions()
            if action not in available:
                action = random.choice(list(available))
        """
        return self._available_actions

    def is_action_available(self, action: int) -> bool:
        """Check if a specific action index is currently available. O(1)."""
        return action in self._available_actions

    # ── Default MDP Functions (students override these) ───────────────────

    @staticmethod
    def _default_state(raw: dict) -> tuple:
        """Minimal state discretization: (grid_x, grid_z, health_bin)."""
        return (
            raw.get("grid_x", 0),
            raw.get("grid_z", 0),
            raw.get("health_bin", 3),
        )

    @staticmethod
    def _default_reward(old_state, action, new_state) -> float:
        """Default: -1 per step to encourage efficiency."""
        return -1.0

    @staticmethod
    def _default_terminal(state, step_count) -> bool:
        """Default: never terminates early (only max_steps truncation)."""
        return False

    # ── Internal ─────────────────────────────────────────────────────────

    def _update_available(self, raw):
        """Parse available_actions from raw state into a frozenset.

        `raw` may be None when the bridge returns a null state (transient
        mineflayer failure — /action 200 with success:false and state:null).
        In that case, keep the last-known action set rather than crashing —
        the next /state or /action call will refresh it.
        """
        if raw is None:
            return
        available = raw.get("available_actions")
        if available is not None:
            self._available_actions = frozenset(available)

    @property
    def _headers(self) -> dict:
        if self._api_key:
            return {"X-API-Key": self._api_key}
        return {}

    def _request(self, method: str, path: str, json=None) -> dict:
        """Shared request with automatic 429 back-off. One rate-limit hit should
        not abort a 500-step episode — the bridge returns a Retry-After header
        telling us when the token bucket will refill. We honour it and retry
        transparently up to 5 times (covers ~5–15s of sustained limiting)."""
        url = f"{self._server_url}{path}"
        last_err = None
        for attempt in range(5):
            try:
                resp = requests.request(
                    method, url, json=json, headers=self._headers,
                    timeout=(10 if method == "GET" else 30), verify=False,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", "1"))
                    time.sleep(max(0.5, retry_after))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                # Network blips: short back-off + retry. Auth errors (401/403)
                # are terminal — stop retrying immediately.
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403):
                    break
                time.sleep(1.0)
        raise ConnectionError(
            f"Cannot reach server at {url} after retries. Check connection and API key.\n{last_err}"
        )

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, json=None) -> dict:
        return self._request("POST", path, json=json)
