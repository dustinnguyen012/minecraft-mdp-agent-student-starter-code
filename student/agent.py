"""
student/agent.py — Provided infrastructure
===========================================

This file contains the TransitionMatrix class, which handles count-based
estimation of T̂(s,a,s') from observed transitions. It is **provided — do
not modify**.

The actual MDP algorithms (Policy Iteration and Value Iteration) live in
student/mdp_agent.py. That is where your TODO blocks are.
"""


class TransitionMatrix:
    """
    Maintains an estimate of T(s, a, s') = P(s' | s, a) from experience.

    Starts from an optional prior (e.g., "moves usually succeed") and
    updates with every observed transition using count-based estimation:

        T̂(s, a, s') = count(s, a, s') / count(s, a)

    For (s, a) pairs never visited, falls back to the prior function. If
    no prior is supplied, falls back to a self-loop (nothing happens).

    API used by mdp_agent.py:
      - record(s, a, s')            — observe one transition
      - get_transitions(s, a)       — list of (prob, next_state)
      - observed_states             — set of all states seen
      - num_transitions             — total transition count
    """

    def __init__(self, prior_fn=None):
        self._prior_fn = prior_fn
        self._counts = {}      # {state: {action: {next_state: count}}}
        self._sa_totals = {}   # {state: {action: total_count}}
        self._observed_states = set()

    def record(self, state, action, next_state):
        """Record one observed (s, a, s') transition."""
        if state not in self._counts:
            self._counts[state] = {}
        if action not in self._counts[state]:
            self._counts[state][action] = {}
        self._counts[state][action][next_state] = (
            self._counts[state][action].get(next_state, 0) + 1
        )
        if state not in self._sa_totals:
            self._sa_totals[state] = {}
        self._sa_totals[state][action] = (
            self._sa_totals[state].get(action, 0) + 1
        )
        self._observed_states.add(state)
        self._observed_states.add(next_state)

    def get_transitions(self, state, action):
        """
        Return T(s, a) as a list of (probability, next_state) tuples.

        Uses empirical counts if available; otherwise falls back to the
        prior; otherwise returns a self-loop.
        """
        total = self._sa_totals.get(state, {}).get(action, 0)
        if total > 0:
            return [(count / total, s_prime)
                    for s_prime, count in self._counts.get(state, {}).get(action, {}).items()]
        elif self._prior_fn:
            return self._prior_fn(state, action)
        else:
            return [(1.0, state)]  # self-loop fallback

    @property
    def observed_states(self):
        return self._observed_states

    @property
    def num_transitions(self):
        return sum(t for sa in self._sa_totals.values() for t in sa.values())
