"""
student/mdp_definition.py
--------------------------
MDP definition for Minecraft tech-tree progression agent.
Goal: bare hands → wooden pickaxe → stone pickaxe → iron pickaxe.
"""

BUCKET_SIZE = 5    # blocks per position bucket
MAX_STEPS   = 600  # episode timeout

def state_fn(raw: dict) -> tuple:
    """
    Convert raw observation into a hashable 14-element state tuple.
    Uses pre-computed boolean flags from the bridge.
    """
    gx = (raw.get("grid_x") or 0) // BUCKET_SIZE
    gz = (raw.get("grid_z") or 0) // BUCKET_SIZE

    return (
        gx,
        gz,
        int(raw.get("health_bin", 3)),
        bool(raw.get("has_wood",           False)),
        bool(raw.get("has_planks",         False)),
        bool(raw.get("has_sticks",         False)),
        bool(raw.get("has_stone",          False)),
        bool(raw.get("has_wood_tools",     False)),
        bool(raw.get("has_stone_tools",    False)),
        bool(raw.get("has_furnace",        False)),
        bool(raw.get("has_table_nearby",   False)),
        bool(raw.get("has_furnace_nearby", False)),
        0 if raw.get("time_of_day", "day") == "day" else 1,
        bool(raw.get("has_food",           False)),
    )


def reward_fn(old_state: tuple, action: int, new_state: tuple) -> float:
    """
    Reward fires only on true state-bit transitions so it never triggers
    when an action silently fails and the state stays the same.
    """
    (_, _, o_hp, o_wood, o_planks, o_sticks, o_stone,
     o_wt, o_st, o_furn, o_table, o_furn_near, o_tod, o_food) = old_state

    (_, _, n_hp, n_wood, n_planks, n_sticks, n_stone,
     n_wt, n_st, n_furn, n_table, n_furn_near, n_tod, n_food) = new_state

    r = -0.05  # small per-step penalty

    # Tech-tree milestones
    if n_wood   and not o_wood:    r += 1.0
    if n_planks and not o_planks:  r += 1.0
    if n_sticks and not o_sticks:  r += 0.5
    if n_stone  and not o_stone:   r += 2.0
    if n_table  and not o_table:   r += 1.5
    if n_wt     and not o_wt:      r += 8.0
    if n_st     and not o_st:      r += 15.0
    if n_furn   and not o_furn:    r += 5.0

    # Iron pickaxe goal — action 77 = craft_iron_pickaxe
    if action == 77 and o_st and not n_st:
        r += 30.0

    # Survival
    if n_hp < o_hp and n_hp <= 1:  r -= 3.0
    if n_tod == 1 and o_tod == 0:  r -= 0.5

    return r


def terminal_fn(state: tuple, step_count: int) -> bool:
    """End episode on timeout only.
    Death check removed — health_bin=0 on reset caused immediate termination.
    """
    if step_count >= MAX_STEPS:
        return True
    return False


def prior_transitions(state: tuple, action: int) -> list:
    """
    Self-loop prior: assume nothing changed until experience says otherwise.
    Used by TransitionMatrix as fallback for unseen (state, action) pairs.
    """
    return [(1.0, state)]