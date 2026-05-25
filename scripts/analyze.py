#!/usr/bin/env python3
"""
scripts/analyze.py — Generate a training report for your MDP bot.
=================================================================

Reads the pickled T̂ snapshots your `mdp_agent.py` saves to `results/` plus the
bot's stdout log (/tmp/bot*.log by default), then produces:

  1. A terse text summary printed to stdout (action counts, rates, exploration).
  2. A handful of Plotly HTML charts in `results/analysis/` you can open in a
     browser to see trends over training.

Run it any time during or after a training session. Your bots don't need to
be stopped — it reads persisted artefacts only, never touches the bridge.

Usage (from the starter root):
    python3 scripts/analyze.py --bot-name BrightAsh_ql --log /tmp/bot0.log

Arguments
---------
--results-dir   PATH  (default: ./results)
--bot-name      NAME  (auto-detected if omitted; picks whichever bot has the
                       most pkl files in results-dir)
--log           PATH  (optional; defaults to /tmp/bot*.log matching BOT_ID if
                       it exists. Parsed for per-episode reward lines.)
--top-actions   N     (default: 15; how many most-used actions to surface)
--out           PATH  (default: <results-dir>/analysis)

Requirements
------------
    pip install plotly
"""

import argparse
import glob
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("This script needs plotly.  Install with:  pip install plotly")
    sys.exit(1)


# ── pkl series ──────────────────────────────────────────────────────────────

def load_pkl_series(results_dir: Path, bot_name: str):
    """Load every saved pkl for a bot, sorted ascending by episode number."""
    files = list(results_dir.glob(f"{bot_name}_ep*.pkl"))
    series = []
    for f in files:
        if f.stat().st_size == 0:
            continue
        m = re.search(r"ep(\d+)\.pkl$", f.name)
        if not m:
            continue
        try:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            data["_episode"] = int(m.group(1))
            data["_file"] = f.name
            series.append(data)
        except Exception:
            pass
    return sorted(series, key=lambda d: d["_episode"])


def autodetect_bot_name(results_dir: Path) -> str:
    """Pick whichever bot has the most pkl files."""
    counts = Counter()
    for f in results_dir.glob("*_ep*.pkl"):
        m = re.match(r"(.+?)_ep\d+\.pkl$", f.name)
        if m:
            counts[m.group(1)] += 1
    if not counts:
        sys.exit(f"No pkl files found in {results_dir}. Did your agent save any?")
    return counts.most_common(1)[0][0]


# ── log parsing ─────────────────────────────────────────────────────────────

EP_LINE_RE = re.compile(
    r"Ep\s+(?P<ep>\d+)\s+\|\s+R:\s+(?P<r>-?\d+(?:\.\d+)?)\s+\|"
    r"\s+Avg\(\d+\):\s+(?P<avg>-?\d+(?:\.\d+)?)\s+\|"
    r"\s+ε:\s+(?P<eps>[\d.]+)\s+\|"
    r"\s+States:\s+(?P<states>\d+)\s+\|"
    r"\s+Transitions:\s+(?P<tr>\d+)"
)

STALL_RE = re.compile(r"Stalled.*ε:\s+[\d.]+\s+→\s+([\d.]+)")


def parse_log(log_path: Path):
    """Extract per-episode rows + stall events from the bot's stdout log."""
    episodes = []
    stalls = 0
    if not log_path.exists():
        return episodes, stalls
    with open(log_path) as f:
        for line in f:
            m = EP_LINE_RE.search(line)
            if m:
                episodes.append({
                    "episode": int(m.group("ep")),
                    "reward": float(m.group("r")),
                    "avg10": float(m.group("avg")),
                    "epsilon": float(m.group("eps")),
                    "states": int(m.group("states")),
                    "transitions": int(m.group("tr")),
                })
            elif STALL_RE.search(line):
                stalls += 1
    return episodes, stalls


# ── derived metrics ─────────────────────────────────────────────────────────

def action_totals(pkl, action_names):
    """From T̂._sa_totals (or T_totals key in dict form), count total uses per action."""
    totals_dict = pkl.get("T_totals", {})
    per_action = Counter()
    for state, a_map in totals_dict.items():
        for a, count in a_map.items():
            per_action[a] += count
    # Turn indices into names for readability.
    return [(action_names[a] if a < len(action_names) else f"act{a}", n)
            for a, n in per_action.most_common()]


def exploration_metrics(final_pkl, episodes):
    """Compact stats block for the text report."""
    num_states = len(final_pkl.get("T_counts", {})) if final_pkl else 0
    num_trans = sum(sum(sp.values()) for a_map in final_pkl.get("T_counts", {}).values()
                    for sp in a_map.values()) if final_pkl else 0
    if episodes:
        ep_span = episodes[-1]["episode"] - episodes[0]["episode"]
        recent_avg = sum(e["reward"] for e in episodes[-10:]) / min(10, len(episodes))
        best = max(e["reward"] for e in episodes)
        worst = min(e["reward"] for e in episodes)
    else:
        ep_span = 0
        recent_avg = best = worst = float("nan")
    return {
        "num_states_observed": num_states,
        "num_transitions": num_trans,
        "avg_transitions_per_state": (num_trans / num_states) if num_states else 0,
        "episodes_logged": len(episodes),
        "episode_span": ep_span,
        "recent_avg_reward": recent_avg,
        "best_episode_reward": best,
        "worst_episode_reward": worst,
    }


# ── plots ───────────────────────────────────────────────────────────────────

def plot_rewards(episodes, out_path, bot_name):
    if not episodes:
        return False
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Per-episode reward", "ε decay"),
                        vertical_spacing=0.12)
    x = [e["episode"] for e in episodes]
    fig.add_trace(go.Scatter(x=x, y=[e["reward"] for e in episodes],
                             mode="lines", name="reward",
                             line=dict(color="#9CA3AF", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[e["avg10"] for e in episodes],
                             mode="lines", name="avg(10)",
                             line=dict(color="#2563EB", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[e["epsilon"] for e in episodes],
                             mode="lines", name="ε",
                             line=dict(color="#DC2626", width=2)), row=2, col=1)
    fig.update_layout(height=600, title=f"{bot_name} — reward & exploration",
                      hovermode="x unified")
    fig.write_html(out_path)
    return True


def plot_growth(episodes, out_path, bot_name):
    if not episodes:
        return False
    fig = make_subplots(rows=1, cols=2, subplot_titles=("State count", "Transition count"))
    x = [e["episode"] for e in episodes]
    fig.add_trace(go.Scatter(x=x, y=[e["states"] for e in episodes],
                             mode="lines+markers", line=dict(color="#059669")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=[e["transitions"] for e in episodes],
                             mode="lines+markers", line=dict(color="#7C3AED")), row=1, col=2)
    fig.update_layout(height=400, title=f"{bot_name} — T̂ growth", showlegend=False)
    fig.write_html(out_path)
    return True


def plot_action_distribution(totals, out_path, bot_name, top_n):
    if not totals:
        return False
    labels = [t[0] for t in totals[:top_n]]
    values = [t[1] for t in totals[:top_n]]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                           marker_color="#2563EB"))
    fig.update_layout(height=max(350, 25 * len(labels)),
                      title=f"{bot_name} — top {len(labels)} actions (cumulative count in T̂)",
                      xaxis_title="times recorded", yaxis=dict(autorange="reversed"))
    fig.write_html(out_path)
    return True


def plot_action_timeline(series, action_names, out_path, bot_name, top_n):
    """Stacked-area chart of top action usage as episodes progress."""
    if not series:
        return False
    top_action_ids = Counter()
    for pkl in series:
        for s, a_map in (pkl.get("T_totals") or {}).items():
            for a, n in a_map.items():
                top_action_ids[a] += n
    top_ids = [a for a, _ in top_action_ids.most_common(top_n)]

    # Per-snapshot counts for each top action.
    per_snapshot = []
    for pkl in series:
        row = {a: 0 for a in top_ids}
        for s, a_map in (pkl.get("T_totals") or {}).items():
            for a, n in a_map.items():
                if a in row:
                    row[a] = row[a] + n
        per_snapshot.append(row)

    fig = go.Figure()
    x = [pkl["_episode"] for pkl in series]
    for a in top_ids:
        name = action_names[a] if a < len(action_names) else f"act{a}"
        fig.add_trace(go.Scatter(x=x, y=[row[a] for row in per_snapshot],
                                 mode="lines", stackgroup="one", name=name))
    fig.update_layout(height=500, title=f"{bot_name} — cumulative action usage over training",
                      xaxis_title="episode", yaxis_title="cumulative T̂ count",
                      hovermode="x unified")
    fig.write_html(out_path)
    return True


# ── action-name lookup ──────────────────────────────────────────────────────

# Snapshot of the bridge's ACTION_LIST. Used as an offline fallback when
# MDP_API_KEY / MDP_SERVER_URL aren't set or the bridge isn't reachable.
# Keep in sync with bridge/server.js ACTION_LIST. If the bridge is reachable,
# the fetched list takes precedence (so this snapshot being slightly stale
# won't matter during normal runs).
_HARDCODED_ACTION_NAMES = [
    "move_north", "move_south", "move_east", "move_west",
    "mine_below", "mine_forward", "place_forward", "noop",
    "craft_planks", "craft_sticks", "craft_crafting_table", "craft_wooden_pickaxe",
    "craft_stone_pickaxe", "craft_wooden_sword", "attack_nearest", "eat",
    "feed_animal", "get_wood", "chop_wood", "craft_wooden_axe",
    "craft_wooden_shovel", "craft_furnace", "craft_stone_sword", "mine_coal",
    "mine_iron", "mine_copper", "mine_gold", "mine_redstone",
    "mine_lapis", "mine_diamond", "mine_emerald", "farm_nearest",
    "craft_torch", "place_torch", "equip_armor", "sleep",
    "smelt_start", "smelt_collect", "fish", "shear_sheep",
    "milk_cow", "toggle_door", "drop_cobblestone", "drop_gravel",
    "drop_dirt", "shoot_arrow", "raise_shield", "drink_potion",
    "mount", "dismount", "climb_up", "hunt_cow",
    "hunt_pig", "hunt_chicken", "hunt_sheep", "hunt_rabbit",
    "attack_villager", "attack_player", "pick_up_meat", "pick_up_ore",
    "pick_up_material", "place_crafting_table", "place_furnace", "place_cobblestone",
    "place_stone", "place_plank", "place_log", "place_redstone",
    "place_dirt", "dig_forward", "mine_stone", "craft_wooden_hoe",
    "craft_stone_hoe", "till_soil", "plant_seed", "craft_stone_axe",
    "craft_stone_shovel", "craft_iron_pickaxe", "craft_iron_axe", "craft_iron_sword",
    "craft_iron_shovel", "craft_iron_hoe", "craft_diamond_pickaxe", "craft_diamond_axe",
    "craft_diamond_sword", "craft_diamond_shovel", "craft_diamond_hoe", "craft_bread",
    "craft_chest", "craft_bed", "craft_shield", "craft_bow",
    "craft_flint_and_steel", "craft_bucket", "pick_up_planks", "hit_vegetation",
    "break_nodrop", "place_chest", "cook_start", "cook_collect",
    "smelt_iron_start", "smelt_iron_collect", "peek_furnace", "close_furnace",
    "craft_smoker", "place_smoker", "smoke_wood", "smelt_cobblestone",
    "smelt_stone", "craft_blast_furnace", "place_blast_furnace",
    # Phase 2 chest IO: 50 actions (bulk/food/tool matrix), 111-160
    "deposit_cobblestone", "withdraw_cobblestone",
    "deposit_gravel", "withdraw_gravel",
    "deposit_dirt", "withdraw_dirt",
    "deposit_food_raw", "withdraw_food_raw",
    "deposit_food_cooked", "withdraw_food_cooked",
    "deposit_wooden_pickaxe", "withdraw_wooden_pickaxe",
    "deposit_wooden_sword", "withdraw_wooden_sword",
    "deposit_wooden_axe", "withdraw_wooden_axe",
    "deposit_wooden_shovel", "withdraw_wooden_shovel",
    "deposit_wooden_hoe", "withdraw_wooden_hoe",
    "deposit_stone_pickaxe", "withdraw_stone_pickaxe",
    "deposit_stone_sword", "withdraw_stone_sword",
    "deposit_stone_axe", "withdraw_stone_axe",
    "deposit_stone_shovel", "withdraw_stone_shovel",
    "deposit_stone_hoe", "withdraw_stone_hoe",
    "deposit_iron_pickaxe", "withdraw_iron_pickaxe",
    "deposit_iron_sword", "withdraw_iron_sword",
    "deposit_iron_axe", "withdraw_iron_axe",
    "deposit_iron_shovel", "withdraw_iron_shovel",
    "deposit_iron_hoe", "withdraw_iron_hoe",
    "deposit_diamond_pickaxe", "withdraw_diamond_pickaxe",
    "deposit_diamond_sword", "withdraw_diamond_sword",
    "deposit_diamond_axe", "withdraw_diamond_axe",
    "deposit_diamond_shovel", "withdraw_diamond_shovel",
    "deposit_diamond_hoe", "withdraw_diamond_hoe",
]


def load_action_names():
    """Fetch action list from the bridge if reachable; fall back to the
    HARDCODED snapshot above if the bridge is unreachable or no API key set.
    This lets analyze.py run offline without losing readable action names."""
    key = os.environ.get("MDP_API_KEY")
    url = os.environ.get("MDP_SERVER_URL", "https://localhost")
    if key:
        try:
            import urllib.request, ssl, json
            ctx = ssl._create_unverified_context()
            req = urllib.request.Request(f"{url}/actions", headers={"X-API-Key": key})
            resp = urllib.request.urlopen(req, timeout=5, context=ctx)
            names = list(json.loads(resp.read())["actions"])
            if names:
                return names
        except Exception:
            pass
    # Offline / unreachable — use hardcoded snapshot.
    return list(_HARDCODED_ACTION_NAMES)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results-dir", type=Path,
                    default=Path(__file__).parent.parent / "results")
    ap.add_argument("--bot-name", default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--top-actions", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        sys.exit(f"results dir not found: {results_dir}")

    bot = args.bot_name or autodetect_bot_name(results_dir)
    out = args.out or (results_dir / "analysis")
    out.mkdir(exist_ok=True, parents=True)

    # Resolve log.
    log = args.log
    if log is None:
        bot_id = os.environ.get("BOT_ID", "0")
        candidate = Path(f"/tmp/bot{bot_id}.log")
        log = candidate if candidate.exists() else Path("/dev/null")

    # Load data.
    series = load_pkl_series(results_dir, bot)
    episodes, stall_count = parse_log(log)
    action_names = load_action_names()
    final = series[-1] if series else None
    totals = action_totals(final, action_names) if final else []

    # Text report.
    print(f"\n╒══════════════════════════════════════════════════════════════")
    print(f"│ MDP training report — {bot}")
    print(f"╘══════════════════════════════════════════════════════════════\n")
    print(f"Results directory : {results_dir}")
    print(f"Log file          : {log if log.exists() else '(not found)'}")
    print(f"Pkl snapshots     : {len(series)}")
    if series:
        print(f"Episode range     : {series[0]['_episode']} → {series[-1]['_episode']}")
    print()

    metrics = exploration_metrics(final, episodes)
    print(f"  Observed states         : {metrics['num_states_observed']}")
    print(f"  Recorded transitions    : {metrics['num_transitions']}")
    print(f"  Avg transitions / state : {metrics['avg_transitions_per_state']:.1f}")
    print(f"  Episodes in log         : {metrics['episodes_logged']}")
    print(f"  Recent avg(10) reward   : {metrics['recent_avg_reward']:.1f}")
    print(f"  Best episode reward     : {metrics['best_episode_reward']:.1f}")
    print(f"  Worst episode reward    : {metrics['worst_episode_reward']:.1f}")
    print(f"  Stall ε-bumps observed  : {stall_count}")
    print()

    if totals:
        print(f"Top {args.top_actions} actions by cumulative count in T̂:")
        total_all = sum(n for _, n in totals)
        for name, n in totals[:args.top_actions]:
            pct = 100.0 * n / total_all if total_all else 0.0
            print(f"  {name:26s} {n:>8d}   {pct:5.1f}%")
        print(f"  {'(other)':26s} {total_all - sum(n for _, n in totals[:args.top_actions]):>8d}")
        print()

    # Plots.
    if plot_rewards(episodes, out / f"{bot}_rewards.html", bot):
        print(f"  wrote {out / f'{bot}_rewards.html'}")
    if plot_growth(episodes, out / f"{bot}_growth.html", bot):
        print(f"  wrote {out / f'{bot}_growth.html'}")
    if plot_action_distribution(totals, out / f"{bot}_actions.html", bot, args.top_actions):
        print(f"  wrote {out / f'{bot}_actions.html'}")
    if plot_action_timeline(series, action_names, out / f"{bot}_action_timeline.html",
                            bot, args.top_actions):
        print(f"  wrote {out / f'{bot}_action_timeline.html'}")
    print()


if __name__ == "__main__":
    main()
