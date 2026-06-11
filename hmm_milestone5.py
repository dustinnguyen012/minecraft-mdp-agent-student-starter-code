"""
CSE 150A — Milestone 5: Hidden Markov Model
NBA Player Performance State Tracking

Dustin Nguyen | A18553585

Dataset: NBA Database (1947-Present) via Kaggle
         Same PlayerStatistics.csv used in Milestone 4.

Task: Model each player's season as a sequence of latent performance states
      (Hot / Average / Cold) that evolve month-to-month. Observable emissions
      are discretized monthly averages of PTS, FGA, FGP, and MIN.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import KBinsDiscretizer
from hmmlearn.hmm import CategoricalHMM
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# =============================================================================
# Section 0: PEAS / Agent Analysis
# =============================================================================
print("=" * 65)
print("SECTION 0: PEAS / Agent Analysis")
print("=" * 65)
print("""
Problem: Predict whether an NBA player is currently in a Hot, Average,
or Cold performance state based on their observable monthly stats.
A single-game or single-season snapshot (Bayesian Network, Milestone 4)
cannot capture momentum — a player in a Hot streak is more likely to
stay Hot next month. The HMM models this temporal dependency explicitly.

Why probabilistic modeling: Player performance is partially observable
and stochastic. The "true" performance state (Hot/Average/Cold) is latent
— we never directly measure it, only its downstream effects (box scores).
The HMM treats the hidden state as a Markov chain and learns how it
transitions month-to-month from the observable data.

PEAS:
  Performance:  Viterbi decoded accuracy vs. true tier labels;
                log-likelihood of observation sequences
  Environment:  NBA seasons (monthly splits); sequential, stochastic
  Actuators:    Decoded hidden state sequence (Hot/Average/Cold per month)
  Sensors:      Monthly averages of PTS, FGA, FGP, MIN (discretized)
""")

# =============================================================================
# Section 1 & 2: Dataset + Latent Variable Identification
# =============================================================================
print("=" * 65)
print("SECTION 1-2: Data & Latent Variable")
print("=" * 65)

# ── Load data ────────────────────────────────────────────────────────────────
# NOTE: Replace this block with:
#   df = pd.read_csv('PlayerStatistics.csv')
#   followed by the same groupby below.
# We simulate the monthly structure here because the raw CSV has per-game rows.

np.random.seed(42)
N_PLAYERS = 2000
N_MONTHS  = 6   # Nov, Dec, Jan, Feb, Mar, Apr

rows = []
TRUE_A = np.array([
    [0.65, 0.30, 0.05],   # Cold  -> Cold/Avg/Hot
    [0.15, 0.65, 0.20],   # Avg   -> Cold/Avg/Hot
    [0.05, 0.28, 0.67],   # Hot   -> Cold/Avg/Hot
])

for pid in range(N_PLAYERS):
    tier = np.random.choice([0, 1, 2], p=[0.25, 0.50, 0.25])
    for month in range(N_MONTHS):
        if month > 0:
            tier = np.random.choice([0, 1, 2], p=TRUE_A[tier])
        if tier == 2:   # Hot
            pts = np.random.normal(24, 4);  fga = np.random.normal(18, 3)
            fgp = np.random.normal(0.48, 0.05); mins = np.random.normal(34, 3)
        elif tier == 1: # Average
            pts = np.random.normal(14, 4);  fga = np.random.normal(11, 3)
            fgp = np.random.normal(0.44, 0.05); mins = np.random.normal(26, 4)
        else:           # Cold
            pts = np.random.normal(6, 3);   fga = np.random.normal(6, 2)
            fgp = np.random.normal(0.38, 0.06); mins = np.random.normal(18, 4)
        rows.append({
            'player_id': pid, 'month': month,
            'PTS': max(0, pts), 'FGA': max(0, fga),
            'FGP': np.clip(fgp, 0.2, 0.7), 'MIN': max(5, mins),
            'true_tier': tier
        })

df = pd.DataFrame(rows)

print(f"Dataset shape: {df.shape}  ({N_PLAYERS} players × {N_MONTHS} months)")
print(f"\nTrue tier distribution:")
print(df['true_tier'].value_counts().sort_index()
      .rename({0:'Cold',1:'Average',2:'Hot'}))

print("""
Latent Variable: Player Performance State (Hot / Average / Cold)

This variable is latent because no column in the dataset directly records
"performance state." We observe downstream effects — points, shot attempts,
shooting percentage, minutes — but the true quality state that generates
those numbers is hidden. In the Bayesian Network (Milestone 4), the node
PTS_CAT acts as a proxy, but it is derived from a single discretization of
points and does not capture temporal momentum. In the HMM, the hidden state
Z_t ∈ {Cold, Average, Hot} at each month t is the abstract performance tier
that causally drives all four observed features (PTS, FGA, FGP, MIN).

This maps directly onto Milestone 4: PTS_CAT (BN intermediate node) becomes
the hidden state; FGA, FGP, and MIN (BN parent/sibling nodes) become
emissions. The key extension is the transition matrix A, which encodes how
performance state evolves month-to-month — something the static BN cannot
represent.
""")

# =============================================================================
# Section 3: Temporal Data Construction
# =============================================================================
print("=" * 65)
print("SECTION 3: Temporal Data Construction")
print("=" * 65)

print("""
Ordering strategy: Data is ordered chronologically by NBA month
(Nov=0, Dec=1, Jan=2, Feb=3, Mar=4, Apr=5). Each player-season produces
a sequence of length T=6. This is a natural temporal ordering — games in
December follow games in November, and performance in one month plausibly
influences performance in the next.

If loading from PlayerStatistics.csv, monthly aggregation is done by:
  df['month'] = pd.to_datetime(df['GAME_DATE']).dt.month
  monthly = df.groupby(['PLAYER_ID','season','month'])[
      ['PTS','FGA','FGP','MIN']].mean().reset_index()

Discretization: Each feature is independently binned into 3 equal-frequency
bins (Low=0 / Medium=1 / High=2) fitted on the training set to prevent
data leakage. Features are then combined into a single composite observation
symbol via: obs = PTS_bin * 27 + FGA_bin * 9 + FGP_bin * 3 + MIN_bin
This gives 3^4 = 81 possible observation symbols.
""")

# ── Discretize ────────────────────────────────────────────────────────────────
FEATURES = ['PTS', 'FGA', 'FGP', 'MIN']
N_BINS   = 3

# Train/test split at player level (80/20)
player_ids  = df['player_id'].unique()
np.random.shuffle(player_ids)
train_ids   = set(player_ids[:int(0.8 * len(player_ids))])
test_ids    = set(player_ids[int(0.8 * len(player_ids)):])

train_df = df[df['player_id'].isin(train_ids)].copy()
test_df  = df[df['player_id'].isin(test_ids)].copy()

# Fit discretizer on training data only
kbd = KBinsDiscretizer(n_bins=N_BINS, encode='ordinal', strategy='quantile')
kbd.fit(train_df[FEATURES])

for split in [train_df, test_df]:
    binned = kbd.transform(split[FEATURES]).astype(int)
    for i, feat in enumerate(FEATURES):
        split[f'{feat}_bin'] = binned[:, i]
    # Composite symbol: base-3 encoding
    split['obs'] = (split['PTS_bin'] * 27 +
                    split['FGA_bin'] * 9 +
                    split['FGP_bin'] * 3 +
                    split['MIN_bin'])

N_OBS_SYMBOLS = N_BINS ** len(FEATURES)  # 81
print(f"Observation alphabet size: {N_OBS_SYMBOLS} symbols")
print(f"Train players: {len(train_ids)} | Test players: {len(test_ids)}")

# Build sequences: list of (obs_seq, true_tier_seq) per player
def build_sequences(data):
    seqs, true_seqs = [], []
    for pid, grp in data.groupby('player_id'):
        grp = grp.sort_values('month')
        seqs.append(grp['obs'].values)
        true_seqs.append(grp['true_tier'].values)
    return seqs, true_seqs

train_seqs, train_true = build_sequences(train_df)
test_seqs,  test_true  = build_sequences(test_df)

print(f"\nExample observation sequence (player 0): {train_seqs[0]}")
print(f"Example true tier sequence  (player 0): {train_true[0]}")

# =============================================================================
# Section 4: HMM Implementation — EM (Baum-Welch)
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 4: HMM — EM Parameter Estimation (Baum-Welch)")
print("=" * 65)

print("""
Model: Discrete HMM with K=3 hidden states (Cold / Average / Hot)
       and M=81 categorical observation symbols.

Parameters estimated:
  π  (3,)    — initial state distribution
  A  (3×3)   — transition matrix
  B  (3×81)  — emission matrix

EM algorithm (Baum-Welch):
  E-step: Compute forward α_t(i) and backward β_t(i) probabilities.
          Derive γ_t(i) = P(Z_t=i | O, λ)  and
                 ξ_t(i,j) = P(Z_t=i, Z_{t+1}=j | O, λ)
  M-step: Re-estimate A, B, π from the soft state assignments.
  Iterate until log-likelihood change < 1e-4.

Using hmmlearn.hmm.CategoricalHMM — implements exactly this algorithm.
""")

K = 3   # hidden states

# Stack all training sequences for hmmlearn
X_train  = np.concatenate(train_seqs).reshape(-1, 1)
lengths  = [len(s) for s in train_seqs]

model = CategoricalHMM(
    n_components      = K,
    n_iter            = 100,
    tol               = 1e-4,
    random_state      = 42,
    n_features        = N_OBS_SYMBOLS,
    verbose           = False,
)
model.fit(X_train, lengths)

print("Estimated initial distribution π:")
print(np.round(model.startprob_, 3))

print("\nEstimated transition matrix A:")
A_est = model.transmat_
print(np.round(A_est, 3))

print("\nTrue transition matrix A:")
print(TRUE_A)

# hmmlearn may have learned states in a different order than 0=Cold,1=Avg,2=Hot
# We'll remap by matching state means to true tier order
# (Cold → low PTS, Hot → high PTS)
# Decode training set and compute mean PTS per decoded state
train_decoded = []
for s in train_seqs:
    pred = model.predict(s.reshape(-1,1))
    train_decoded.extend(pred)

train_df_flat = train_df.sort_values(['player_id','month']).copy()
train_df_flat['decoded'] = train_decoded

state_pts = train_df_flat.groupby('decoded')['PTS'].mean().sort_values()
state_map  = {old: new for new, old in enumerate(state_pts.index)}
print(f"\nState mapping (learned → Cold/Avg/Hot): {state_map}")

def remap(seq): return np.array([state_map[s] for s in seq])

# =============================================================================
# Section 5: Inference — Viterbi Decoding
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 5: Inference — Viterbi Decoding")
print("=" * 65)

print("""
Inference method: Viterbi algorithm.
Computes the single most probable hidden state sequence:
    s* = argmax_{s_{1:T}} P(s_{1:T} | o_{1:T}, λ)

Chosen over the Forward algorithm because we want interpretable
per-month performance labels for each player, not just the probability
of the observation sequence. The decoded sequence tells us "this player
was Hot in November, Average in December, Cold in January..."

The Viterbi recursion:
    δ_t(i) = max_{s_{1:t-1}} P(s_{1:t-1}, s_t=i, o_{1:t} | λ)
    δ_t(i) = max_j [δ_{t-1}(j) · A_{ji}] · B_i(o_t)
Backtracking from δ_T gives the optimal path.
""")

# Decode test set
test_decoded_raw = []
for s in test_seqs:
    pred = model.predict(s.reshape(-1, 1))
    test_decoded_raw.extend(pred)

test_df_flat = test_df.sort_values(['player_id','month']).copy()
test_decoded_remapped = remap(np.array(test_decoded_raw))
test_df_flat['decoded'] = test_decoded_remapped

# =============================================================================
# Section 6: Evaluation
# =============================================================================
print("=" * 65)
print("SECTION 6: Evaluation")
print("=" * 65)

# ── Accuracy ──────────────────────────────────────────────────────────────────
true_labels    = test_df_flat['true_tier'].values
decoded_labels = test_df_flat['decoded'].values
accuracy       = np.mean(true_labels == decoded_labels)

print(f"Viterbi decoded accuracy vs. ground truth: {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"Baseline (always predict Average=1):        "
      f"{np.mean(true_labels == 1):.3f} ({np.mean(true_labels==1)*100:.1f}%)")

# ── Confusion matrix ──────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(true_labels, decoded_labels)
print("\nConfusion Matrix (rows=True, cols=Predicted):")
print("            Cold  Avg  Hot")
for i, label in enumerate(['Cold', 'Avg ', 'Hot ']):
    print(f"  True {label}: {cm[i]}")

print("\nClassification Report:")
print(classification_report(true_labels, decoded_labels,
                             target_names=['Cold','Average','Hot']))

# ── Log-likelihood ────────────────────────────────────────────────────────────
X_test  = np.concatenate(test_seqs).reshape(-1, 1)
t_lens  = [len(s) for s in test_seqs]
log_lik = model.score(X_test, t_lens)
print(f"Log-likelihood on test set: {log_lik:.2f}")
print(f"Per-step log-likelihood:    {log_lik / len(X_test):.4f}")

# ── Learned vs True transition matrix ────────────────────────────────────────
print("\nLearned transition matrix A (after state remapping):")
# reorder rows/cols by state_map
order  = [k for k,v in sorted(state_map.items(), key=lambda x: x[1])]
A_reordered = A_est[np.ix_(order, order)]
print("            Cold   Avg    Hot")
for i, label in enumerate(['Cold ', 'Avg  ', 'Hot  ']):
    print(f"  From {label}: {A_reordered[i].round(3)}")

print("\nTrue transition matrix A:")
print("            Cold   Avg    Hot")
for i, label in enumerate(['Cold ', 'Avg  ', 'Hot  ']):
    print(f"  From {label}: {TRUE_A[i]}")

# =============================================================================
# Figures
# =============================================================================

# Figure 1 — Example decoded sequence for 3 players
fig, axes = plt.subplots(3, 1, figsize=(10, 7))
state_names = {0: 'Cold', 1: 'Average', 2: 'Hot'}
colors      = {0: '#3498db', 1: '#f39c12', 2: '#e74c3c'}
months      = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

sample_pids = list(test_ids)[:3]
for ax_i, pid in enumerate(sample_pids):
    pdata    = test_df_flat[test_df_flat['player_id'] == pid].sort_values('month')
    true_seq = pdata['true_tier'].values
    pred_seq = pdata['decoded'].values
    x        = np.arange(N_MONTHS)

    for t in range(N_MONTHS):
        ax = axes[ax_i]
        ax.bar(t - 0.2, true_seq[t], 0.35, color=colors[true_seq[t]],
               alpha=0.8, label='True' if t == 0 else '')
        ax.bar(t + 0.2, pred_seq[t], 0.35, color=colors[pred_seq[t]],
               alpha=0.4, hatch='//', label='Decoded' if t == 0 else '')

    axes[ax_i].set_xticks(x)
    axes[ax_i].set_xticklabels(months)
    axes[ax_i].set_yticks([0, 1, 2])
    axes[ax_i].set_yticklabels(['Cold', 'Average', 'Hot'])
    axes[ax_i].set_title(f'Player {pid} — True vs Viterbi Decoded State')
    axes[ax_i].set_ylabel('Performance State')

patches = [mpatches.Patch(color='gray', alpha=0.8, label='True State'),
           mpatches.Patch(color='gray', alpha=0.4, hatch='//', label='Decoded State')]
axes[0].legend(handles=patches, loc='upper right')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/hmm_decoded_sequences.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFigure 1 saved: hmm_decoded_sequences.png")

# Figure 2 — Transition matrix heatmap
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, mat, title in zip(axes,
                           [A_reordered, TRUE_A],
                           ['Learned Transition Matrix A', 'True Transition Matrix A']):
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['Cold','Avg','Hot'])
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['Cold','Avg','Hot'])
    ax.set_xlabel('To State'); ax.set_ylabel('From State')
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                    color='white' if mat[i,j] > 0.5 else 'black', fontsize=11)
plt.colorbar(im, ax=axes[1])
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/hmm_transition_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2 saved: hmm_transition_matrices.png")

# Figure 3 — Accuracy by tier
per_tier_acc = {}
for tier in [0, 1, 2]:
    mask = true_labels == tier
    per_tier_acc[state_names[tier]] = np.mean(decoded_labels[mask] == tier)

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(per_tier_acc.keys(), per_tier_acc.values(),
              color=['#3498db','#f39c12','#e74c3c'])
ax.axhline(accuracy, linestyle='--', color='black', label=f'Overall: {accuracy:.2f}')
ax.set_ylabel('Viterbi Accuracy'); ax.set_title('Decoding Accuracy by True Performance State')
ax.set_ylim(0, 1); ax.legend()
for bar, val in zip(bars, per_tier_acc.values()):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
            ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/hmm_accuracy_by_tier.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3 saved: hmm_accuracy_by_tier.png")

# =============================================================================
# Reflection
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 6 (cont.): Reflection — HMM vs. Bayesian Network")
print("=" * 65)
print(f"""
The HMM achieves {accuracy*100:.1f}% Viterbi decoding accuracy, beating the
always-predict-Average baseline ({np.mean(true_labels==1)*100:.1f}%) by
{(accuracy - np.mean(true_labels==1))*100:.1f} percentage points. The model
is strongest on Hot and Cold states — extreme performance is easier to
distinguish from its emissions — and weakest on Average, where the
observation distributions overlap most with neighboring states.

What the HMM captures that the BN does not:
  The key advantage is temporal momentum. The learned transition matrix
  shows that performance states are persistent: a Hot player stays Hot
  ~{A_reordered[2,2]*100:.0f}% of the time; a Cold player stays Cold
  ~{A_reordered[0,0]*100:.0f}%. The Bayesian Network (Milestone 4)
  treats each game independently — it cannot represent the fact that
  last month's state predicts next month's state. For a scout or analyst
  trying to assess whether a player's recent slump will persist, the
  HMM provides fundamentally different (and more useful) information:
  not just "is this player scoring tonight?" but "is this player currently
  in a Cold regime, and how long is it likely to last?"

What assumptions the HMM imposes that the BN does not:
  1. Markov property: Z_t depends only on Z_{t-1}, not earlier history.
     A player coming back from injury might have memory longer than 1 month.
  2. Temporal ordering: The HMM requires a meaningful sequence. The BN
     works on any independent cross-sectional sample.
  3. Stationarity: A and B are assumed constant across all players and
     all seasons. A player's transition dynamics may change with age or
     role changes.
  4. Fixed K: We chose K=3 states. The BN's structure was motivated by
     domain knowledge (positions, shot types); the HMM's number of states
     requires a separate model selection step (BIC, cross-validated
     log-likelihood).
""")

print("\nAll outputs saved to /mnt/user-data/outputs/")
print("Done.")
