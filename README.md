# NBA High Scorer Prediction & Minecraft MDP Agent
### CSE 150A — Milestone 6 Final Report
**Dustin Nguyen | A18553585**

---

## 1. Introduction

### Problem Statement

This project applies probabilistic reasoning to two connected problems: predicting whether an NBA player will score 20+ points in a game, and training an autonomous agent to navigate sequential decision-making in Minecraft. Both share a core challenge — an agent with uncertain, partially observable state must make decisions that lead to favorable long-term outcomes.

NBA player performance is inherently stochastic. Even a star player does not score 20+ every night — the outcome depends on interacting factors like shot volume, efficiency, playing time, and home court advantage that cannot be deterministically predicted. Beyond a single game, performance fluctuates over a season in streaks that a static snapshot model cannot capture. In Minecraft, actions succeed or fail stochastically, rewards are sparse, and the agent must plan dozens of steps ahead to progress through the tech tree.

### Why Probabilistic Modeling

Deterministic classifiers assign a single label with no notion of confidence. When features are noisy — as NBA stats inherently are — this produces brittle predictions. A logic-based rule system cannot capture conditional dependencies between variables or update beliefs gracefully when data is missing. Probabilistic models address this by reasoning about likely outcomes given partial evidence, inferring hidden state from observable proxies, and planning sequences of actions that maximize expected future reward. These three capabilities map exactly onto our three models: the BN, HMM, and MDP.

### PEAS Analysis

**Bayesian Network — High Scorer Prediction**

| PEAS | Description |
|------|-------------|
| Performance | Accuracy, Precision/Recall on HIGH_SCORER class, F1-score |
| Environment | NBA games (stochastic, partially observable, multi-agent) |
| Actuators | Posterior probability P(HIGH_SCORER=1 \| evidence) |
| Sensors | FGA, FG%, Minutes Played, Position, Home/Away flag |

**Hidden Markov Model — Season Performance Tracking**

| PEAS | Description |
|------|-------------|
| Performance | Log-likelihood of observation sequences; Viterbi decoded accuracy vs. ground truth |
| Environment | Time-ordered monthly stat aggregates across an NBA season |
| Actuators | Decoded latent performance state per month (Hot / Average / Cold) |
| Sensors | Monthly averages of PTS, FGA, FGP, MIN (discretized into 3 bins each) |

**MDP Agent — Minecraft Tech-Tree Progression**

| PEAS | Description |
|------|-------------|
| Performance | Cumulative reward per episode; tech tier reached; iron pickaxe crafted |
| Environment | Live Minecraft server via REST bridge (tile.ucsd.edu); stochastic, partially observable |
| Actuators | 186 discrete actions (movement, mining, crafting, smelting, combat) |
| Sensors | Position, health_bin, inventory boolean flags, proximity flags, time of day |

### Project Overview

Three models were built across three milestones, each extending the last:

1. **Bayesian Network**: A DAG encoding conditional dependencies between observable game statistics and the binary HIGH_SCORER target. Uses Maximum Likelihood Estimation for CPTs and Variable Elimination for exact inference.

2. **Hidden Markov Model**: Adds temporal structure to the BN. A player's latent performance state (Hot / Average / Cold) evolves as a Markov chain month-to-month; observable stats are emissions. Parameters estimated via Baum-Welch (EM); decoded via Viterbi.

3. **MDP Agent**: A model-based RL agent in Minecraft. Learns transition model T̂(s,a,s') from experience and solves for an optimal policy via Value Iteration and Policy Iteration — the same Bellman machinery that underpins the BN's conditional reasoning and the HMM's transition matrix.

---

## 2. Dataset & Preprocessing

### Dataset Source

**NBA Database (1947–Present)** via Kaggle (`PlayerStatistics.csv`). Filtered to post-2010 seasons, minimum 5 minutes played per game.

| Split | Rows |
|-------|------|
| Full filtered dataset | 446,911 |
| Training set (80%) | 357,528 |
| Test set (20%) | 89,383 |

### Task Definitions

- **BN**: Binary classification — predict HIGH_SCORER (points ≥ 20) from a single game's observable statistics
- **HMM**: Sequence decoding — infer the latent performance trajectory (Hot / Average / Cold) across a season's monthly splits
- **MDP**: Policy optimization — learn to advance from bare hands to iron pickaxe in Minecraft by maximizing cumulative reward

### Variables

| Variable | Raw Field | Type | Preprocessing |
|----------|-----------|------|---------------|
| FGA | fieldGoalsAttempted | Continuous | Discretized Low/Med/High (training bins only) |
| FGP | fieldGoalsPercentage | Continuous | Discretized Low/Med/High |
| MIN | numMinutes | Continuous | Discretized Low/Med/High |
| POS | startingPosition | Categorical | Guard / Forward / Center / Bench |
| IS_HOME | home | Binary | 0=away, 1=home |
| PTS_CAT | points | Continuous | Discretized Low/Med/High (intermediate BN node) |
| HIGH_SCORER | points ≥ 20 | Binary | Target variable |

**Class imbalance**: Only ~14% of games are HIGH_SCORER=1 (50,878 vs 306,650) — a 6:1 imbalance that biases predictions toward the majority class.

### Preprocessing Steps

All bins were computed on the training set only and applied to the test set to prevent data leakage. No Laplace smoothing was needed given 357,528 training rows — zero-count CPT cells were extremely unlikely.

For the HMM: per-game rows were aggregated into monthly averages per player-season and ordered chronologically (Oct → Apr), producing sequences of length T=6 per player-season. The same 3-bin equal-frequency discretization as the BN was applied to PTS, FGA, FGP, and MIN, with a composite observation symbol encoding all four bins as a base-3 number (81 possible symbols).

For the MDP: the bridge pre-computes boolean flags (`has_wood`, `has_wood_tools`, etc.) that directly encode tech-tree gates. Position is bucketed with `grid_x // 5` to keep the state space finite. Raw inventory dicts and the 5×5 nearby grid were excluded — both are unhashable and would create an exponentially large, mostly unreachable state space.

### Design Justifications

The 5-block position bucket for the MDP was chosen after empirical observation that 10-block buckets produced only 2 unique states in early training. Finer buckets (5 blocks) produced 307 unique states across 168 episodes — enough spatial resolution for meaningful policy learning without blowing up the state space. The boolean inventory flags were chosen over raw counts because they encode the exact tech-tree gates the bridge uses to determine action availability.

---

## 3. Methods

### 3.1 Bayesian Network

#### Model Structure (DAG)

```
    POS ──────────► FGA ──────────► MIN ──────┐
     │               │                         │
     └──────────► FGP               ▼          ▼
                   │         ┌── PTS_CAT ──► HIGH_SCORER ◄── IS_HOME
                   └─────────┘
```

**Nodes**: POS, FGA, FGP, MIN, IS_HOME, PTS_CAT, HIGH_SCORER

**Edge rationale**:
- `POS → FGA`: A player's position causally determines shot volume — guards take more 3-pointers, centers shoot closer to the basket
- `POS → FGP`: Position determines shooting role and typical efficiency profile
- `FGA → MIN`: Players who attempt more shots generally play more minutes; role and playing time are correlated
- `FGA, FGP → PTS_CAT`: Points = volume × efficiency. This is a v-structure (collider) — FGA and FGP are marginally independent but become dependent once PTS_CAT is observed
- `PTS_CAT, MIN, IS_HOME → HIGH_SCORER`: The 20-point threshold depends directly on scoring level, playing time, and home court
- `IS_HOME ⊥ POS` (d-separated) — game location is independent of player position

#### Parameter Estimation

Conditional probability tables (CPTs) estimated via Maximum Likelihood Estimation:

$$P(X_i \mid \text{Parents}(X_i)) = \frac{\text{count}(X_i = x,\ \text{Parents}(X_i) = pa)}{\text{count}(\text{Parents}(X_i) = pa)}$$

No Laplace smoothing needed at 357,528 rows.

#### Independence Assumptions

The BN assumes the Markov condition: each node is conditionally independent of its non-descendants given its parents. Key assumed independences: `IS_HOME ⊥ POS` (d-separated — game location independent of player role); `FGA ⊥ FGP | POS` (shot volume and efficiency are independent once position is known).

#### Inference

Variable Elimination — computes exact posterior P(HIGH_SCORER=1 | evidence) by summing out intermediate variables in topological order.

#### Improvements from Milestone 4

The original model tested with fewer features. PER was considered as an additional node but excluded because it is derived from the same points column as HIGH_SCORER, which would create a near-trivial predictive path. The final structure uses only pre-game observable features (FGA, FGP, MIN, POS, IS_HOME) plus PTS_CAT as an intermediate node.

**Library**: `pgmpy` — `DiscreteBayesianNetwork`, `DiscreteMLE`, `VariableElimination`

---

### 3.2 Hidden Markov Model

#### Model Structure

The HMM models a player's season as a latent Markov chain emitting observable statistics at each monthly time step.

- **State space**: {Hot, Average, Cold} — K=3 hidden states representing above/at/below baseline performance
- **Observation space**: 81 composite symbols (3⁴) encoding discretized monthly averages of PTS, FGA, FGP, MIN
- **Initial distribution π**: P(performance state at season start), learned from data
- **Transition matrix A** (3×3): A[i][j] = P(state_j at month t+1 | state_i at month t)
- **Emission matrix B** (3×81): B[i][k] = P(observation_k | hidden_state_i)

#### Connection to BN

The HMM shares the BN's observable nodes (FGA, FGP, MIN, PTS) as emissions, but adds temporal structure. The BN's HIGH_SCORER target corresponds to the downstream consequence of the latent performance state — instead of predicting a single game, the HMM infers which performance state a player occupies each month and how it evolves. The key addition is transition matrix A, which captures month-to-month performance momentum that the static BN cannot represent.

#### Parameter Estimation — Baum-Welch (EM)

Parameters (π, A, B) are learned via the Baum-Welch algorithm:

**E-step**: Compute forward α_t(i) = P(o_1,...,o_t, s_t=i | λ) and backward β_t(i) = P(o_{t+1},...,o_T | s_t=i, λ). Derive:

$$\gamma_t(i) = \frac{\alpha_t(i)\beta_t(i)}{\sum_j \alpha_t(j)\beta_t(j)}, \qquad \xi_t(i,j) = \frac{\alpha_t(i)\,A_{ij}\,B_j(o_{t+1})\,\beta_{t+1}(j)}{\sum_{i,j}\alpha_t(i)\,A_{ij}\,B_j(o_{t+1})\,\beta_{t+1}(j)}$$

**M-step**: Re-estimate parameters:

$$A_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}, \qquad B_{ik} = \frac{\sum_{t:\,o_t=k} \gamma_t(i)}{\sum_t \gamma_t(i)}, \qquad \pi_i = \gamma_1(i)$$

Convergence: log-likelihood change < 1e-4, typically ~35 iterations.

#### Inference — Viterbi

$$s^* = \arg\max_{s_{1:T}} P(s_{1:T} \mid o_{1:T},\ \lambda)$$

Viterbi was chosen over the Forward algorithm because we want interpretable per-month labels (the single best state sequence), not just the marginal probability of the observation sequence. This directly supports the task: identifying when a player entered a Cold streak and how long it lasted.

#### Improvements from Milestone 5

K=3 states was selected after comparing log-likelihoods for K=2, 3, and 4. K=2 collapsed Hot and Average into a single state; K=4 produced two nearly identical intermediate states with no interpretable distinction. K=3 produced the best balance of fit and interpretability.

**Library**: `hmmlearn.hmm.CategoricalHMM`

---

### 3.3 Markov Decision Process

#### Formal Description

MDP tuple (S, A, T, R, γ):

- **S**: 14-element state tuples. ~307 unique states observed across 168 episodes
- **A**: 186 discrete actions fetched from the bridge at runtime
- **T(s,a,s')**: Learned online via count-based MLE: T̂(s,a,s') = count(s,a,s') / count(s,a). Falls back to self-loop prior for unseen (s,a) pairs
- **R(s,a,s')**: Milestone-shaped reward (see table below)
- **γ = 0.97**: Far-sighted — the iron pickaxe goal is ~100+ steps away; γ^100 ≈ 0.048 still preserves meaningful signal

#### State Representation

| Index | Feature | Values | Rationale |
|-------|---------|--------|-----------|
| 0–1 | gx, gz | int (pos // 5) | 5-block position buckets — finer than 10 to discover more distinct states |
| 2 | health_bin | 0–3 | Survival signal; enables "eat when low" behavior |
| 3 | has_wood | bool | First tech-tree step |
| 4 | has_planks | bool | Intermediate crafting material |
| 5 | has_sticks | bool | Required for all tool recipes |
| 6 | has_stone | bool | Cobblestone — required for stone tools and furnace |
| 7 | has_wood_tools | bool | Wooden pickaxe — tech gate for coal/stone mining |
| 8 | has_stone_tools | bool | Stone pickaxe — tech gate for iron mining |
| 9 | has_furnace | bool | Unlocks smelting pipeline |
| 10 | has_table_nearby | bool | Crafting table within 4 blocks (gates 3×3 recipes) |
| 11 | has_furnace_nearby | bool | Furnace within 4 blocks (gates smelting) |
| 12 | time_of_day | 0=day, 1=night | Night increases mob spawn risk |
| 13 | has_food | bool | Food for health recovery |

#### Reward Function

Rewards fire only on true state-bit transitions — they never trigger when an action silently fails and the state stays unchanged.

| Event | Reward | Rationale |
|-------|--------|-----------|
| First log obtained | +1.0 | First tech-tree step |
| First planks crafted | +1.0 | Intermediate material |
| First cobblestone mined | +2.0 | Harder to obtain; gates stone tools |
| Crafting table placed nearby | +1.5 | Enables 3×3 recipes |
| Wooden pickaxe crafted | +8.0 | Major milestone — unlocks mining |
| Stone pickaxe crafted | +15.0 | Bigger milestone — unlocks iron |
| Furnace crafted | +5.0 | Unlocks smelting |
| Iron pickaxe crafted | +30.0 | **Goal state** |
| Entered critical health | −3.0 | Discourages reckless behavior |
| Nightfall transition | −0.5 | Encourages shelter/caution |
| Per step | −0.05 | Discourages idle looping |

Milestone magnitudes are calibrated against γ=0.97: a reward N steps away is worth γ^N of its face value, so each milestone is large enough to overcome the accumulated step penalties between it and the next goal.

#### Planning Algorithms

**Value Iteration** — Bellman optimality update:

$$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} \hat{T}(s,a,s') \cdot V_k(s') \right]$$

Convergence: max_s|V_{k+1}(s) − V_k(s)| < θ = 10⁻⁴. On the 307-state model, VI typically converges in 20–80 iterations.

**Policy Iteration** — alternates between:
1. **Policy Evaluation**: iterate V(s) = R(s,π(s)) + γ·Σ T(s,π(s),s')·V(s') until max|ΔV| < θ
2. **Policy Improvement**: π'(s) = argmax_a [R(s,a) + γ·Σ T(s,a,s')·V(s')]

Both algorithms are run every 5 episodes. Their policies are compared — agreement percentage measures model stability.

#### Exploration Strategy

Epsilon-greedy with exponential decay:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| ε_start | 0.90 | Start mostly random to explore the state space rapidly |
| ε_decay | 0.994 per episode | Gradual shift from exploration to exploitation |
| ε_min | 0.10 | Always keep 10% exploration to handle newly discovered states |

After 168 episodes, ε ≈ 0.327 — the agent is approximately 67% exploiting its learned policy.

#### Connection to Earlier Models

The MDP's T̂(s,a,s') is structurally identical to the HMM's transition matrix A — both encode P(next state | current state). The MDP generalizes this by conditioning on a chosen action and using the model to plan rather than decode. The BN's CPTs and MDP's T̂ are both estimated by the same MLE frequency-counting procedure; only the conditioned variables differ.

---

## 4. Training & Implementation

### Bayesian Network
- **Train/test split**: 80/20 random split (357,528 train / 89,383 test)
- **Parameter estimation**: MLE frequency counts, no smoothing
- **Inference**: `pgmpy.inference.VariableElimination`
- **Library**: `pgmpy` — `DiscreteBayesianNetwork`, `DiscreteMLE`
- **Code**: [`notebook.ipynb`](notebook.ipynb)

### Hidden Markov Model
- **Sequences**: Monthly stat aggregates per player-season (Oct–Apr, T=6 steps)
- **Hidden states**: K=3 (Hot / Average / Cold)
- **Observations**: 81 composite symbols (3⁴ bins over PTS, FGA, FGP, MIN)
- **EM convergence**: log-likelihood change < 1e-4, ~35 iterations
- **State selection**: K=3 chosen by comparing log-likelihoods for K=2,3,4
- **Library**: `hmmlearn.hmm.CategoricalHMM`
- **Code**: [`notebook_hmm.py`](notebook_hmm.py)

### MDP Agent
- **Episodes completed**: 168
- **States discovered**: 307
- **Total transitions**: 33,600
- **Hyperparameters**: γ=0.97, ε_start=0.90, ε_decay=0.994, ε_min=0.10, θ=1e-4
- **MAX_STEPS**: 200 per episode (server-paced; each step ~1–3s)
- **BUCKET_SIZE**: 5 blocks
- **Replan frequency**: Every 5 episodes (both VI and PI run and compared)
- **Checkpoint frequency**: Every 5 episodes, atomic save to `results/`
- **Libraries**: Custom VI/PI; `gymnasium` for env wrapper; `requests` for bridge communication
- **Code**: [`student/mdp_definition.py`](student/mdp_definition.py), [`student/mdp_agent.py`](student/mdp_agent.py)

---

## 5. Results & Discussion

### 5.1 Bayesian Network Results

| Model | Accuracy | Precision (High) | Recall (High) | F1 (High) |
|-------|----------|-----------------|---------------|-----------|
| Majority Class Baseline | 85.5% | — | — | — |
| Naive Bayes | 89.6% | 0.61 | 0.80 | 0.69 |
| **Bayesian Network** | **90.0%** | **0.78** | **0.43** | **0.55** |

**Confusion Matrix:**

| | Predicted: Not High | Predicted: High |
|---|---|---|
| **Actual: Not High** | 74,815 (TN) | 1,585 (FP) |
| **Actual: High** | 7,392 (FN) | 5,591 (TP) |

**Interpretation**: The BN achieves 90.0% accuracy, beating the majority class baseline by 4.5pp and Naive Bayes by 0.4pp. Explicitly modeling the conditional dependencies between position, shot volume, efficiency, and minutes provides more accurate predictions than assuming all features are independent. The v-structure at PTS_CAT captures the shot-volume × efficiency interaction that Naive Bayes cannot represent.

The model is conservative: precision is 78% but recall is only 43% on HIGH_SCORER=1. It correctly identifies high scorers when confident but misses many true positives. The 6:1 class imbalance drives this — predicting "Not High" is almost always safe.

**Key limitation**: PTS_CAT is derived from the same `points` column as HIGH_SCORER, creating a strong predictive path that is somewhat circular. A cleaner formulation would predict solely from pre-shot features.

---

### 5.2 Hidden Markov Model Results

**Learned transition matrix A:**

| From \ To | Cold | Average | Hot |
|-----------|------|---------|-----|
| **Cold** | 0.65 | 0.30 | 0.05 |
| **Average** | 0.15 | 0.65 | 0.20 |
| **Hot** | 0.05 | 0.28 | 0.67 |

Performance states are highly persistent — a Hot player stays Hot 67% of the time; a Cold player stays Cold 65%. This captures the "streak" phenomenon that the BN cannot model.

| Metric | Value |
|--------|-------|
| Baseline (always predict Average) | 47.9% |
| Viterbi decoded accuracy vs. ground truth | 55.0% |
| Improvement over baseline | +7.1pp |
| EM convergence | ~35 iterations |

**Comparison to BN**: The HMM captures temporal momentum the BN cannot — January performance informs February's likely state via A. The BN treats each observation independently and cannot recognize that a player in a Hot state is more likely to produce HIGH_SCORER=1 games next month. However, the HMM requires temporally ordered data and is sensitive to K; the BN works on any cross-sectional sample and has more interpretable conditional probability tables.

---

### 5.3 MDP Results

**Training summary:**

| Metric | Value |
|--------|-------|
| Episodes completed | 168 |
| States discovered | 307 |
| Total transitions recorded | 33,600 |
| Avg reward first 10 episodes | −10.55 |
| Avg reward best episode seen | −2.50 (ep31) |
| Avg reward last 10 episodes | −10.15 |
| Avg PI/VI agreement | ~96.7% |
| ε at end of training | 0.327 |
| Iron pickaxe achieved | No (insufficient training time) |

**Learning curve analysis**: Rewards started at −14.50 (episode 1), improved to −2.50 by episode 31 as the policy began learning the early tech-tree steps, then regressed back to −10 range as the state space grew from 12 to 307 states and the agent entered a new exploration phase. This U-shaped curve is characteristic of model-based RL — each time new states are discovered, the value function must re-converge over a larger state space.

**Policy behavior**: During early episodes (ε≈0.90), the agent takes mostly random actions. By episode 30, the policy had learned consistent early tech-tree behavior: the PI/VI agreement jumped to 96.7% over 30 states, indicating the learned T̂ had stabilized for the initially explored region. As the state space grew to 307 states, the policy continued refining.

**Baseline comparison**: A random policy (ε=1.0 throughout) achieves approximately −10 to −14 reward per episode — pure step penalties with no milestone rewards. The learned policy at episode 31 achieved −2.50, a 74% improvement over the random baseline, before regressing as new states were explored.

**PI vs VI**: VI consistently converged in <0.05 seconds. PI hit the 500-iteration cap in early training when T̂ was sparse and noisy, but settled at 96.7% agreement with VI by episode 30 once the model stabilized. Both algorithms produce nearly identical policies on the converged model, validating that the learned T̂ is an accurate representation of the environment.

---

### 5.4 Cross-Model Comparison

| Model | Task | Key Metric | Result | Key Strength |
|-------|------|-----------|--------|--------------|
| Bayesian Network | Single-game HIGH_SCORER prediction | Accuracy / F1 | 90.0% acc, 0.55 F1 | Interpretable CPTs; handles missing features gracefully |
| Hidden Markov Model | Season performance trajectory | Viterbi accuracy | 55.0% vs 47.9% baseline | Captures temporal streaks and momentum |
| MDP Agent | Minecraft tech-tree progression | Reward improvement | −14.50 → −2.50 best episode | Sequential planning under uncertainty |

**What each model captures uniquely**: The BN answers "given these stats right now, how likely is a 20-point game?" The HMM answers "given the last few months of performance, what state is this player in and where is it going?" The MDP answers "given my current inventory and position, what sequence of actions maximizes future reward?" Each model strictly requires the previous model's reasoning as a building block — the MDP's T̂ is the HMM's A generalized to actions, and the HMM's emissions are the BN's observable nodes extended through time.

**Strongest model**: The MDP agent demonstrates the most complete form of probabilistic reasoning — it simultaneously estimates a transition model from noisy experience (like BN's MLE), tracks state evolution over time (like HMM's A), and plans sequences of actions to maximize expected future reward (Bellman). The BN and HMM are passive classifiers/decoders; the MDP actively changes the world.

**Key assumptions and trade-offs**:
- **BN**: Conditional independence given parents — violated by correlated stats (PTS and USG%). Large dataset makes CPTs reliable, but PTS_CAT creates a partially circular prediction path.
- **HMM**: Markov property — reasonable month-to-month, less so for injury recovery with longer memory. K=3 states chosen empirically; stationarity assumed across all players.
- **MDP**: Stationary environment — partially violated as the Minecraft world changes as the bot digs. Tabular T̂ requires bounded state space.

---

### 5.5 Limitations & Future Work

**Bayesian Network**:
- Coarse 3-bin discretization loses information. Using 5 bins or quantile-based discretization would improve the precision/recall balance.
- The 6:1 class imbalance biases toward predicting 0. SMOTE oversampling or threshold tuning (0.5 → 0.3) would improve recall on the HIGH_SCORER class.
- PTS_CAT creates a near-trivial predictive path. Removing points entirely and predicting from FGA, FGP, MIN, POS, IS_HOME would test whether the model truly learns scoring dynamics from pre-shot features.

**Hidden Markov Model**:
- Monthly splits are too coarse — game-level sequences would capture within-month variance but require much more data for EM convergence.
- K=3 states was selected by inspection. Formal selection via BIC or cross-validated log-likelihood would be more principled.
- Emissions do not account for opponent defensive quality. Adding opponent defensive rating as a feature would improve state discrimination between Hot and Average.

**MDP Agent**:
- 168 episodes was insufficient to converge the policy over 307 states. The U-shaped reward curve (improving then regressing as new states were discovered) suggests the agent needed several hundred more episodes to stabilize.
- No `has_iron_tools` bit in the state tuple — goal detection relies on checking raw inventory after action 77, not state-embedded information.
- Inventory carries across episodes, meaning accumulated materials from early random episodes artificially assist later ones. A true inventory reset would produce cleaner, more interpretable learning curves.
- Position buckets (5 blocks) may still be too coarse to distinguish the exact location of a crafting table. Adding `distance_to_nearest_table` as a discretized feature would improve crafting navigation.

---

## 6. Conclusion

This project built three probabilistic models with increasing complexity: a Bayesian Network for static single-game prediction, a Hidden Markov Model for seasonal trajectory modeling, and an MDP agent for sequential planning in Minecraft.

The BN demonstrated that explicitly modeling conditional dependencies outperforms both the majority class baseline (85.5%) and Naive Bayes (89.6%), achieving 90.0% accuracy. The structured v-structure at PTS_CAT captures the shot-volume × efficiency interaction that independence-assuming models miss. The HMM extended this to temporal reasoning, learning that performance states are persistent (Hot→Hot 67%, Cold→Cold 65%) — a phenomenon invisible to the static BN. The MDP closed the loop from perception to action, with policy improvement visible at episode 31 (−2.50 reward, 74% better than the random baseline) before the exploration-exploitation tradeoff drove further state discovery.

The strongest model is the MDP agent — it requires all three forms of probabilistic reasoning simultaneously: MLE estimation (BN's CPTs), transition modeling (HMM's A), and Bellman planning. High PI/VI agreement (96.7%) after 6,000 transitions validates that the learned T̂ accurately captures the local environment dynamics.

The key insight is that uncertainty is fundamental to both domains — NBA stats are noisy proxies for true player quality, and Minecraft actions succeed stochastically. Probabilistic models handle this by maintaining distributions over latent states rather than committing to point estimates. This approach is most valuable when observations are noisy, the true state is latent, and decisions have long-horizon consequences. It struggles when the Markov assumption is severely violated, the state space is too large for tabular methods, or rewards are too sparse for the value function to propagate before the training budget runs out.

---

## 7. Statement of Collaboration

**Dustin Nguyen (A18553585)**: Implemented the Bayesian Network (Milestone 4), HMM (Milestone 5/6), and MDP agent (`mdp_definition.py`, `mdp_agent.py`). Ran all Minecraft training experiments (168 episodes, 307 states, 33,600 transitions). Wrote the full final report.

*This project was completed individually.*

---

## 8. Citations & AI Disclosure

### Libraries
- **pgmpy** — Ankan, A. & Panda, A. (2015). pgmpy: Probabilistic Graphical Models using Python. SciPy 2015. https://pgmpy.org/
- **hmmlearn** — https://hmmlearn.readthedocs.io
- **gymnasium** — Farama Foundation (2022). https://gymnasium.farama.org/
- **numpy** — Harris et al. (2020). Nature, 585, 357–362.
- **matplotlib** — Hunter, J. D. (2007). Computing in Science & Engineering, 9(3), 90–95.
- **plotly** — Plotly Technologies Inc. (2015). https://plotly.com
- **requests** — Reitz, K. (2011). https://requests.readthedocs.io
- **hmmlearn** — https://hmmlearn.readthedocs.io

### Dataset
- NBA Database (1947–Present): https://www.kaggle.com/datasets/eoinamoore/historical-nba-data-and-player-box-scores

### Textbooks
- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Russell, S. & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
- UCSD CSE 150A Lecture Notes — Bayesian Networks, HMMs, MDPs.

### AI Disclosure
**Claude (Anthropic)** was used throughout this project to help implement `mdp_definition.py` and `mdp_agent.py`, debug the TransitionMatrix API (`record`/`observed_states`/`num_transitions`), fix the `terminal_fn` health-check bug that caused 0-step episodes, and structure this final report. All design decisions — DAG structure, state representation, reward function, hyperparameters — were made by Dustin Nguyen. Claude also helped in Milestone 4 to identify a pgmpy API compatibility issue (`BayesianNetwork` → `DiscreteBayesianNetwork`) and diagnose a pandas `TypeError` from mixed-type columns.
