# Task 1 — Notebook Plan
## Sequential list of cells/tasks to implement in the Task 1 notebook

> **Reading guide**: Each item is a notebook cell or small group of cells.  
> Priority **10** = critical for grade / core deliverable. **1** = nice-to-have polish.  
> Cells marked 🟦 are **Markdown** (explanation/theory). Cells marked 🟩 are **Code**.  
> The notebook should read like a learning document: a reader with DL background but no RL background should finish it understanding DQN from scratch.

---

---
# **0. Project Setup**
*Environment bootstrap, constants, reproducibility. Run once at the start of every session.*

---
## 0.1 Constants & Hyperparameters

| #     | Type | Task            | Priority | Description                                                                                                                                                                                                                                                                                                      |
| ----- | ---- | --------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.1.1 | 🟩    | Constants block | **9**    | Central config dict or constants: `SEED`, `DEVICE`, `OBS_MODE="rgb"`, `DIFFICULTY=0`, `ROUND_TIME=60`, `TICKS_PER_SEC=10`, `GAMMA=0.99`, `LR`, `BATCH_SIZE` (even if =1 for Task 1), `EPS_START`, `EPS_END`, `EPS_DECAY`, `N_EPISODES`, `EVAL_EVERY`. Having all hyperparams in one place makes ablations clean. |

---
## 0.2 Imports & Environment Setup

| #     | Type | Task            | Priority | Description                                                                                                                                                           |
| ----- | ---- | --------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.2.1 | 🟩    | Imports         | **9**    | `torch`, `numpy`, `matplotlib`, `gymnasium`, `SpaceRaceEnv`, `collections`, `random`, `time`, `os`, `json`. Set `random.seed`, `np.random.seed`, `torch.manual_seed`. |
| 0.2.2 | 🟩    | Device check    | **8**    | Print CUDA availability, select `device`. Brief inline comment on why GPU matters for CNN training.                                                                   |
| 0.2.3 | 🟩    | Utility helpers | **6**    | `set_seed()`, `save_checkpoint()`, `load_checkpoint()` helper functions. Keeps later cells clean.                                                                     |

---

---
# **1. The SpaceRace Environment**
*Before building any agent, we must deeply understand the playground: what the agent sees, what it can do, and what good/bad play looks like.*

---
## 1.1 Environment Instantiation & API Tour

| #     | Type | Task             | Priority | Description                                                                                                                                                                                                 |
| ----- | ---- | ---------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.1.1 | 🟦    | Intro markdown   | **7**    | Explain the SpaceRace game concept, the goal (maximize crossings in 60s), the action space (0=up, 1=down), reward structure (+1 crossing, -0.25 collision, +0.02 up, -0.01 down). Include the reward table. |
| 1.1.2 | 🟩    | Env creation     | **9**    | `env = SpaceRaceEnv(obs_mode="rgb", difficulty=0, ...)`. Call `reset()`. Print `obs.shape`, `obs.dtype`, `env.action_space`, `info.keys()`.                                                                 |
| 1.1.3 | 🟩    | Single step demo | **8**    | Execute one step with action=0 and action=1. Print `obs.shape`, `reward`, `terminated`, `truncated`, `info`. Show the diff between the two resulting obs visually.                                          |

---
## 1.2 Observation Space Analysis (EDA)

| #     | Type | Task                         | Priority | Description                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----- | ---- | ---------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.2.1 | 🟦    | Obs space theory             | **8**    | Explain RGB obs: shape (54,39,3), uint8 [0-255]. Explain the 3 pixel colors: dark blue=background, yellow=debris, cyan=ship. Mention upsampling 3× from 18×13 grid.                                                                                                                                                                                                                                                                     |
| 1.2.2 | 🟩    | **Plot: Sample frames grid** | **9**    | Collect 6–8 frames at different game moments (start, mid-game, near collision, after crossing). Show as `plt.imshow` grid. Caption each. This is the visual anchor for the whole notebook — the reader sees the game for the first time here.                                                                                                                                                                                           |
| 1.2.3 | 🟩    | Pixel value analysis         | **7**    | Histogram of pixel values across channels (R, G, B) for a batch of frames. Show that values cluster around 3 known colors. Motivates normalization.                                                                                                                                                                                                                                                                                     |
| 1.2.4 | 🟩    | Normalized obs demo          | **8**    | Show `obs_float = obs.astype(np.float32) / 255.0`. Print min/max/mean. Show side-by-side of raw vs normalized frame. This is the preprocessing step the CNN will receive.                                                                                                                                                                                                                                                               |
| 1.2.5 | 🟩    | Channel decomposition plot   | **6**    | Plot the R, G, B channels separately for one frame. Useful to visually see where the ship/debris/background lives in each channel.                                                                                                                                                                                                                                                                                                      |
| 1.2.6 | 🟦    | Frame stacking motivation    | **7**    | Explain *why* a single frame is not enough to capture motion/direction of debris. Introduce frame stacking concept: stack N consecutive frames along channel dim. **Decision taken: R-channel only (best object separation: background=5, ship=90, debris=230 in R — far better than grayscale where debris≈182 and ship≈185 are nearly identical). Stack 4 consecutive frames, no skip → input shape (4, 54, 39) in PyTorch (C,H,W).** |

> ✅ **D1 RESOLVED — Frame stacking**: 4 consecutive frames, no frame skip.
> ✅ **D2 RESOLVED — Preprocessing**: R-channel only (NOT grayscale). Grayscale gives nearly identical values for debris and ship. R-channel: background=5, ship=90, debris=230 → perfectly separated. Pipeline: `obs[:,:,0].astype(float32) / 255.0` → stack 4 → shape (4,54,39).


---
## 1.3 Reward & Episode Analysis

| #     | Type | Task                                      | Priority | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ----- | ---- | ----------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.3.1 | 🟩    | **Random agent rollout**                  | **9**    | Run 5 full episodes with random actions. Collect per-step rewards and episode scores. Print mean/std score. This is the **random baseline** — the floor we must beat.                                                                                                                                                                                                                                                                                                            |
| 1.3.2 | 🟩    | **Plot: Random agent score distribution** | **8**    | Bar chart of scores across random episodes. Box plot or histogram. Establishes the lower bound clearly.                                                                                                                                                                                                                                                                                                                                                                          |
| 1.3.3 | 🟩    | Reward signal analysis                    | **7**    | Plot cumulative reward per step for one episode. Annotate crossing events (+1) and collision events (-0.25). Reader understands the reward density and sparsity.                                                                                                                                                                                                                                                                                                                 |
| 1.3.4 | 🟦    | Episode structure explanation             | **7**    | Explain: `terminated=False` always, `truncated=True` at 60s, collisions cause respawn (not termination). Crucial to understand that done = `terminated or truncated`.                                                                                                                                                                                                                                                                                                            |
| 1.3.5 | 🟩    | **Reward scale analysis**                 | **8**    | Key insight: at 10 ticks/sec × 60s = 600 steps, moving up ~70% = 420 × +0.02 = **+8.4 from movement alone** vs ~5 crossings = **+5.0**. Per-step movement reward dominates total reward (~77%). This means optimizing total reward ≠ maximizing crossings (the actual score metric). Show this calculation explicitly. Discuss: does the agent risk always running up (maximizing +0.02/step) while ignoring collisions? This is a genuine tension worth noting in the analysis. |

---

---
# **2. Heuristic Policy**
*Before any RL, we build a hand-coded agent using privileged semantic information. This is our gold-standard "intelligent" baseline.*

---
## 2.1 Semantic Observation & Game Understanding

| #     | Type | Task                       | Priority | Description                                                                                                                                                                                                                                                          |
| ----- | ---- | -------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.1.1 | 🟦    | Heuristic motivation       | **8**    | Explain: semantic obs is available via `info["semantic_obs"]` at training time (shape 18×13×3, channels: ship, debris, time). At evaluation, this is disabled. Heuristic is only for baseline comparison and optionally warm-starting the replay buffer (in Task 2). |
| 2.1.2 | 🟩    | Semantic obs visualization | **8**    | Plot the 3 channels of semantic obs for a sample frame: channel 0 (ship position), channel 1 (debris map), channel 2 (time remaining). Side by side with the RGB frame. Visual intuition for why heuristics are easy to build from semantic obs.                     |

---
## 2.2 Heuristic Implementation

| #     | Type | Task                                               | Priority | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----- | ---- | -------------------------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.2.1 | 🟦    | Algorithm description                              | **9**    | Write the heuristic logic in pseudocode. Two versions, both implemented in `src/heuristic/heuristic.py` and selectable via `mode` constant (set in notebook cell 0.1): (A) **Naive** (`HEURISTIC_NAIVE`): check only 1 row above ship's current row — if clear, go up; else wait. Assignment baseline. (B) **Enhanced** (`HEURISTIC_ENHANCED`, default): (1) find ship's row from `info["semantic_obs"]` channel 0; (2) scan N=3 rows ahead in ship's column for debris; (3) if clear → action=0 (up); (4) if blocked → scan adjacent rows above/below for nearest clear row; (5) move toward it. Ship moves **vertically** (up/down). Debris moves **horizontally**. Goal: be in the right row when the gap arrives. |
| 2.2.2 | 🟩    | `extract_info_from_obs()`                          | **9**    | Extract ship_row, ship_col from `info["semantic_obs"]` channel 0. Build debris map from channel 1. Return ship position + full debris grid for lookahead logic.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2.2.3 | 🟩    | **`heuristic_action(info, mode)` — both versions** | **10**   | Call `heuristic_action(info, mode=HEURISTIC)` where `HEURISTIC` is the constant from cell 0.1. `mode=None` defaults to `HEURISTIC_ENHANCED`. Compare naive vs enhanced on same episodes to show improvement. Both versions in one function — clean interface, easy to ablate.                                                                                                                                                                                                                                                                                                                                                                                                                                         |

---
## 2.3 Heuristic Evaluation

| #     | Type | Task                                   | Priority | Description                                                                                                                                                                                  |
| ----- | ---- | -------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2.3.1 | 🟩    | **Heuristic rollout**                  | **10**   | Run 5–10 episodes with the heuristic. Collect scores, timing, collision counts.                                                                                                              |
| 2.3.2 | 🟩    | **Table: Baseline comparison**         | **10**   | Summary table comparing Random vs Heuristic: mean score, std score, mean crossings, mean collisions, avg episode duration. This is a graded deliverable (Heuristic Policy = 10%).            |
| 2.3.3 | 🟩    | **Plot: Score comparison bar chart**   | **9**    | Side-by-side bar chart (with error bars) of Random vs Heuristic mean scores. Clear visual showing the heuristic's improvement.                                                               |
| 2.3.4 | 🟩    | **Plot: Heuristic episode trajectory** | **7**    | Plot cumulative reward over steps for best heuristic episode. Annotate crossing events. Compare with a random episode trajectory on the same axes.                                           |
| 2.3.5 | 🟦    | Heuristic analysis & limitations       | **8**    | Discuss: what does the heuristic do well? What does it fail at? (e.g., debris approaching from the side, multi-row blocking). This motivates RL — the heuristic is hand-crafted and brittle. |

---

---
# **3. Neural Network Architecture**
*Design the CNN that maps raw pixels to Q-values. Every design decision is explained and justified.*

---
## 3.1 Architecture Design

| #     | Type | Task                                              | Priority | Description                                                                                                                                                                                                                                                                                                                                                        |
| ----- | ---- | ------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 3.1.1 | 🟦    | DQN theory intro                                  | **10**   | Explain DQN: takes state s, outputs Q(s,a) for all actions a. How it relates to a Q-table (but generalized to continuous state spaces). Include the diagram: frame → CNN → flatten → FC → [Q(s,0), Q(s,1)].                                                                                                                                                        |
| 3.1.2 | 🟦    | Architecture design rationale                     | **9**    | Justify choices: why CNNs for image input (spatial invariance, parameter efficiency). Why small network (stability, training speed). Why NO pooling (position-sensitive game — exact ship/debris location matters, pooling destroys this). Why strided convolutions instead. Note input size (54×39 is small).                                                     |
| 3.1.3 | 🟩    | `DQN` class implementation                        | **10**   | PyTorch `nn.Module`. **Architecture: Conv1(8×8, stride=4, 16 filters, ReLU) → Conv2(4×4, stride=2, 32 filters, ReLU) → Flatten(480) → FC1(128, ReLU) → FC2(2, no activation)**. Input: (N_FRAMES=4, 54, 39). He init. No BN, no Dropout. Constructor takes `n_frames`, `n_actions=2`.                                                                              |
| 3.1.4 | 🟩    | Architecture summary + **dimension verification** | **9**    | `torchsummary.summary()` or manual layer-by-layer shape computation. **Explicitly verify**: Input(4,54,39) → Conv1(8×8,s=4) → (16,12,8) → Conv2(4×4,s=2) → (32,5,3) → Flatten = **480** → FC1(128) → FC2(2). Confirm math: H_out = ⌊(H_in−k)/s⌋+1. Print actual param count. This cell catches any future architecture change that breaks the FC1 input dimension. |
| 3.1.5 | 🟩    | Sanity check forward pass                         | **8**    | Feed a random batch through the network. Assert output shape = (batch, 2). Confirm gradients flow.                                                                                                                                                                                                                                                                 |
| 3.1.6 | 🟦    | Input preprocessing pipeline                      | **8**    | Document the full pipeline: raw RGB (54,39,3) uint8 → **R-channel only** `obs[:,:,0]` → float32 → /255 → FrameStack deque (4 frames) → stack to (4,54,39) → PyTorch tensor on device. Why R-channel: debris=230, ship=90, background=5 — perfect separation (grayscale fails: debris≈182 ≈ ship≈185).                                                              |

---

---
# **4. Q-Learning Loss Function**
*The mathematical heart of DQN. Explained from first principles.*

---
## 4.1 Theory

| #     | Type | Task                                          | Priority | Description                                                                                                                                                                                                                                                    |
| ----- | ---- | --------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.1.1 | 🟦    | Bellman equation                              | **10**   | Derive the Bellman optimality equation: `Q*(s,a) = E[r + γ max_a' Q*(s',a')]`. Explain γ (discount factor), the bootstrapping idea, and why the loss is MSE between predicted Q(s,a) and target `r + γ max_a' Q(s',a')`. Show the formula clearly using LaTeX. |
| 4.1.2 | 🟦    | Task 1 caveat: instability without target net | **9**    | Explain why in Task 1 we use the *same* network for both prediction and target — this creates a moving target problem. The target changes every update. This is intentional for Task 1 (to motivate the target network in Task 2).                             |

---
## 4.2 Implementation

| #     | Type | Task                      | Priority | Description                                                                                                                                                                                                                                               |
| ----- | ---- | ------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.2.1 | 🟩    | `compute_loss()` function | **10**   | Takes `(state, action, reward, next_state, done, network, gamma)`. Computes predicted Q(s,a) via forward pass, computes target `r + γ * (1-done) * max_a' Q(s',a')` with `torch.no_grad()`, returns MSE loss. Well-commented with inline math references. |
| 4.2.2 | 🟩    | Loss function unit test   | **8**    | Test `compute_loss` on a handcrafted mini-example where the correct answer is known analytically. E.g.: reward=1, done=True → target=1 regardless of next Q-value. Confirms implementation correctness.                                                   |

---

---
# **5. Exploration Strategy (ε-Greedy)**
*How the agent balances trying new things vs exploiting what it knows.*

---
## 5.1 Theory & Implementation

| #     | Type | Task                              | Priority | Description                                                                                                                                                                     |
| ----- | ---- | --------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 5.1.1 | 🟦    | Exploration-exploitation tradeoff | **9**    | Explain the dilemma. Why purely greedy fails early in training (Q-values uninformed). Why purely random never improves. ε-greedy as the practical solution.                     |
| 5.1.2 | 🟩    | `EpsilonGreedy` class or function | **9**    | Takes `eps_start`, `eps_end`, `eps_decay`. Method `select_action(state, network)`: with prob ε returns random action, else argmax Q. Method `step()` to decay ε.                |
| 5.1.3 | 🟩    | **Plot: ε decay schedule**        | **7**    | Plot ε vs training step for chosen decay schedule (linear or exponential). Annotate the exploration→exploitation transition point. Reader immediately understands the schedule. |

---

---
# **6. Basic DQN Training Loop**
*The core training logic for Task 1: online, no replay, no target network.*

---
## 6.0 Data Structures & Plot Scaffolding ⭐
*Build and test all data structures and visualizations BEFORE running any real experiment. This prevents losing training time to data bugs.*

| #     | Type | Task                                             | Priority | Description                                                                                                                                                                                                                         |
| ----- | ---- | ------------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.0.1 | 🟦    | Strategy explanation                             | **8**    | Explain the "scaffolding first" approach: define the `results` dict schema, build all plot/table functions with dummy data, confirm they work, only then train.                                                                     |
| 6.0.2 | 🟩    | `results` dict schema                            | **9**    | Define: `results = {"runs": []}`. Each run dict: `{name, config, episode_scores, episode_losses, episode_qvalues, episode_epsilons, episode_steps, episode_crossings, episode_collisions, action_distribution, training_time_sec}`. |
| 6.0.3 | 🟩    | `save_results()` / `load_results()` helpers      | **8**    | Save/load `results` as JSON (crash-safe). Call `save_results()` after every training run.                                                                                                                                           |
| 6.0.4 | 🟩    | **Dummy data generation + all plot/table tests** | **9**    | Generate synthetic run data with `np.random`. Call every plot and table function. Confirm all work correctly before any real training. Fix any issues here.                                                                         |



| #     | Type | Task                        | Priority | Description                                                                                                                                         |
| ----- | ---- | --------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.1.1 | 🟩    | `preprocess_obs()` function | **9**    | Takes raw uint8 obs (54,39,3) → extracts R-channel `obs[:,:,0]` → float32 → /255 → returns (54,39) array. Called at every env step.                 |
| 6.1.2 | 🟩    | `FrameStack` buffer class   | **8**    | Maintains deque of N=4 frames. `reset(obs)` fills all slots. `push(obs)` adds new, pops oldest. `get()` returns stacked tensor (4,54,39) on device. |

---
## 6.2 Heuristic Data Collection & BC Pre-training

| #     | Type | Task                             | Priority | Description                                                                                                                                                                                                                                             |
| ----- | ---- | -------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.2.1 | 🟦    | BC pre-training motivation       | **8**    | Explain: before RL, run the heuristic to collect (RGB_obs, action) pairs and train the DQN as a classifier (`CrossEntropyLoss`, label_smoothing=0.1). Gives good weight initialization so RL starts from a meaningful point rather than random weights. |
| 6.2.2 | 🟩    | Heuristic data collection + save | **9**    | Run 10 heuristic episodes (~6000 transitions). Save to `heuristic_data.npz`. Print stats (mean score, coverage). Loading from disk avoids re-running.                                                                                                   |
| 6.2.3 | 🟩    | BC training loop                 | **8**    | Load data. Train DQN with `CrossEntropyLoss(label_smoothing=0.1)` on (obs→action) pairs for N epochs. Plot BC training loss. Save BC-initialized checkpoint.                                                                                            |
| 6.2.4 | 🟩    | BC sanity check                  | **7**    | Run 3 greedy eval episodes with BC-initialized DQN (no RL yet). Compare to random baseline. Shows BC already yields meaningful behavior before any RL.                                                                                                  |

---
## 6.3 Training Loop

| #     | Type | Task                             | Priority | Description                                                                                                                                                                                                                                                                |
| ----- | ---- | -------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6.3.1 | 🟦    | Training loop design explanation | **9**    | Pseudocode first: reset env → init frame stack → BC-initialized weights → loop: select action (ε-greedy) → step env → compute loss (Bellman, **every single step**) → backprop → clip grads → decay ε → append metrics → repeat. Reader understands the flow before code.  |
| 6.3.2 | 🟩    | **`train_basic_dqn()` function** | **10**   | Full training loop. **Updates every step** (no replay — intentional; this is the unstable baseline). Tracks per-episode: score, mean_loss, mean_qvalue, epsilon, crossings, collisions, action_counts, duration. Appends to `results` dict. Saves JSON after each episode. |
| 6.3.3 | 🟩    | Optimizer setup                  | **8**    | Adam, LR=1e-4. Gradient clipping `clip_grad_norm_(params, max_norm=10)`. Explain why clipping is essential without a target network (prevents gradient explosions).                                                                                                        |
| 6.3.4 | 🟩    | Model checkpoint saving          | **7**    | Save best model (by score) and final model. Consistent filename: `task1_dqn_{run_name}.pth`.                                                                                                                                                                               |

---
## 6.4 Training Execution

| #     | Type | Task                                         | Priority | Description                                                                                                                                                  |
| ----- | ---- | -------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 6.4.1 | 🟩    | **Run training (experiment A: base config)** | **10**   | Execute training with default hyperparams (LR=1e-4, γ=0.99, exponential ε-decay). Print running metrics. Log start/end time. Append to `results`. Save JSON. |
| 6.4.2 | 🟦    | Training observations                        | **7**    | Markdown cell written after observing the training run. Comment on what happened: learning trend, instability signs, loss behavior. Part of the narrative.   |



---

---
# **7. Results & Analysis**
*Visualize, interpret, and critique the training results. This is where the grade is won or lost.*

---
## 7.1 Learning Curves

| #     | Type | Task                              | Priority | Description                                                                                                                                                                                               |
| ----- | ---- | --------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.1.1 | 🟩    | **Plot: Score vs episode**        | **10**   | Line chart of episode score over training. Add rolling mean (window=20). X-axis = episodes, Y-axis = score. This is the primary learning curve.                                                           |
| 7.1.2 | 🟩    | **Plot: Loss vs step**            | **9**    | Training loss over time. Helps diagnose divergence, instability, or convergence.                                                                                                                          |
| 7.1.3 | 🟩    | **Plot: ε vs episode**            | **8**    | Overlay the ε decay on the score curve (dual y-axis). Shows the exploration→exploitation transition directly correlated with score trend.                                                                 |
| 7.1.4 | 🟩    | **Plot: Q-value evolution**       | **8**    | Track mean Q-value over eval episodes during training. Shows whether estimates grow sensibly or diverge. Also plot predicted max Q vs actual cumulative return per episode — catches overestimation bias. |
| 7.1.5 | 🟩    | Smoothed multi-metric dashboard   | **7**    | 2×2 subplot: score, loss, ε, Q-value — all vs episode. The single-glance training summary figure.                                                                                                         |
| 7.1.6 | 🟩    | **Plot: Action distribution**     | **7**    | % of up vs down actions per episode window over training. Shows if policy converges to a preferred action or stays near 50/50 (random).                                                                   |
| 7.1.7 | 🟩    | **Plot: Steps per crossing**      | **6**    | Average steps between crossing events per episode. Shows if agent navigates more efficiently over time.                                                                                                   |
| 7.1.8 | 🟩    | **Plot: Success/Collision ratio** | **8**    | Crossings vs collisions per episode over training. Direct game-performance metric.                                                                                                                        |

---
## 7.2 Evaluation on Held-Out Episodes

| #     | Type | Task                                | Priority | Description                                                                                                                                                                 |
| ----- | ---- | ----------------------------------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.2.1 | 🟩    | **Greedy evaluation function**      | **10**   | `evaluate_agent(network, env, n_episodes, device)`: runs with ε=0 (pure greedy), returns list of scores. Used for all final comparisons.                                    |
| 7.2.2 | 🟩    | **Evaluate trained DQN**            | **10**   | Run greedy eval on trained model. 5–10 episodes. Print mean/std/max score.                                                                                                  |
| 7.2.3 | 🟩    | **Plot: Eval episode trajectories** | **8**    | Plot cumulative reward per step for 3 eval episodes. Shows what the trained agent actually does.                                                                            |
| 7.2.4 | 🟩    | **Table: Final results summary**    | **10**   | Table with columns: Agent, Mean Score, Std Score, Max Score, Mean Crossings, Mean Collisions, Training Time. Rows: Random, Heuristic, Basic DQN. The key deliverable table. |

---
## 7.3 Ablation: Hyperparameter Sensitivity

| #     | Type | Task                            | Priority | Description                                                                                                                                                                   |
| ----- | ---- | ------------------------------- | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.3.1 | 🟩    | **LR ablation**                 | **7**    | Train with 2–3 learning rates (e.g., 1e-2, 1e-3, 1e-4). Plot score curves on same axes. Show which LR is best and why instability occurs at high LR.                          |
| 7.3.2 | 🟩    | **γ (discount) ablation**       | **6**    | Test γ ∈ {0.9, 0.95, 0.99}. Show how γ affects long-term planning (higher γ → values future crossings more).                                                                  |
| 7.3.3 | 🟩    | **ε decay schedule comparison** | **7**    | Linear vs exponential decay. Plot score curves. Discuss tradeoff: fast decay → less exploration but faster convergence; slow decay → more exploration but slower convergence. |
| 7.3.4 | 🟦    | Ablation conclusions            | **7**    | Summarize findings. Which config is selected as the best Task 1 baseline, and why? These conclusions feed directly into Task 2.                                               |

---
## 7.4 Analysis: Why Task 1 is Unstable

| #     | Type | Task                             | Priority | Description                                                                                                                                                                                                                                                                                        |
| ----- | ---- | -------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 7.4.1 | 🟦    | **Instability analysis**         | **10**   | Key theoretical section. Explain: (1) without replay, consecutive transitions are highly correlated → gradient updates are biased; (2) without a target network, targets change after every update → "chasing a moving target". Motivate Task 2. This section is part of the presentation content. |
| 7.4.2 | 🟩    | **Plot: Loss variance analysis** | **8**    | Show the high variance/spikiness of the loss curve. Compare to a smoothed version. Quantify instability (e.g., std of loss in sliding windows). Concrete evidence for the instability argument.                                                                                                    |

---

---
# **8. Efficiency Metrics**
*Graded: score/training_time and Q-value/training_time.*

---

| #   | Type | Task                               | Priority | Description                                                                                                    |
| --- | ---- | ---------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| 8.1 | 🟩    | **Efficiency metrics computation** | **9**    | Compute `score_per_hour = mean_score / training_hours` and `qval_per_hour`. Print clearly formatted.           |
| 8.2 | 🟩    | **Table: Efficiency summary**      | **8**    | Table: Agent, Mean Score, Training Time (h), Score/h, Q-val/h. Will be extended in Task 2 to show improvement. |

---

---
# **9. Local Evaluation & Submission (Codabench Prep)**
*Test the agent under the exact Codabench conditions. Submit early to validate format BEFORE wasting time training.*

---

| #   | Type | Task                               | Priority | Description                                                                                                                                                                                                                                                                                                                                                                                        |
| --- | ---- | ---------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 9.1 | 🟦    | Evaluation protocol                | **8**    | Explain that Codabench creates env with `include_semantic_info=False`. The agent must expose `select_action(obs)` accepting only the RGB frame. Critically: `select_action` is called **one frame at a time** — the Agent class must maintain its own internal FrameStack across calls.                                                                                                            |
| 9.2 | 🟩    | **`Agent` class implementation**   | **10**   | Wraps the trained DQN. **Crucially: maintains its own `FrameStack` internally** (deque of 4 R-channel frames). Constructor: loads checkpoint, initializes empty FrameStack. `reset()` method: clears frame buffer (call at episode start). `select_action(obs)` method: extract R-channel → push to FrameStack → if buffer not full yet, pad with zeros → forward pass → argmax. No semantic info. |
| 9.3 | 🟩    | Local evaluation run               | **9**    | `env = SpaceRaceEnv(obs_mode="rgb", include_semantic_info=False)`. Run agent for 3 episodes (calling `agent.reset()` between episodes). Print scores. Confirm the agent works in eval mode.                                                                                                                                                                                                        |
| 9.4 | �    | **Early submission format test** ⭐ | **10**   | **Do this as soon as Agent class exists (even with random/untrained weights).** Create submission zip: `agent.py` (Agent class) + checkpoint file. Submit to Codabench just to verify format is accepted. Catch format errors early — before investing hours in training. Note: this can be a random-weight agent; the goal is only format validation.                                             |
| 9.5 | 🟦    | Submission notes                   | **6**    | Notes: how to run `local_eval.py`, zip structure required, where checkpoint lives, how to select best submission on the leaderboard.                                                                                                                                                                                                                                                               |

---

---
# **10. Conclusions & Next Steps**
*Wrap up Task 1 findings and set the stage for Task 2.*

---

| #    | Type | Task                                     | Priority | Description                                                                                                                                                                                                                                                  |
| ---- | ---- | ---------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 10.1 | 🟦    | **Task 1 conclusions**                   | **10**   | Summary of: architecture chosen, loss function, training results, heuristic comparison. Key finding: basic DQN learns *something* but is unstable. State the performance gap vs heuristic.                                                                   |
| 10.2 | 🟦    | **Limitations & motivations for Task 2** | **9**    | List the problems observed: correlated updates, moving targets, data inefficiency (each experience used once). Connect each problem to the Task 2 solution (replay buffer → correlation, target network → moving targets). This is a great narrative bridge. |
| 10.3 | 🟦    | Key hyperparameters selected             | **7**    | Summary table of all final hyperparams used for Task 1. Architecture details, optimizer, ε schedule, γ, etc. Reproducibility record.                                                                                                                         |

---

## ✅ Resolved Decisions (formerly Open Doubts)

### D1 — Frame Stacking ✅ RESOLVED
**4 consecutive frames, no skip.** Input shape: (4, 54, 39) in PyTorch.

### D2 — Preprocessing: Grayscale vs R-channel ✅ RESOLVED
**R-channel only.** Grayscale is bad for this game (debris≈182, ship≈185 — nearly identical). R-channel: background=5, ship=90, debris=230 — perfect separation. Pipeline: `obs[:,:,0].astype(float32) / 255.0`.

### D3 — ε decay schedule ✅ RESOLVED
**Exponential decay as default for Task 1.** Linear vs exponential comparison is part of the Task 1 ablation (section 7.3.3). ε-greedy vs Boltzmann comparison is Task 3.

### D4 — Network architecture depth ✅ RESOLVED
**2 conv layers for Task 1** (stability first): Conv1(8×8,s=4,16f) → Conv2(4×4,s=2,32f) → Flatten(480) → FC1(128) → FC2(2). ~140K params. No BN, no Dropout. Task 1 ablation: also test 3-layer variant.

### D5 — Online update frequency ✅ RESOLVED
**Update every single step** (Option A). Most honest "no replay" implementation. The instability this produces is exactly what we need to document and motivate Task 2.

