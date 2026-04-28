# Project Plan — SpaceRace Deep Reinforcement Learning

## Big Picture

We are training a deep neural network agent to play **SpaceRace**: a 2-action game (move up / move down) where a spaceship navigates through horizontally-moving debris to reach the top of the screen as many times as possible in a 60-second round.

The agent **only sees RGB pixel frames** (54×39×3 uint8). It must learn, from raw pixels and scalar rewards, which direction to move. This is the classic Deep RL setup: raw sensory input → action selection via learned Q-values.

The project is structured in **3 incremental tasks**, each building on the previous one. The deliverable for each is both code and a cumulative presentation (each phase improves and extends the previous one).

---

## The 3 Tasks — What We Are Building

### Task 1 — Basic DQN (no Replay, no Target Net)
The simplest possible DQN: one CNN that takes the current frame and outputs Q-values for both actions. At each step, we do a forward pass, pick the action (ε-greedy), observe the reward, and immediately update the network with a single Bellman step. No memory of past transitions, no separate target network → very unstable, but it's the baseline.

We also build a **heuristic policy** that reads semantic game state (available in `info` at training time) to make smart decisions. This heuristic is not RL — it's hand-coded logic. We use it as a performance baseline and to understand the game, **not** to warm-start the buffer (no buffer in Task 1).

**Key challenge in Task 1**: Without a replay buffer the training signal is highly correlated and noisy. We expect unstable learning curves. The point is to document this and understand *why* the next improvements matter.

**Graded components**: Architecture (5%), Loss function (10%), Training loop + evaluation (5%), Heuristic (10%), Presentation (3%), Peer review (2%) → **35% total**

---

### Task 2 — Enhanced DQN (Experience Replay + Target Network)
We add the two classic DQN stabilizers:

1. **Replay Buffer** (≤10,000 transitions): Store `(s, a, r, s', done)` tuples and train on random mini-batches. This breaks temporal correlations and lets us reuse data. We test multiple buffer sizes (1, 100, 10,000) to observe the on-policy vs off-policy effect. We also warm-start the buffer using the heuristic policy from Task 1.
2. **Target Network**: A frozen copy of the Q-network, updated every ~100 steps. This makes the Bellman targets stationary during a training window, dramatically stabilizing learning.

Together these two changes replicate the original 2015 DeepMind DQN paper. We expect significantly smoother and higher-performing learning curves. The comparative analysis vs Task 1 is a key deliverable.

**Graded components**: Experience Replay (10%), Target Network (10%), Integration + Evaluation (5%), Presentation (3%), Peer review (2%) → **30% total**

---

### Task 3 — Exploration Strategies
With a stable DQN in place (Task 2), we isolate and study the **exploration strategy**:

1. **ε-Greedy with decay**: The standard. We test linear, exponential, and step-based decay schedules.
2. **Boltzmann (Softmax) Exploration**: Instead of hard ε-greedy, action probabilities follow softmax of Q-values divided by temperature T. High T → uniform exploration. Low T → greedy. We test different temperature schedules.

We run controlled experiments (same architecture, same buffer, same target network, only exploration differs) and do a thorough comparative analysis: learning speed, final score, stability, hyperparameter sensitivity.

**Graded components**: ε-Greedy (10%), Boltzmann (10%), Comparative Analysis (10%), Presentation (3%), Peer review (2%) → **35% total**

---

## Important Technical Details to Keep in Mind

### Observation & Preprocessing
- Input: `(54, 39, 3)` uint8 → must normalize to float32 in [0,1] before feeding the CNN
- The guide hints at **frame stacking** (2–4 frames) to give the network temporal information (e.g. debris direction, ship velocity). This is architecturally important and should be decided early.
- At evaluation on Codabench: `include_semantic_info=False` → the agent's `select_action(obs)` must work on RGB only, no semantic info.

### Network Architecture
- Input is small (54×39), so the CNN does not need to be large — smaller = more stable
- Output: 2 Q-values (one per action)
- Convolutional layers to capture spatial patterns + fully connected layers for Q-value head

### Training Constraints
- Each training run ≤ 4 hours
- Use `ticks_per_second=10` for faster simulation

### Evaluation Metrics (consistent across all tasks)
- **Score**: how many crossings per episode (primary metric)
- **Q-value**: average max Q-value (diagnostic metric)
- **Efficiency**: score / training_time, Q-value / training_time
- **Learning curves**: score and loss vs. steps/episodes

### Difficulty Progression
- Start at **difficulty 0** (deterministic debris, normal speed) for all tasks
- Optionally test difficulty 1–3 as bonus experiments once the agent is working

### Code Submission
- The agent must expose a `select_action(obs)` method that takes only the RGB frame
- Local evaluation: `local_eval.py`
- Codabench submission for competition ranking

---

## Cumulative Presentation Strategy
Each task's presentation **improves and extends** the previous one:
- Task 1: Architecture, loss, training loop, heuristic baseline
- Task 2: Add replay buffer and target network sections + comparative plots vs Task 1
- Task 3: Add exploration strategy section + full comparative analysis

Keep all plots and tables reproducible from the notebooks.
