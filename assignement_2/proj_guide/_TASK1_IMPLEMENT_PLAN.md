# Task 1 — Horizontal Implementation Plan

How to build Task 1 step by step.
Each step is: write notebook cell → implement src/ function → run test → confirm output.
Never skip ahead. Each step must be green before the next one starts.

Reference style: `reinforcement_q_learning.ipynb` (DQN CartPole) — follow its markdown+code rhythm.
Lab pattern for imports/setup: `ex_colab_set_up.ipynb` — FAST_RUN flag, autoreload, IN_COLAB guard.

---

## Step 0 — Notebook skeleton + config

**Notebook cells to write (section 0):**
- Cell 0.1: FAST_RUN flag + all constants (TASK_NAME, paths, hyperparams)
  - `N_EPISODES = 5 if FAST_RUN else 500`
  - `INPUT_CHANNEL = 0`, `N_FRAMES = 4`, `OBS_H = 54`, `OBS_W = 39`
  - `GAMMA = 0.99`, `LR = 1e-4`, `GRAD_CLIP = 10.0`
  - `EPS_START = 1.0`, `EPS_END = 0.05`, `EPS_DECAY_STEPS = 50000`
  - Heuristic mode constants: `HEURISTIC_NAIVE = "naive"`, `HEURISTIC_ENHANCED = "enhanced"`, `HEURISTIC = HEURISTIC_ENHANCED`
- Cell 0.2: imports (torch, numpy, matplotlib, collections.deque, json, pathlib)
  - autoreload + IN_COLAB guard (same as ex_colab_set_up.ipynb)
- Cell 0.3: device check `device = torch.device("cuda" if ... else "cpu")`
- Cell 0.4: path setup — `ROOT`, `OUTPUTS_DIR`, `CHECKPOINTS_DIR`, `PLOTS_DIR`, `RESULTS_PATH`
  - `Path(...).mkdir(parents=True, exist_ok=True)` for each

**No config.py** — all constants stay in cell 0.1.
Add `requirements.txt` to `assignement_2/` root (torch, numpy, matplotlib + SpaceRace env package). Cell 0.2 does `%pip install -r requirements.txt -q` when `IN_COLAB`.

**Test:** run cells, confirm dirs exist, confirm device prints.

---

## Step 1 — Environment EDA

**Notebook cells to write (section 1):**
- Cell 1.1: `make_env()` call, `env.reset()`, print obs shape + dtype + action space
- Cell 1.2: render 3 frames at different timesteps, show with `plt.imshow`
- Cell 1.3: pixel analysis — extract R, G, B channels; print mean pixel values for bg / ship / debris
  - confirm R-channel separation: bg≈5, ship≈90, debris≈230
  - confirm grayscale fails (debris≈ship≈182)
- Cell 1.3.5: reward scale analysis — run 1 episode with random policy, log rewards per step
  - categorise: movement_reward (small, per step) vs crossing_reward (large, per crossing)
  - print fraction: movement rewards = ~77% of total reward
- Cell 1.4: print episode length (truncated at 60s), count crossings in random episode

**src/ to implement:**
- `src/env/env_factory.py` → `make_env(difficulty=0, obs_mode="rgb")` + constants

**Test:** confirm `obs.shape == (54, 39, 3)`, `obs.dtype == uint8`, `env.action_space.n == 2`.

---

## Step 2 — Preprocessing + FrameStack

**Notebook cells to write (section 1 continued / section 3.1.4):**
- Cell 1.5: show R-channel extraction visually — `fig, axes` side by side: raw RGB vs R-channel
- Cell 3.1.4 (dim verification): print tensor shape through preprocess + FrameStack
  - `stack.shape == (4, 54, 39)` — confirm in output

**src/ to implement:**
- `src/env/preprocess.py`:
  - `preprocess(obs) -> np.ndarray shape (54,39) float32` — `obs[:,:,0].astype(np.float32) / 255.0`
  - `FrameStack(n_frames, h, w)`:
    - `__init__`: `self.frames = deque(maxlen=n_frames)`, zeros init
    - `reset()`: fill deque with n_frames zero frames
    - `push(frame)`: append frame (shape h,w)
    - `get() -> np.ndarray shape (n_frames,h,w)`: `np.stack(list(self.frames))`

**Test (`src/env/__preprocess_test.py`):**
```python
# shape and dtype
obs = np.random.randint(0, 255, (54, 39, 3), dtype=np.uint8)
f = preprocess(obs)
assert f.shape == (54, 39), f"got {f.shape}"
assert f.dtype == np.float32
assert f.max() <= 1.0

# FrameStack sliding window
fs = FrameStack(4, 54, 39)
fs.reset()
for _ in range(6):
    fs.push(np.ones((54, 39), dtype=np.float32))
s = fs.get()
assert s.shape == (4, 54, 39), f"got {s.shape}"

# reset clears to zeros
fs.reset()
assert fs.get().sum() == 0.0

print("preprocess tests passed")
```
Run test before next step.

---

## Step 3 — DQN Architecture

**Notebook cells to write (section 3):**
- Cell 3.1: markdown cell — architecture diagram (ASCII or description)
- Cell 3.2: instantiate `DQNNet`, print `summary` (manual: print each layer + param count)
- Cell 3.1.4: dimension verification cell
  ```python
  dummy = torch.zeros(1, 4, 54, 39)
  out = net(dummy)
  assert out.shape == (1, 2), f"got {out.shape}"
  # Print intermediate shapes — computed: (16,12,8) and (32,5,3) → flatten=480
  # H_out = floor((H_in - kernel) / stride) + 1
  # Conv1: floor((54-8)/4)+1=12, floor((39-8)/4)+1=8 → (16,12,8)
  # Conv2: floor((12-4)/2)+1=5,  floor((8-4)/2)+1=3  → (32,5,3)
  x = net.conv1(dummy)  # expect (1, 16, 12, 8)
  x = net.conv2(x)      # expect (1, 32, 5, 3)
  print("flatten dim:", x.numel() // x.shape[0])  # expect 480 = 32*5*3
  ```

**src/ to implement:**
- `src/agent/dqn.py` → `DQNNet(n_frames=4, h=54, w=39, n_actions=2)`:
  ```python
  # Conv1: in=n_frames, out=16, kernel=8x8, stride=4, padding=0
  # Conv2: in=16, out=32, kernel=4x4, stride=2, padding=0
  # Flatten -> FC1(128, ReLU) -> FC2(n_actions)
  # He init on all layers
  ```

**Test (`src/agent/__agent_test.py`):**
```python
net = DQNNet()
dummy = torch.zeros(1, 4, 54, 39)
out = net(dummy)
assert out.shape == (1, 2)
assert not torch.isnan(out).any()
print("DQNNet tests passed")
```

**!!!!! Decision checkpoint:** After cell 3.1.4 — confirm actual flatten dim from printed output.
If not 480, adjust FC1 input size. Do NOT hardcode 480 until verified by running the cell.

---

## Step 4 — Q-Learning Loss

**Notebook cells to write (section 4):**
- Cell 4.1: markdown — Bellman equation, MSE loss formula (use KaTeX in markdown cell)
- Cell 4.2: `compute_loss()` smoke test — pass random (s,a,r,s',done) single tuple, confirm scalar output

**src/ to implement:**
- `src/training/loss.py` → `compute_loss(net, s, a, r, s_next, done, gamma, device)`:
  - s shape: `(1, 4, 54, 39)` tensor
  - `q_values = net(s)` → shape (1, 2); `q_val = q_values[0, a]` — indexing preserves grad_fn, backward works
  - `with torch.no_grad(): q_next = net(s_next).max().item()`
  - `target_val = r + gamma * q_next * (1 - done)`
  - `target = torch.tensor([[target_val]], device=device)`  # explicit (1,1) shape
  - `loss = F.mse_loss(q_val.unsqueeze(0).unsqueeze(0), target)`
  - return loss

**Test:**
```python
net = DQNNet()
s = torch.zeros(1, 4, 54, 39)
loss = compute_loss(net, s, 0, 1.0, s, False, 0.99, "cpu")
assert loss.ndim == 0           # scalar
assert torch.isfinite(loss)
print("loss tests passed")
```

---

## Step 5 — Epsilon-Greedy Exploration

**Notebook cells to write (section 5):**
- Cell 5.1: markdown — epsilon decay formula
- Cell 5.2: plot epsilon schedule over N steps (FAST_RUN: 1000 steps, real: 50000)

**src/ to implement:**
- `src/training/epsilon.py` → `EpsilonScheduler(start, end, decay_steps)`:
  - `step() -> float`: `eps = end + (start-end) * exp(-steps_done / decay_steps)`; increment counter
  - `current` property

**Test:** schedule goes from ~1.0 to ~0.05 over decay_steps; never below end.

---

## Step 6 — Heuristic + BC Pre-Training

**Notebook cells to write (section 2 and 6.2):**
- Cell 2.1: markdown — enhanced heuristic description (3-row lookahead + lateral column scan)
- Cell 2.2: run heuristic for 3 episodes, print mean reward — confirm beats random
- Cell 2.3: visualise heuristic decisions — show frame with chosen action annotated
- Cell 6.2.1: `collect_heuristic_data(n_episodes=50 if not FAST_RUN else 3)` → save to `heuristic_data.npz`
  - each sample: `(state_stack (4,54,39), action int)`
  - target: ~6000 transitions
- Cell 6.2.2: load data, print shapes, class balance
- Cell 6.2.3: BC training loop (CrossEntropyLoss label_smoothing=0.1, Adam lr=1e-3, 10 epochs)
  - print train accuracy per epoch
- Cell 6.2.4: evaluate BC accuracy on held-out 20% — target >80%

**src/ to implement:**
- `src/heuristic/heuristic.py`:
  - Mode constants at top of file: `HEURISTIC_NAIVE = "naive"`, `HEURISTIC_ENHANCED = "enhanced"`
  - `heuristic_action(info, mode=None) -> int`:
    - `mode=None` defaults to `HEURISTIC_ENHANCED` (best available)
    - `info` is the dict from `env.step()` — uses `info["semantic_obs"]` (shape 18×13×3, available only during training)
    - `HEURISTIC_NAIVE`: check 1 row above ship's current row; if clear → action=0 (up); else → action=1 (down/wait)
    - `HEURISTIC_ENHANCED`: (1) find ship's row from semantic channel 0; (2) scan N=3 rows ahead in ship's column for debris (channel 1); (3) if clear → up; (4) if blocked → scan adjacent rows (up/down) for nearest clear row; (5) move toward it. Ship moves **vertically** — action 0=up, action 1=down. Debris moves **horizontally**. Heuristic decides the right vertical row to be in to pass through gaps.
    - **!!!!! Clarify action semantics before coding: test action=0 and action=1 in EDA cell 1.1.3 to confirm which is up**
- `src/heuristic/bc_trainer.py`:
  - `collect_heuristic_data(env, n_episodes, heuristic_mode=None) -> (states, actions)`:
    - instantiates its own `FrameStack` internally — no external state passed in
    - calls `fs.reset()` at the start of each episode — no cross-episode corruption
    - calls `heuristic_action(info, mode=heuristic_mode)` at each step
    - returns numpy arrays: states shape (N, 4, 54, 39), actions shape (N,)
  - `bc_pretrain(net, states, actions, epochs, lr, label_smoothing) -> net`

**Test (`src/heuristic/__heuristic_test.py`):**
```python
# heuristic returns valid action without crash
# build a minimal fake info dict with semantic_obs zeros (no obstacles)
import numpy as np
semantic = np.zeros((18, 13, 3), dtype=np.float32)
semantic[9, 6, 0] = 1.0  # ship at row 9, col 6
info = {"semantic_obs": semantic}

from heuristic import heuristic_action, HEURISTIC_NAIVE, HEURISTIC_ENHANCED
for mode in (HEURISTIC_NAIVE, HEURISTIC_ENHANCED, None):
    a = heuristic_action(info, mode=mode)
    assert a in (0, 1), f"mode={mode} returned {a}"

print("heuristic tests passed")
```

---

## Step 7 — Agent Class

**Notebook cells to write (section 9.2):**
- Cell 9.2: paste full Agent class — this is what Codabench calls
  ```python
  class Agent:
      def __init__(self, checkpoint_path=None):
          self.net = DQNNet(...)
          if checkpoint_path: self.net.load_state_dict(torch.load(...))
          self.net.eval()
          self._stack = FrameStack(N_FRAMES, OBS_H, OBS_W)

      def reset(self):
          self._stack.reset()

      def get_state(self):
          """Public accessor — training loop reads current stack without touching internals."""
          return self._stack.get()   # (N_FRAMES, H, W) float32 numpy

      def select_action(self, obs):  # obs: (54,39,3) uint8
          frame = preprocess(obs)    # (54,39) float32
          self._stack.push(frame)
          state = torch.tensor(self._stack.get()).unsqueeze(0)  # (1,4,54,39)
          with torch.no_grad():
              return int(self.net(state).argmax(dim=1).item())
  ```
- Cell 9.4 (early submission test — run this IMMEDIATELY after Agent class exists):
  ```python
  # Early submission format test — random weights, just check no crash
  agent = Agent(checkpoint_path=None)
  obs, _ = env.reset()
  agent.reset()
  for _ in range(50):
      action = agent.select_action(obs)
      assert action in (0, 1), f"bad action: {action}"
      obs, _, terminated, truncated, _ = env.step(action)
      if terminated or truncated:
          obs, _ = env.reset()
          agent.reset()
  print("submission format test passed")
  ```

**src/ to implement:**
- `src/agent/agent.py` → `Agent` class (same as notebook cell but importable)

**!!!!! Run cell 9.4 now. Do not wait until training is done.**

---

## Step 8 — Training Loop (Scaffolding First)

**Notebook cells to write (section 6):**
- Cell 6.0: scaffolding run — 5 episodes, print per-episode reward, no logging, no saving
  - confirm training loop works end-to-end before any optimisation
  - use FAST_RUN=True config
- Cell 6.3: full training loop
  ```python
  # per episode:
  #   env.reset() -> obs; agent.reset()
  #   step loop:
  #     s = agent.get_state()           # (4,54,39) — read state BEFORE action
  #     action = eps_scheduler.step() + agent.select_action(obs)  # select_action pushes new frame
  #     obs_next, reward, terminated, truncated, _ = env.step(action)
  #     s_next = agent.get_state()      # (4,54,39) after push — read via public method
  #     loss = compute_loss(net, s, action, reward, s_next, done, gamma, device)
  #     optimizer.zero_grad(); loss.backward()
  #     torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
  #     optimizer.step()
  #     obs = obs_next
  #     if done: break
  ```
- Cell 6.4: execution cell — loop over experiments list, call `run_episode`, accumulate results

**src/ to implement:**
- `src/training/train.py` → `run_episode(env, agent, net, optimizer, eps_scheduler, cfg, device) -> dict`:
  - returns `{"episode": int, "total_reward": float, "loss_mean": float, "epsilon": float, "steps": int}`

**Test (`src/training/__training_test.py`):**
```python
# run_episode returns dict with expected keys
env = make_env()
agent = Agent()
net = DQNNet()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
eps = EpsilonScheduler(1.0, 0.05, 1000)
metrics = run_episode(env, agent, net, opt, eps, gamma=0.99, device="cpu")
for key in ("total_reward", "loss_mean", "epsilon", "steps"):
    assert key in metrics, f"missing key: {key}"
print("training tests passed")
```

---

## Step 9 — Results & Plots

**Notebook cells to write (section 7 + 8):**
- Cell 7.1: `plot_learning_curve(rewards_per_episode)` — rolling mean ±std
- Cell 7.2: action distribution bar chart (% action 0 vs 1 over last N episodes)
- Cell 7.3: Q-value monitoring — log `net(state).max().item()` per episode, plot over training
  - check for overestimation (Q >> actual return)
- Cell 8.1: efficiency metrics — steps per crossing, episode length distribution

**src/ to implement:**
- `src/evaluation/metrics.py` → `episode_metrics(rewards, actions, crossings) -> dict`
- `src/evaluation/plots.py` → `plot_learning_curve()`, `plot_action_dist()`, `plot_q_values()`
- `src/evaluation/persistence.py` → `save_results(results, path)`, `load_results(path)`, `save_checkpoint(net, path)`

**Test (`src/evaluation/__evaluation_test.py`):**
```python
# save/load roundtrip
results = {"runs": [{"episode": 1, "total_reward": 10.0}]}
save_results(results, "/tmp/test_results.json")
loaded = load_results("/tmp/test_results.json")
assert loaded["runs"][0]["total_reward"] == 10.0
print("evaluation tests passed")
```

---

## Step 10 — Experiment Runs

**Notebook cells to write (section 6.4):**
- Cell 6.4.0: smoke test — 1 experiment, FAST_RUN=True, 5 episodes, confirm results dict populated
- Cell 6.4.1: Experiment A — base DQN, no BC pre-training, N_EPISODES=500
- Cell 6.4.2: Experiment B — with BC pre-training, compare learning curve vs A
- Cell 6.4.3: ablations — vary eps_decay, lr, n_frames (pick 2-3 ablations max)
- Cell 6.4.4: best model — reload best checkpoint, run 10 evaluation episodes, print mean reward

Results pattern:
```python
results = {"runs": []}
# after each experiment:
results["runs"].append(metrics)
save_results(results, RESULTS_PATH)
```

---

## Step 11 — Conclusions + Submission

**Notebook cells to write (section 9 + 10):**
- Cell 9.1: load best checkpoint into Agent
- Cell 9.2: Agent class (already written in Step 7 — just confirm it still works)
- Cell 9.3: run Agent for 10 episodes, print crossing counts per episode
- Cell 9.4: early submission format test (already written in Step 7)
- Cell 10.1: conclusions — 3-4 bullet points: what worked, what didn't, what Task 2 will fix

**Final submission checklist:**
- [ ] `submission/agent.py` — self-contained, no src/ imports, checkpoint path hardcoded
- [ ] `submission/checkpoint.pth` — best weights
- [ ] Early submission test passes (cell 9.4)
- [ ] `task1_results.json` saved
- [ ] All plots saved to `outputs/plots/`

---

## Grading Alignment

| Component                       | Weight | Covered in Step    |
| ------------------------------- | ------ | ------------------ |
| Architecture (DQN CNN)          | 5%     | Step 3             |
| Loss function (Bellman MSE)     | 10%    | Step 4             |
| Training loop                   | 5%     | Step 8             |
| Heuristic + BC pre-training     | 10%    | Step 6             |
| Presentation / notebook clarity | 3%     | All markdown cells |
| Peer review                     | 2%     | N/A                |

---

## Key Flags + Gotchas

- **!!!!! Action semantics:** confirm action=0 vs action=1 direction in EDA cell 1.1.3 before writing heuristic.
- **!!!!! Flatten dim:** do not hardcode until cell 3.1.4 output confirms — expected 32×5×3=480.
- **FrameStack in Agent:** `select_action()` is called 1 frame at a time by Codabench. Agent owns FrameStack. `reset()` must be called at episode start. Use `agent.get_state()` (public) — never `agent._stack` directly from training code.
- **Heuristic uses `info`, not `obs_rgb`:** `heuristic_action(info, mode=HEURISTIC)` reads `info["semantic_obs"]` (privileged training-time data). BC data collection must pass `info` from `env.step()`.
- **Heuristic mode pattern:** constants `HEURISTIC_NAIVE / HEURISTIC_ENHANCED` defined in `heuristic.py`; notebook cell 0.1 sets `HEURISTIC = HEURISTIC_ENHANCED`. Pass `HEURISTIC` everywhere — one line to swap strategy.
- **`collect_heuristic_data` owns its FrameStack:** the function instantiates `FrameStack` internally and calls `reset()` at each episode boundary. No external FrameStack argument.
- **FAST_RUN flag:** all expensive cells guard on `if not FAST_RUN` — this is non-negotiable.
- **Every-step update:** Task 1 has no replay buffer. One gradient step per env step. Loss uses single (s,a,r,s') tuple.
- **BC pre-training:** train net weights BEFORE the RL loop, then continue training with RL. Do not freeze weights.
- **Reward scale:** movement reward dominates (77%). Do not interpret reward as crossings.
- **No config.py:** all hyperparams in notebook cell 0.1. Test files define their own inline defaults.
