# Project Structure — SpaceRace DQN (All 3 Phases)

## Overview

The notebook is the entry point and the story. All reusable logic lives in `src/`.
Notebook cells call `src/` functions; they do not contain reusable logic themselves.
Each phase (task) has its own notebook. `src/` is shared across all tasks.

---

## Folder Layout

```
assignement_2/
    requirements.txt             # pip install -r requirements.txt — works locally and on Colab
    tasks/
        task1/
            task1.ipynb          # Task 1 notebook — vanilla DQN, no replay
            outputs/
                checkpoints/     # .pth files per experiment (e.g. A_base.pth)
                results/         # task1_results.json, submission files
                plots/           # learning curves, action dist, Q-value plots
                heuristic_data.npz  # saved (state, action) transitions from heuristic
        task2/
            task2.ipynb          # Task 2 notebook — replay buffer + target network
            outputs/
                checkpoints/
                results/
                plots/
        task3/
            task3.ipynb          # Task 3 notebook — exploration strategies comparison
            outputs/
                checkpoints/
                results/
                plots/
    src/
        env/
            preprocess.py        # preprocess(obs) -> (54,39) float32; FrameStack class
            env_factory.py       # make_env() helper; env constants (OBS_SHAPE, N_ACTIONS)
            __preprocess_test.py
        agent/
            dqn.py               # DQNNet(n_frames, h, w, n_actions) — CNN architecture
            agent.py             # Agent: internal FrameStack, select_action(obs), get_state(), reset()
            __agent_test.py
        heuristic/
            heuristic.py         # heuristic_action(info, mode=None) — mode selects which heuristic;
                                 #   HEURISTIC_NAIVE / HEURISTIC_ENHANCED constants defined here;
                                 #   mode=None defaults to HEURISTIC_ENHANCED (best);
                                 #   notebook cell 0.1 sets HEURISTIC = HEURISTIC_ENHANCED to switch globally
            bc_trainer.py        # collect_heuristic_data(), bc_pretrain(net, data_path)
            __heuristic_test.py
        training/
            loss.py              # compute_loss(net, batch, gamma) — Bellman MSE, no replay
            train.py             # run_episode(env, agent, optimizer, cfg) -> metrics dict
            replay_buffer.py     # ReplayBuffer(capacity) — Task 2 onwards
            target_net.py        # update_target(policy_net, target_net, tau_or_hard) — Task 2
            epsilon.py           # EpsilonScheduler(start, end, decay_steps, mode)
            boltzmann.py         # BoltzmannSelector(temperature) — Task 3
            __training_test.py
        evaluation/
            metrics.py           # episode_metrics(rewards, actions, crossings) -> dict
            plots.py             # plot_learning_curve(), plot_action_dist(), plot_q_values()
            persistence.py       # save_results(results, path), load_results(path), save_checkpoint()
            __evaluation_test.py
        __init__.py
```

---

## What Lives Where

### In the Notebook (not in src/)
- Experiment loop: `for cfg in experiments: results = run_experiment(cfg)`
- Results accumulator dict + JSON save call
- All markdown narrative, plots rendered inline
- Hyperparameter sweeps and ablation tables
- Codabench Agent class (copied from `src/agent/agent.py` + self-contained)
- Early submission format test cell

### In src/ (reusable, importable)
- All classes and functions called from 2+ cells or tested independently
- Preprocessing pipeline (FrameStack is tricky — put it in src/ and test it)
- Network architecture (DQNNet)
- Loss computation
- Training loop function (returns metrics; notebook drives the loop)
- Heuristic logic and BC pre-training
- Replay buffer (Task 2+)
- Target network update (Task 2+)
- Exploration schedulers (Task 3+)
- Metrics, plots, persistence helpers

---

## Phase-by-Phase src/ Build-Up

### Task 1 — src/ additions
- `env/preprocess.py` — `preprocess()`, `FrameStack`
- `env/env_factory.py` — `make_env()`, constants
- `agent/dqn.py` — `DQNNet`
- `agent/agent.py` — `Agent` (owns FrameStack, select_action, get_state, reset)
- `heuristic/heuristic.py` — `heuristic_action(info, mode=None)` + mode constants
- `heuristic/bc_trainer.py` — `collect_heuristic_data()`, `bc_pretrain()`
- `training/loss.py` — `compute_loss()` (no replay: single (s,a,r,s',done) tuple)
- `training/train.py` — `run_episode()` (every-step update, no replay)
- `training/epsilon.py` — `EpsilonScheduler`
- `evaluation/metrics.py`, `plots.py`, `persistence.py`

### Task 2 — src/ additions (on top of Task 1)
- `training/replay_buffer.py` — `ReplayBuffer(capacity)`
- `training/target_net.py` — `update_target()`
- Update `training/train.py` to accept optional replay buffer + target net
- Update `agent/agent.py` if needed (same Agent, just different training driver)

### Task 3 — src/ additions (on top of Task 2)
- `training/boltzmann.py` — `BoltzmannSelector`
- Update `training/epsilon.py` to expose a common `ExplorationPolicy` interface
- Comparison harness in notebook (not in src/)

---

## Test Files

Each domain folder has one `__xxx_test.py`. Tests use plain `assert` statements.
Run with: `python src/env/__preprocess_test.py` etc. (no pytest needed).
All tests must pass before moving to the next notebook section.

Test coverage targets:
- `__preprocess_test.py`: FrameStack shape, dtype, reset, sliding window
- `__agent_test.py`: Agent.reset() clears stack, select_action returns int in {0,1}
- `__heuristic_test.py`: heuristic returns 0 or 1 on synthetic frames; no-obstacle case
- `__training_test.py`: loss value finite + scalar; run_episode returns dict with expected keys
- `__evaluation_test.py`: metrics dict has required keys; save/load results roundtrip

---

## No config.py

All hyperparameters and constants live in **notebook cell 0.1** — visible to the reader, easy to tweak for ablations. This includes the heuristic mode selector:

```python
# notebook cell 0.1 — heuristic constants (maps to heuristic.py mode strings)
HEURISTIC_NAIVE    = "naive"
HEURISTIC_ENHANCED = "enhanced"
HEURISTIC          = HEURISTIC_ENHANCED   # change this one line to switch globally
```

`src/` test files that need defaults define their own small inline constants — no shared config module needed.

---

## Outputs / Persistence Pattern

Same pattern as Assignment 1:
- `results = {"runs": []}` accumulator at top of notebook
- After each run: `results["runs"].append(run_metrics)`
- Save: `save_results(results, "outputs/results/task1_results.json")`
- Checkpoints: `torch.save(net.state_dict(), f"outputs/checkpoints/{cfg.run_id}.pth")`
- Plots: `fig.savefig(f"outputs/plots/{cfg.run_id}_curve.png")`

---

## Submission Pattern

The Codabench evaluator calls `agent.select_action(obs)` once per step, where `obs` is the raw (54,39,3) uint8 frame. The Agent must own its internal FrameStack and call `reset()` at episode start.

`submission/agent.py` is a self-contained copy of `src/agent/agent.py` with the checkpoint path hardcoded. No imports from `src/` — fully standalone.

```
submission/
    agent.py             # self-contained Agent class
    checkpoint.pth       # best trained weights
    requirements.txt     # only what Codabench needs
```

Early submission test (cell 9.4): run the agent for one episode with random weights and verify `select_action()` returns 0 or 1 at every step without crashing.
