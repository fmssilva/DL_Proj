
read all these questions and tasks and notes. and give me the best answer to each one of them, and then mark each of them, one by one, with [DONE] so we can keep track of the ones we did already. 

---

## [DONE] Q1 — "Wait" / do-nothing action?

one question. we have actions up and down. should we consider also the "wait" or do nothing action? or according to the assignment guide we really can't use it?

> **ANSWER**: No wait action. The env action space is binary: 0=up, 1=down only. "Waiting" is achieved by pressing down — the ship moves away from debris, stalling progress. The guide itself confirms: "waiting is achieved by not moving or moving down to dodge." Stick with 2 actions.

---

## [DONE] Q2 — Frame stacking: 4 frames good?

in terms of frames to use, maybe lets use 4 for better "history track"? is it a good choice? or what criteria should guide that decision?

> **ANSWER**: Yes, 4 frames is the right choice — it is the original DQN (Mnih 2015) standard. For SpaceRace, debris moves horizontally so temporal context is needed to infer direction and speed. 4 consecutive frames at 10 ticks/sec = 0.4 seconds of history. Use consecutive frames, no skip (frame-skip=1). **Decision: 4 frames, no skip.**

---

## [DONE] Q3 — Single channel: grayscale vs G-channel vs R-channel?

in terms of each frame representation, if we convert to gray we still capture well the 3 types of colors for background, debris and ship? or do we even not need the conversion to gray and we can use just one channel?

Colors:
- Background: (5, 10, 20)
- Debris: (230, 180, 70)
- Ship: (90, 220, 250)

> **ANSWER**: **Use R-channel only.** Grayscale is terrible here: debris≈182, ship≈185 — nearly identical. G-channel: debris=180, ship=220 — still close. R-channel: background=5, ship=90, debris=230 — three perfectly separated values. Decision: `frame = obs[:,:,0].astype(np.float32) / 255.0` then stack 4 → input shape `(4, 54, 39)` in PyTorch (C,H,W). Resolves D1 and D2.

---

## [DONE] Q4 — Rewards: follow default? Need temporal bonus?

The default rewards are: +1.0 crossing, -0.25 collision, +0.02 moving up, -0.01 moving down.
Are we following that? Would a temporal reward help?

> **ANSWER**: Yes, use default rewards as-is. The +0.02/step-up is already a temporal incentive. Math: going straight up to crossing gives ~0.076/step. Oscillating gives 0.005/step — 15x less. No farming risk. No reward modification needed for any task.

---

## [DONE] Q5 — Episode termination: explain and collision handling?

- `terminated` is always `False` during normal gameplay
- `truncated` becomes `True` when the 60-second timer expires
- Collisions do **not** end the episode — the ship respawns at the bottom after a brief delay

We don't mark collision with a special done flag because the -0.25 reward is enough?

> **ANSWER**: `terminated=False` always (no hard game-over). `truncated=True` at 60s. `done = terminated or truncated` = `done = truncated`. Collisions are NOT terminal — ship respawns, episode continues. In the Bellman update at collision: `done=False` so we DO bootstrap from the respawn state. The -0.25 reward alone is sufficient to teach collision avoidance. No special flag needed.

---

## [DONE] Q6 — Constants in the CONSTANTS cell?

About the main constants like the difficulty level, lets put them all in the "CONSTANTS" cell in the beginning of the notebook.

> **ANSWER**: Confirmed. Already in the notebook plan at cell 0.1.1. Include: `SEED`, `DEVICE`, `OBS_MODE`, `DIFFICULTY`, `ROUND_TIME`, `TICKS_PER_SEC`, `GAMMA`, `LR`, `EPS_START`, `EPS_END`, `EPS_DECAY`, `N_EPISODES`, `EVAL_EVERY`, `N_FRAMES=4`, `INPUT_CHANNEL=0` (R channel index).

---

## [DONE] Q7 — Heuristic: what is it, behavioral cloning, saving data?

Explain what the heuristic is and how we use it. Can we do BC pre-training (collect (s,a) from heuristic, train as classifier, then switch to RL)? Should we save the data to disk? How many transitions?

> **ANSWER**:
>
> **What the heuristic is**: Hand-coded rule using `info["semantic_obs"]` — "if debris directly above → go down (dodge), else go up." Not RL, not neural network. Available only during training.
>
> **Two uses**: (1) Task 1 graded baseline (10%): run ~10 episodes, compare scores vs random. (2) Task 2 warm-start: pre-fill replay buffer with heuristic-generated transitions.
>
> **Behavioral Cloning (BC) pre-training**: Yes, allowed and smart. Collect (RGB_obs, action) pairs from heuristic episodes. Train DQN with `nn.CrossEntropyLoss(label_smoothing=0.1)` on the Q-value outputs (no extra head needed — softmax over 2 Q-values = classification). Then switch to Q-learning for RL. Label smoothing (0.1) because the heuristic is imperfect.
>
> **Save to disk**: Yes. Format: `np.savez_compressed('heuristic_data.npz', obs=..., actions=..., rewards=..., next_obs=..., dones=...)`. Target: **5,000–10,000 transitions** (~15-30 MB compressed). At 600 steps/episode x 10 episodes = 6000 transitions — perfect.

---

## [DONE] Q8 — Submission / Codabench: how does it work?

Where and how is submission done? Strategy?

> **ANSWER**: Codabench is an academic competition platform (like Kaggle for research). No special library — submit a zip with `Agent` class + checkpoint. At eval: `SpaceRaceEnv(obs_mode="rgb", include_semantic_info=False)`. `Agent.select_action(obs)` must be self-contained (preprocess R-channel + frame stack + /255 + forward + argmax). Strategy: (1) submit early with any working agent to validate format; (2) test every checkpoint locally with `local_eval.py` before submitting; (3) keep a separate "best checkpoint" file; (4) meet each phase deadline.

---

## [DONE] Q9 — Architecture: pooling, stride, kernels, filters, layers, FC, init, optim?

Decisions on the full CNN architecture.

> **ANSWER**:
>
> **No pooling** — position-sensitive game. Use strided convolutions for downsampling.
>
> **Task 1 architecture** (2 conv layers, ~140K params, stability first):
> - Conv1: kernel=8×8, stride=4, filters=16, ReLU → output (16, 12, 8)
> - Conv2: kernel=4×4, stride=2, filters=32, ReLU → output (32, 5, 3)
> - Flatten → 480
> - FC1: 480→128, ReLU
> - FC2: 128→2 (Q-values, no activation)
>
> **Intuition**: Conv1 detects large-scale debris fields + general ship zone. Conv2 refines to exact positions. FC128 = game-state vector. 2 Q-values out.
>
> **Frame stacking**: 4 consecutive frames, no skip. Input: (4, 54, 39).
> **Initialization**: He (Kaiming) — correct for ReLU.
> **Optimizer**: Adam, LR=1e-4.
> **Gradient clipping**: `clip_grad_norm_(params, max_norm=10)`.
> **Activation**: ReLU (except final output).
> **BatchNorm**: NO — destabilizes RL.
> **Dropout**: NO — adds stochasticity to Q-value estimates.
>
> **Task 1 ablation**: also test 3-layer variant (add Conv3: 3×3, stride=1, 64 filters) after base config works.

---

## [DONE] Q10 — Training loop structure: accumulator variable + JSON saving?

Is it better to have a persistent accumulator variable and run the training cell multiple times?

> **ANSWER**: Yes — accumulator dict + JSON saving:
> ```python
> # Cell A (run once): init
> results = {"runs": []}
>
> # Cell B (run multiple times): train + append
> run = train_basic_dqn(config)
> results["runs"].append(run)
> save_json(results, "task1_results.json")  # crash-safe
>
> # Cell C (reload anytime):
> results = load_json("task1_results.json")
> ```
> Each run dict: `{name, config, scores, losses, qvalues, epsilons, timestamps, training_time}`. All plots and tables read from `results["runs"]`. Adding a Section 6.0 "Data Structures & Plot Scaffolding" to the notebook plan.

---

## [DONE] Q11 — Plots/tables data structure first strategy?

Build all plots/tables with dummy data first, then run experiments?

> **ANSWER**: Yes — excellent strategy. Define `results` schema → build all plot functions → test with `np.random` dummy data → only then run training. Prevents losing 4h of training time to a plot bug. Adding Section 6.0 "Data Structures & Plot Scaffolding" to the notebook plan as a dedicated section before training.

---

## [DONE] Q12 — D3: ε decay for Task 1 or later?

D3 ε decay: exponential vs linear — is this for task 1 or later?

> **ANSWER**: ε-greedy is needed in Task 1. D3 belongs here: implement **exponential decay** as the default for Task 1, and run linear vs exponential as part of the Task 1 ablation (section 7.3.3). ε-greedy vs Boltzmann comparison is Task 3. D3 resolved: **exponential decay as default, linear in ablation.**

---

## [DONE] Q13 — D5: Online update every step vs every N?

D5 Online update frequency: every step vs every N steps. What does the assignment say? What is best?

> **ANSWER**: Assignment does not specify frequency. **Use every single step (Option A).** Reasons: (1) Most honest "no replay" implementation — Task 1 is meant to be the unstable baseline; (2) Accumulating N consecutive steps as pseudo-batch doesn't fix correlation; (3) The instability from every-step updates is the key evidence for Task 1 analysis and directly motivates Task 2. D5 resolved: **update every step**.

---

## [DONE] Q14 — Is ε-greedy (Section 5) for Task 1 or later?

Confirm in the assignment guide if ε-greedy should be implemented in Task 1 or is it for Task 3.

> **ANSWER**: ε-greedy is part of Task 1 — every DQN needs exploration. Task 3 is where we compare ε-greedy vs Boltzmann. Both Task 1 and Task 2 use ε-greedy. Section 5 stays in the Task 1 notebook.

---

## [DONE] Q15 — Reward clipping needed?

Reward clipping: scale to [-1,1]. But rewards are already [-1,1] so we don't need to do anything?

> **ANSWER**: Correct. Rewards are already in [-1, +1] by design. No clipping or scaling needed.

---

## [DONE] Q16 — Reward shaping loop concern (+0.02 oscillation farming)?

Does the +0.02 up / -0.01 down reward create a farming loop risk?

> **ANSWER**: Not a real concern. Math: oscillating = +0.005/step. Straight crossing = ~0.076/step. Crossings are 15x more valuable. Agent will naturally learn crossings dominate. Leave rewards as-is.

---

## [DONE] Q17 — Instability notes: tiny LR, TD-error watch?

Notes say: use LR 1e-4 or 5e-5, watch the loss for divergence ("Liar's Feedback Loop").

> **ANSWER**: Very applicable. Use **LR=1e-4** for Task 1. Watch TD-error (loss) — trending to infinity = the moving-target feedback loop (same network for prediction and target). Expected in Task 1 — document it. Gradient clipping (`max_norm=10`) is essential. "Liar's Feedback Loop" framing is excellent for Section 7.4 instability analysis.

---

## [DONE] Q18 — All hyperparameters to tune in Task 1?

What are all the hyperparams to tune in Task 1?

> **ANSWER**:
>
> **RL (ablations in section 7.3)**:
> - `LR`: try {5e-5, 1e-4, 5e-4}
> - `GAMMA`: try {0.9, 0.95, 0.99}
> - `EPS_DECAY`: fast/medium/slow decay schedules
>
> **Architecture (ablation: 2 vs 3 conv layers)**:
> - Conv filter counts
> - FC hidden size (128 vs 256)
> - `N_FRAMES`: fixed at 4
>
> **Fixed (not ablated)**: difficulty=0, round_time=60, ticks=10, input channel=R only.

---

## [DONE] Q19 — Extra metrics to track and show?

Beyond score and Q-value, what other metrics are nice to have?

> **ANSWER**:
> - **Q-value overestimation check**: predicted max Q vs actual cumulative return per episode
> - **Action distribution**: % up vs down per episode window — shows policy convergence
> - **Average steps per crossing**: shows if agent gets faster over time
> - **Loss std per sliding window**: quantifies instability directly — key for section 7.4
> - **Success/Collision ratio**: confirmed from notes
>
> These are all added to section 7 of the notebook plan.

---

## [DONE] Q20 — Practical checklist (obs mode, termination, wait action)?

Final checklist notes: obs mode, termination handling, wait action.

> **ANSWER**:
> ✅ `obs_mode="rgb"` for agent, `info["semantic_obs"]` only in heuristic
> ✅ `done = terminated or truncated` = effectively `done = truncated`
> ✅ "Wait" = action 1 (down). Heuristic uses this as the dodge action.
