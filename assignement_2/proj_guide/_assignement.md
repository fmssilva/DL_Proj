For this assignment, you will train a deep neural network to play the SpaceRace game by observing the game state. SpaceRace is a single-player environment inspired by the classic 1973 Atari arcade game, where you control a spaceship that must navigate through horizontally-moving debris to reach the top of the screen as many times as possible within a 60-second round.





## **Objective**

The main goal of this assignment is to provide you with hands-on experience in reinforcement learning. Your focus should be on experimenting with different scenarios, testing various approaches, and understanding what aids the agent's learning and how deep reinforcement learning works.

You are encouraged to start with the default scenario (difficulty 0) and progressively test harder difficulty levels as you improve your agent.

### Environment Overview

The `SpaceRaceEnv` class (found in `SpaceRace/space_race_env.py`) provides a Gymnasium-compatible interface. The main methods are:

- `reset()`
    - Start a new game and return the initial observation
- `step(action)`
    - Execute an action and return `(observation, reward, terminated, truncated, info)`

### Action Space

The action parameter in `step(action)` should be one of:

- `0` — Move the ship **up** (toward the goal)
- `1` — Move the ship **down** (away from the goal)

### Observation Space

There are two Observation Spaces. The Easy and Normal. Easy will be explored during the tutorial class dedicated to RL, while you will use Normal for the Assignment.

### Easy observation space (semantic)
### Easy observation space (semantic)

The observation is a `numpy.ndarray` of shape `(18, 13, 3)` with `dtype=float32`:

- **Channel 0**: Ship position (1.0 at ship cell, 0.0 elsewhere)
- **Channel 1**: Debris positions (1.0 where debris occupies, 0.0 elsewhere)
- **Channel 2**: Normalized time remaining (uniform value in [0, 1])

```python
from SpaceRace.space_race_env import SpaceRaceEnv

env = SpaceRaceEnv(obs_mode="semantic")
obs, info = env.reset()

print(obs.shape)   # (18, 13, 3)
print(obs.dtype)   # float32

# Find ship position
ship_pos = np.where(obs[:, :, 0] == 1.0)
print(f"Ship at row {ship_pos[0][0]}, col {ship_pos[1][0]}")

# Check time remaining
print(f"Time remaining: {obs[0, 0, 2] * 100:.0f}%")
```

### Normal observation space (rgb)
The observation is a `numpy.ndarray` of shape `(54, 39, 3)` with `dtype=uint8`:

- **Pixel values**: RGB colors in range [0, 255]
- **Background (space)**: Dark blue `(5, 10, 20)`
- **Debris**: Yellow/tan `(230, 180, 70)`
- **Ship**: Cyan `(90, 220, 250)`

The image is upsampled by a factor of 3 from the grid size (18×13 → 54×39).

```python
from SpaceRace.space_race_env import SpaceRaceEnv

env = SpaceRaceEnv(obs_mode="rgb")
obs, info = env.reset()

print(obs.shape)   # (54, 39, 3)
print(obs.dtype)   # uint8

# Normalize for neural network input
obs_normalized = obs.astype(np.float32) / 255.0
```

### Rewards
The default reward structure is:

- **+1.0** for reaching the top row (completing a crossing)
- **−0.25** for colliding with debris
- **+0.02** for moving up
- **−0.01** for moving down

### Episode Termination

- `terminated` is always `False` during normal gameplay
- `truncated` becomes `True` when the 60-second timer expires
- Collisions do **not** end the episode — the ship respawns at the bottom after a brief delay

### Difficulty Levels

You can create different scenarios using the `difficulty` argument (0 to 3):

| Difficulty | Description                                   |
| ---------- | --------------------------------------------- |
| 0          | Baseline — deterministic debris, normal speed |
| 1          | Faster debris                                 |
| 2          | Random debris initialization                  |
| 3          | Higher debris density + random initialization |

Example usage:
```python
from SpaceRace.space_race_env import SpaceRaceEnv

env = SpaceRaceEnv(difficulty=0, round_time_seconds=60, ticks_per_second=10)
obs, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Final score: {info['score']}")
env.close()
```

### Visualizing the Game

To see how the game works visually:

```bash
# Random agent visual demo (requires pygame)
python SpaceRace/display_demo.py

# Manual keyboard play
python SpaceRace/manual_play.py
```

## Using Better Examples with a Heuristic Policy

Playing the game randomly may not yield many good examples for the agent to learn from. To improve this, you can use experiences taken from games played by a heuristic policy.

### Accessing Semantic Information
While your agent must learn from **RGB observations only**, the environment provides semantic information in the `info` dictionary for building heuristics:

```python
from SpaceRace.space_race_env import SpaceRaceEnv

env = SpaceRaceEnv(obs_mode="rgb")  # Agent sees RGB
obs, info = env.reset()

# RGB observation for the neural network
print(obs.shape)  # (54, 39, 3) - this is what your agent trains on

# Semantic observation for heuristic development (in info dict)
semantic_obs = info["semantic_obs"]
print(semantic_obs.shape)  # (18, 13, 3) - use this for heuristics only!
```

The semantic observation has three channels:

- **Channel 0**: Ship position (1.0 at ship cell, 0.0 elsewhere)
- **Channel 1**: Debris positions (1.0 where debris occupies, 0.0 elsewhere)
- **Channel 2**: Normalized time remaining (uniform value in [0, 1])

### Building a Heuristic Policy

You can write a heuristic that analyzes the semantic observation to:

1. Locate the ship's row
2. Check if debris blocks the cell directly above
3. Decide whether to move up (if clear) or wait/move down (if blocked)

This is an example heuristic and you can develop it to obtain better training data.

Example heuristic extraction:
```python
def extract_info_from_obs(semantic_obs):
    """Extract ship and debris info from semantic observation."""
    ship_channel = semantic_obs[:, :, 0]
    debris_channel = semantic_obs[:, :, 1]

    # Find ship position
    ship_pos = np.where(ship_channel == 1.0)
    if len(ship_pos[0]) > 0:
        ship_row, ship_col = ship_pos[0][0], ship_pos[1][0]
    else:
        ship_row, ship_col = None, None

    # Check debris in the row above the ship
    if ship_row is not None and ship_row > 0:
        debris_above = debris_channel[ship_row - 1, ship_col] > 0
    else:
        debris_above = False

    return ship_row, ship_col, debris_above

def heuristic_policy(info):
    """Simple heuristic: move up if clear, otherwise move down."""
    semantic_obs = info["semantic_obs"]
    ship_row, ship_col, debris_above = extract_info_from_obs(semantic_obs)

    if debris_above:
        return 1  # Move down (wait for debris to pass)
    return 0  # Move up
```

### Using the Heuristic to Generate Training Data

Use the heuristic policy to collect good experiences for warm-starting your replay buffer:
```python
env = SpaceRaceEnv(obs_mode="rgb")
obs, info = env.reset()

experiences = []
done = False
while not done:
    action = heuristic_policy(info)  # Uses semantic info
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    # Store transition with RGB observations (what the NN will learn from)
    experiences.append((obs, action, reward, next_obs, terminated or truncated))

    obs, info = next_obs, next_info
    done = terminated or truncated

print(f"Heuristic score: {info['score']}")
```

> **⚠️ Evaluation Warning**: During Codabench evaluation, the environment is created with `include_semantic_info=False`. This means `info["semantic_obs"]` will **not be available** — your agent's `select_action(obs)` method must work with only the RGB observation. The semantic info is provided only for training-time heuristics.
> 

To test your agent under evaluation conditions locally:

```python
# This mirrors the Codabench evaluation setup
env = SpaceRaceEnv(obs_mode="rgb", include_semantic_info=False)
```

## Deep Q-Network
Deep Q-Network (DQN) receives a state s as input and outputs the Q-value Q(s, a) for every action a. This allows the network to act as a regular table, where you can look up the Q-value using the state as the input key, and observe the produced values as outputs for every action. Typically, the DQN is trained using experience replay and a target network.

## Exploration strategies

The tradeoff between exploration and exploitation has been extensively studied in the literature.  Many strategies exist. We have discussed **$\epsilon$-Greedy** where the chosen action $a$ is the maximizer of que Q-value function with probability $1-\epsilon$ or a random action taken uniformly from the set $\mathcal{A}$ of possible actions:
a(s) = \begin{cases} \argmax_{a \in \mathcal{A}} Q_\pi(a,s) & \text{ with probability } 1-\epsilon \\ \text{randomUniform}(\mathcal{A}) & \text{ with probability } \epsilon  \end{cases}


The probability of exploration $\epsilon$ is, in general, decreased according to a decay pattern we saw in class. But there are other possibilities. For example, you could investigate other relevant distributions, namely depending on the current state $s$, or the states already seen so far. Or you could try a constant $\epsilon$ throughout training.  You should implement, test and analyze two exploration strategies.

**Boltzmann Exploration** assigns a probability to each possible action based on its estimated value, promoting a stochastic selection mechanism that favors higher-value actions while still allowing for exploration of lesser-known options. The probability $P(a_i)$ of selecting action $a_i$ is calculated using the softmax function

P(a_i)=\frac{\exp(Q(a_i)/T)}{\sum_{j}\exp(Q(a_j)/T)},



where $Q(a_i)$ is the estimated value (expected return) of action $a_i$, and $T$ is a temperature parameter that controls the level of exploration.  The temperature parameter T plays an important role in determining the exploration-exploitation balance. If T is high, a more uniform probability distribution across actions, encouraging exploration. All actions have nearly equal chances of being selected, regardless of their Q-values.

## Experience Replay

Maintaining a replay buffer allows us to reuse collected data multiple times. Additionally, sampling batches randomly from the buffer breaks the correlation between consecutive data, which can make training more stable. Implementing the replay buffer enables experience replaying.

In RL, the distinction between on-policy (like SARSA) and off-policy (like Q-learning) methods affects replay buffer design. On-policy methods learn from experiences generated by the current policy, so a smaller buffer (1,000-3,000) with recent experiences works best. Off-policy methods like the DQN can learn from any past experiences regardless of which policy generated them, allowing for larger buffers (5,000-10,000) that provide more diverse training data and improve stability. Nevertheless, if the experience was generated by a very different policy, it might not provide useful learning signals or could even hinder performance. Maintaining a balance between buffer size and recency of experiences is important for effective learning.




## Target network

During the update process, we attempt to push the predicted Q values towards the target Q values, which are the immediate reward plus the bootstrapped Q values. Since a single update affects the entire network, it causes the non-stationarity of the target Q values. To make the target Q values stationary, we can use another network to provide the target Q values. We can periodically update this network using the weights of the original network. This is known as the target network.

## Tasks

### Task 1. Basic DQN Implementation without Experience Replay and Target network

Implement a basic Deep Q-Network (DQN) without using Experience Replay or Target Network. This implementation should focus on the core DQN architecture and Q-learning loss function.

### 1.1 Set up the neural network architecture

Design and implement a neural network architecture suitable for processing the game's state images and outputting Q-values for each possible action. Consider using convolutional layers to process the image input effectively. Note that a smaller network will be more stable to train.

### 1.2 Define and implement the Q-learning loss function

This is a critical component of the DQN implementation. Create a loss function that computes the difference between the predicted Q-values and the target Q-values calculated using the Bellman equation:
Loss = (r + γ * max(Q(s', a')) - Q(s, a))²
where r is the immediate reward, γ is the discount factor, Q(s', a') represents the Q-values for the next state, and Q(s, a) represents the current predicted Q-value. This loss function drives the learning process by pushing the predicted Q-values toward their target values.

### 1.3 Implement the training loop and evaluation

Create a training loop that allows the agent to interact with the environment, collect experiences, and update the network using the defined loss function. Test the implementation by running experiments to establish a Q-Learning baseline performance, documenting the agent's learning progress and any challenges encountered.

### 1.4 Implement an heuristic policy
Implement a heuristic policy that helps the agent learn more effectively by providing better examples than random play. The heuristic should make reasonable decisions about snake movement based on food location, avoiding collisions with walls and itself. You can implement this by using the game's state information (through the get_state() method) to create a policy that moves the snake toward food while avoiding obstacles, as explained above, in ‣. Document the heuristic baseline performance without RL.

- **Resource Constraint:** Training must not exceed 4 hours.
- **Evaluation:** Performance (score and Q-value) and efficiency (score / training time, Q-value / training time) baseline measurements.

*Deliverables:*

- Code implementation.
- A brief presentation detailing:
    - The neural network architecture and its effectiveness for the task.
    - Implementation details of the Q-learning loss function.
    - Results of training experiments and baseline performance metrics.
    - Heuristic policy algorithm and its performance compared to random play.

### Task 2. Enhanced DQN with Experience Replay and Target Network

- Improve the DQN of Task 1 with:
    - Experience Replay Buffer (size ≤ 10,000).
    - Target Network (updated around every 100 steps).

### 2.1 Experience Replay Buffer Implementation
The experience replay buffer stores transitions (state, action, reward, next_state, done) to break the correlation between consecutive samples and improve training stability. Here are key considerations for implementing an effective replay buffer:

- **Buffer Structure:** Use a data structure that allows efficient storage and sampling of experiences. A circular buffer or deque with a maximum size works well.
- **Transition Storage:** Store each transition as a tuple of (state, action, reward, next_state, done) where:
    - state: The current state (image of the board)
    - action: The action taken
    - reward: The reward received
    - next_state: The resulting state after taking the action
    - done: Boolean indicating if the episode terminated
- **Memory Management:** When the buffer reaches capacity, replace old experiences with new ones (FIFO approach).
- **On-Policy and Off-Policy training and the size of the Replay Buffer:** test different replay buffer capacities, for example, {1, 100, 10000}.
- **Batch Sampling:** Implement random sampling to pull a batch of transitions for training. This randomization helps break temporal correlations.
- **Prioritized Experience Replay (Optional Enhancement):** Consider implementing prioritized sampling where transitions with higher TD errors are sampled more frequently.
    
    Temporal Difference (TD) errors represent the difference between the predicted value of a state-action pair and the actual observed value during reinforcement learning. Mathematically, the TD error is defined as:
    δ = r + γ \max_{a'} Q(s', a') - Q(s, a)


    - r is the immediate reward received
- γ is the discount factor
- Q(s', a') is the estimated value of the next state-action pair
- Q(s, a) is the estimated value of the current state-action pair

In Prioritized Experience Replay, transitions with larger TD errors are considered more "surprising" or informative, as they represent experiences where the agent's predictions were furthest from reality. By sampling these high-error transitions more frequently during training, the agent can focus learning on the most informative experiences, potentially accelerating the learning process and improving overall performance.

The sampling probability for each transition can be determined using the formula:
P(i) = \frac{|δ_i|^α}{\sum_j |δ_j|^α}

Where α is a hyperparameter that determines how much prioritization is used (α = 0 corresponds to uniform sampling).

Here is a base implementation of the Replay Buffer:


```python
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        # Add experience to buffer
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        # Random sampling of experiences
        experiences = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # Separate the tuple into batches
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Convert to appropriate tensor format
        return (np.array(states), np.array(actions), 
                np.array(rewards), np.array(next_states), 
                np.array(dones))
                
    def __len__(self):
        return len(self.buffer)

```

### 2.2 Integration with DQN Training

After implementing the replay buffer, integrate it into your training loop with these steps:

- **Initial Population:** Collect initial experiences (using your epsilon-greedy exploration) to populate the buffer before starting training. Also try the heuristic policy to warm-start the Replay Buffer.
- **Experience Collection:** For each step in the environment:
    - Select an action using your exploration strategy
    - Execute the action and observe the reward, next state, and done flag
    - Store this transition in the replay buffer
- **Training:** After each step (or periodically):
    - Sample a random batch from the replay buffer
    - Compute target Q-values using the target network
    - Update the primary network using the calculated loss
- **Efficiency Tips:**
    - Use batching for both forward and backward passes.
    - Normalize state inputs to stabilize training
    - Warm start using the heuristic policy.

### 2.3 Target Network Implementation

The target network is a copy of the main Q-network that provides stable Q-value targets during training:

- **Initialization:** Create an identical copy of your main DQN network.
- **Update Strategy:** Update the target network every 100 steps as specified, using either:
    - Hard update: Copy all weights directly
    - Soft update: Gradually update using a small τ value (e.g., τ = 0.01), where target_weights = τ*online_weights + (1-τ)*target_weights
- **Target Q-Value Calculation:** Use the target network to compute the max Q-value for the next state when calculating the TD target.
- **Resource Constraint:** Training must not exceed 4 hours.
- **Evaluation:** Performance (score and Q-value) and efficiency (score / training time, Q-value / training time) improvements.

*Deliverables:*

- Code implementation.
- A comparative analysis presentation, improving the previous presentation, plus highlighting:
    - Implementation details of Experience Replay, including buffer size experiments and their impact on performance.
    - Target Network update strategies tested and their effects on training stability.
    - Comparative analysis showing improvements in both performance metrics and training efficiency over the basic DQN.
    - Visualizations of learning curves demonstrating the stabilizing effects of these enhancements.



### Task 3. Exploration Strategies

Implement and compare two different exploration strategies for your DQN agent:

- **Epsilon-Greedy:** Implement the standard ε-greedy strategy with a decaying epsilon parameter. Experiment with different initial epsilon values and decay rates to find an optimal balance between exploration and exploitation. Document how different decay schedules affect learning performance.
- **Boltzmann Exploration:** Implement temperature-based Boltzmann exploration using the softmax function as described earlier in the assignment. Test different temperature schedules and analyze how they affect exploration patterns compared to ε-greedy.

### 3.1 Implementing Epsilon-Greedy

For the ε-greedy strategy, implement the following:

- **Initialization:** Set initial epsilon value (e.g., 1.0 for full exploration at the beginning)
- **Decay Schedule:** Implement a decay function that reduces epsilon over time. Consider linear, exponential, or step-based decay schedules.
- **Action Selection:** With probability epsilon, select a random action; otherwise, select the action with the highest Q-value.
- **Analysis:** Track and visualize how epsilon changes throughout training and how it affects the agent's performance.

### 3.2 Implementing Boltzmann Exploration

For Boltzmann exploration, implement:

- **Temperature Parameter:** Set an initial temperature value and implement a schedule for decreasing it over time.
- **Softmax Calculation:** Compute action probabilities using the softmax function based on Q-values and temperature.
- **Action Sampling:** Sample actions according to the calculated probability distribution.
- **Analysis:** Track how temperature affects the action distribution and compare the exploration patterns with ε-greedy.

### 3.3 Comparative Analysis

Perform a thorough comparison of both exploration strategies:

- **Learning Efficiency:** Compare how quickly each strategy learns effective policies.
- **Final Performance:** Evaluate which strategy achieves higher final scores and Q-values after equivalent training time.
- **Exploration Behavior:** Analyze how each strategy explores the state space differently.
- **Stability:** Assess which strategy produces more stable learning curves.
- **Hyperparameter Sensitivity:** Determine how sensitive each strategy is to its hyperparameters.

*Deliverables:*

- Code implementation.
- A comprehensive final presentation, improving the previous presentation, plus discussing:
    - Implementation details for both exploration strategies, including parameter settings and decay/temperature schedules.
    - Comparative analysis of Epsilon-Greedy vs. Boltzmann exploration with respect to learning efficiency, exploration behavior, and final performance.
    - Visualizations showing how different exploration strategies affect learning curves and state space coverage.
    - Recommendations for which exploration strategy works best for the Snake game environment and why.








# Remember

1. **Start simple**: Even a hand-coded heuristic beats random play
2. **Use frame stacking**: Consider stacking 2 to 4 frames to capture motion
3. **Monitor metrics**: Track score, loss, and ε/temperature during training
4. **Experiment with difficulty**: Start with difficulty 0, then increase
5. **Visualize learning**: Plot learning curves to diagnose issues
6. **Test locally**: Always run `local_eval.py` before submitting to competition
7. **Submit early**: Each phase has a deadline -- don't wait until the last minute
8. **Iterate**: Use competition feedback to improve between submissions




# Grading Rubric
Task	Criterion	Weight (%)

Task 1: Basic DQN	
Neural Network Architecture	5%
Innovative and well-justified architecture tailored to the task; clear rationale for design choices.

	Q-Learning Loss Function	10%
    Accurate and well-explained implementation demonstrating strong understanding of Q-Learning principles.


	Training Loop and Evaluation	5%
    Efficient and well-structured training loop; comprehensive evaluation with insightful analysis of results.


    	Heuristic Policy	10%
Well-designed heuristic significantly improving performance; clear and thorough justification.


	Presentation Quality	3%
    Clear, concise, and well-structured presentation; effectively communicates insights, methodologies, and findings; utilizes visual aids proficiently.
    » so for this i want to have good plots, nice tables, etc... 



	Peer Review	2%
    Actively participates in peer review; delivers detailed, constructive feedback with actionable suggestions; reflects on peer feedback to improve own work.


    
Task 2: Enhanced DQN with Experience Replay and Target Network	Experience Replay	10%
Efficient and well-justified implementation demonstrating strong understanding of experience replay concepts.



	Target Network 	10%
    Efficient and well-justified implementation demonstrating strong understanding of target network concepts.


	Integration and Performance Evaluation	5%
    Seamless integration leading to significant performance improvement; comprehensive evaluation with insightful analysis of results.



	Presentation Quality	3%
    Clear, concise, and well-structured presentation; effectively communicates insights, methodologies, and findings; utilizes visual aids proficiently.


	Peer Review	2%
    Actively participates in peer review; delivers detailed, constructive feedback with actionable suggestions; reflects on peer feedback to improve own work.



Task 3: Exploration Strategies	Epsilon-Greedy 	10%
Efficient and well-justified implementation demonstrating strong understanding of epsilon-greedy strategy and parameter tuning.


	Boltzmann Exploration	10%
    Efficient and well-justified implementation demonstrating strong understanding of Boltzmann exploration strategy and parameter tuning.



	Comparative Analysis of Exploration Strategies	10%
    Comprehensive and insightful comparison using multiple metrics; provides in-depth interpretation of results and implications for exploration strategies.



	Presentation Quality	3%
    Clear, concise, and well-structured presentation; effectively communicates insights, methodologies, and findings; utilizes visual aids proficiently.



	Peer Review	2%
    Actively participates in peer review; delivers detailed, constructive feedback with actionable suggestions; reflects on peer feedback to improve own work.