# Reinforcement Learning: Q-Learning

In this class, we moved beyond passive value estimation and introduced **Q-learning**, one of the core building blocks of modern RL.

## From Value Functions to Action-Value Functions

Recall that a **value function** estimates how good it is to be in a state under a given policy:

$V^{\pi}(s) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s\right]$

The **action-value function**, or Q-function, refines this by also conditioning on the action:

$Q^{\pi}(s,a) = \mathbb{E}_{\pi}\left[G_t \mid S_t = s, A_t = a\right]$

## Thought Experiment: Policy Improvement via Q-Functions

Let’s suppose we already have a reasonably good policy $\pi$ and an accurate estimate of its associated **action-value function** $Q^{\pi}$.

Now, we ask: **can we do better just by looking at** $Q^{\pi}$**?**

We define a new policy $$\pi_{\text{new}}$$ that **greedily selects actions** which maximize the estimated return:

$\pi_{\text{new}}(a_t,s_t)=
\begin{cases}
1 & \text{if } a_t=\arg\max_a Q^{\pi}(s_t,a) \\
0 & \text{otherwise}
\end{cases}$

This is a **deterministic policy** that always chooses the best action according to the current Q-function.

Key Point:
This is a **policy improvement step**:

> Given a policy $\pi$ and its Q-function $Q^{\pi}$, the greedy policy with respect to $Q^{\pi}$ is at least as good as $\pi$, and often better.
> 

This is guaranteed by the **Policy Improvement Theorem** in dynamic programming.

## The Policy Iteration Loop

1. **Run policy to collect batch of data**
    
    Execute the current policy $\pi$ in the environment to collect transitions of the form:
    
    $(s_t,a_t,r_{t+1},s_{t+1})$
    
2. **Fit model to estimate expected return**
    
    Use the collected data to update the Q-function estimate:
    
    $Q^{\pi}(s_t,a_t) \approx r_{t+1} + \gamma\max_a Q^{\pi}(s_{t+1},a)$
    
3. **Improve policy**
    
    Define a new policy $\pi'$ that is greedy with respect to the estimated $Q^{\pi}$.
    
4. **Repeat**
    
    Set $\pi \leftarrow \pi'$ and go back to step 1.


    ## Why This Matters

This framework underpins many reinforcement learning algorithms:

- **Q-learning** performs this implicitly with TD updates and a Q-table.
- **Fitted Q-Iteration** implements this loop using batches of experience.
- **Deep Q-Learning (DQN)** uses a neural network to approximate $Q^{\pi}$.
- **Actor-Critic methods** rely on $Q^{\pi}$ to improve a separate policy (the actor).











# Deep Q-Network (DQN) Overview

DQN is a groundbreaking algorithm that combines Q-learning with deep neural networks to handle complex state spaces in reinforcement learning.

## Key Components

- **Experience Replay:** Stores and randomly samples past experiences to break correlations in sequential data and improve learning stability
- **Target Network:** Separate network used to generate target values, updated less frequently to reduce overestimation bias
- **Deep Neural Network:** Approximates the Q-function, mapping states to action-values

## Other Innovations

DQN introduced several crucial innovations that helped stabilize the training of deep reinforcement learning agents:

- Use of convolutional neural networks to process visual input
- Frame stacking to capture temporal information
- Reward clipping to handle different scales of rewards

## Training Process


## Training Process

1. Collect experiences (state, action, reward, next state) and store in replay buffer
2. Sample random batch of experiences from buffer
3. Compute target Q-values using target network
4. Update main network weights using gradient descent
5. Periodically update target network weights

## Limitations

- Can struggle with continuous action spaces
- Memory requirements for replay buffer can be substantial
- May require significant computational resources for training

## Applications

DQN has been successfully applied to:

- Atari games (achieving human-level performance)
- Robotic control tasks
- Resource management problems
- Game AI development

## The Q-Function
Q(s,a) = r + γ \max_{a'}[Q(s',a')]



Where:

- Q(s,a) is the Q-value for state s and action a
- r is the immediate reward
- γ is the discount factor
- max[Q(s',a')] is the maximum Q-value for the next state

## DQN for the cart pole classical control problem

The cart pole problem is a classic control task where a pole is attached to a cart that moves along a frictionless track. The goal is to keep the pole balanced upright by moving the cart left or right. This creates an excellent testbed for DQN as it involves:

- A continuous state space (cart position, cart velocity, pole angle, pole angular velocity)
- A discrete action space (move left or right)
- Clear success/failure conditions (pole angle within threshold, cart within bounds)

The challenge lies in learning a policy that can maintain balance for as long as possible, making it an ideal problem for demonstrating DQN's capabilities in control tasks.

Here are the steps to work with the DQN cart pole tutorial:

1. Open the provided tutorial link in your browser
2. Click the "Copy to Drive" button at the top of the notebook to save it to your university Google account
3. Make sure you're signed in with your university credentials
4. Once copied, the notebook will open in your own Google Drive environment
5. Connect to a runtime by clicking the "Connect" button in the top right
6. Run each code cell in sequence by clicking the play button or using Shift+Enter

Important notes:

- Read through the explanations before running each cell to understand what the code is doing
- The training process may take some time to complete
- Pay attention to the visualization of the cart pole environment and training progress
- Try experimenting with different hyperparameters after successfully running the base example

After completing the tutorial, you'll have hands-on experience with the DQN for the cart pole problem using PyTorch.

## DQN for the Mountain Car problem

The mountain car problem is another classical reinforcement learning control task where an underpowered car must drive up a steep mountain. The key aspects of this problem are:

- The car doesn't have enough engine power to climb the mountain directly from a standing start
- The agent must learn to build momentum by driving back and forth on the mountain slopes
- Success requires developing a non-obvious strategy of moving away from the goal initially

The state space consists of two continuous variables:

- Position of the car along the track
- - Velocity of the car

The action space is discrete with three possible actions:

- Push left (accelerate backwards)
- No push (coast)
- Push right (accelerate forwards)

This environment is particularly interesting for reinforcement learning because:

- It requires exploration to discover the solution
- It demonstrates the concept of delayed rewards
- The optimal policy is non-obvious to human intuition

The challenge makes it an excellent problem for testing DQN's ability to learn complex strategies through exploration and exploitation.


```python
env = gym.make("MountainCar-v0")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.99
lr = 1e-3
batch_size = 64
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
target_update = 10
buffer_limit = 10000
num_episodes = 500

```

Adapt the DQN from the tutorial to this environment.