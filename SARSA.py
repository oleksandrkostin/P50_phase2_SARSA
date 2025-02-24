import numpy as np
import random

# Define spatial grid (5x5)
SPATIAL_RELATIONS = ["behind", "on", "under", "infront", "beside"]
OBJECTS = ["table", "chair", "shelf", "box", "flower"]
ACTIONS = ['up', 'down', 'left', 'right']  # Possible movements

# Hyperparameters
EPSILON = 0.1  # Exploration rate
ALPHA = 0.1    # Learning rate
GAMMA = 0.9    # Discount factor
EPISODES = 500  # Training episodes

# Initialize Q-table (5x5 grid x 4 actions)
Q_table = np.zeros((5, 5, len(ACTIONS)))

# User-defined goal
goal_position = None

# Function to manually set goal
def set_goal():
    global goal_position
    print("\nDefine the goal position (e.g., 'behind table' -> (0,0))")
    
    while True:
        relation = input(f"Choose spatial relation {SPATIAL_RELATIONS}: ").strip().lower()
        obj = input(f"Choose object {OBJECTS}: ").strip().lower()

        if relation in SPATIAL_RELATIONS and obj in OBJECTS:
            goal_position = (SPATIAL_RELATIONS.index(relation), OBJECTS.index(obj))
            print(f"Goal set: {relation} {obj} -> {goal_position}")
            break
        else:
            print("Invalid input. Try again.")

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    x, y = state
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)  # Explore
    else:
        return ACTIONS[np.argmax(Q_table[x, y])]  # Exploit

# Function to take a step in the environment
def step(state, action):
    x, y = state
    if action == 'up':
        x = max(x - 1, 0)
    elif action == 'down':
        x = min(x + 1, 4)
    elif action == 'left':
        y = max(y - 1, 0)
    elif action == 'right':
        y = min(y + 1, 4)

    if (x, y) == goal_position:
        feedback = "y"
    else:
        feedback = "n"

    if feedback == "y":
        reward = 10  # Positive reward for correct placement
    else:
        reward = -1  # Negative reward for incorrect placement

    return (x, y), reward

# Training loop for SARSA with human feedback
def train(visualize_callback=None, new_goal_position=None):
    global goal_position  # Declare goal_position as global
    if new_goal_position is not None:
        goal_position = new_goal_position  # Assign the provided goal_position to the global variable
    reward_history = []

    for episode in range(EPISODES):
        state = (random.randint(0, 4), random.randint(0, 4))  # Start at random position
        action = choose_action(state)
        total_reward = 0

        while state != goal_position:  # Until goal is reached
            next_state, reward = step(state, action)
            next_action = choose_action(next_state)

            # SARSA update rule
            x, y = state
            nx, ny = next_state
            action_idx = ACTIONS.index(action)
            next_action_idx = ACTIONS.index(next_action)

            Q_table[x, y, action_idx] += ALPHA * (reward + GAMMA * Q_table[nx, ny, next_action_idx] - Q_table[x, y, action_idx])

            state = next_state
            action = next_action
            total_reward += reward

            # Call visualization callback if provided
            if visualize_callback:
                visualize_callback(Q_table, episode, state, total_reward, goal_reached=False)

        # Episode ends when goal is reached
        if visualize_callback:
            visualize_callback(Q_table, episode, state, total_reward, goal_reached=True)

        reward_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")

    return reward_history