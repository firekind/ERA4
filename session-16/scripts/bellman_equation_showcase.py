import numpy as np

# Grid parameters
N = 4  # 4x4 grid
gamma = 1.0  # No discounting
theta = 1e-4  # Convergence threshold

# Initialize value function
V = np.zeros((N, N))

# Terminal state (bottom-right)
terminal_state = (N - 1, N - 1)

# Actions: up, down, left, right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# Reward function
def get_reward(state):
    if state == terminal_state:
        return 0
    return -1


# Check if state is valid
def is_valid(state):
    row, col = state
    return 0 <= row < N and 0 <= col < N


# Get next state given current state and action
def get_next_state(state, action):
    next_row = state[0] + action[0]
    next_col = state[1] + action[1]
    next_state = (next_row, next_col)

    # If next state is out of bounds, stay in current state
    if not is_valid(next_state):
        return state
    return next_state


# Value iteration
iteration = 0
while True:
    delta = 0
    V_new = V.copy()

    for row in range(N):
        for col in range(N):
            state = (row, col)

            # Skip terminal state
            if state == terminal_state:
                continue

            # Calculate value for current state
            value = 0
            for action in actions:
                next_state = get_next_state(state, action)
                reward = get_reward(state)

                # Bellman equation
                # P(s'|s,a) = 1.0 for deterministic transitions
                # But agent moves with equal probability, so P(a) = 0.25
                value += 0.25 * (reward + gamma * V[next_state])

            V_new[state] = value
            delta = max(delta, abs(V_new[state] - V[state]))

    V = V_new
    iteration += 1

    if delta < theta:
        break

print(f"Converged after {iteration} iterations")
print("\nFinal Value Function:")
print(V)
