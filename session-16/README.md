# City Navigation RL Agent

Deep Q-Network (DQN) agent that learns to navigate a city map and reach sequential waypoints using reinforcement learning.

## Features

- Temperature-based exploration with Softmax action selection
- Prioritized experience replay (successful episodes prioritized)
- Sequential waypoint navigation (A1 → A2 → A3)
- Real-time visualization with sensor feedback
- Configurable hyperparameters via YAML

## Quick Start

1. Run the application:
```bash
uv run scripts/cityrl.py
```

2. Load a config (File dialog will open). Examples configs are present in the `assets/` folder.

3. Right-click on map to:
   - Set start point
   - Add waypoints

4. Click "Start" to begin training

5. Use "Pause" and "Reset" as needed

## Controls

- **Right-click map**: Add start point or waypoint
- **Start**: Begin RL training loop
- **Pause/Resume**: Pause or continue training
- **Reset**: Reset simulation state

## Configuration

Training configurations are loaded from YAML files in the `assets/` folder.

## Architecture

- **Input**: 9 features (7 sensors + angle to target + distance to target)
- **Network**: 128 → 256 → 256 → 128 → 5 (actions)
- **Actions**: Left, Straight, Right, Sharp Left, Sharp Right
- **Sensors**: 7 distance sensors at [-45°, -30°, -15°, 0°, 15°, 30°, 45°]

## Reward Structure

- Step penalty: -0.1
- Clear path ahead: +20 (based on center sensor)
- Moving away from target: -10
- Collision: -100 (episode ends)
- Reach waypoint: +100 (continues to next waypoint)

## How It Works

1. **Exploration**: Uses temperature-based Softmax action selection
   - High temperature (50.0) → More exploration
   - Decays to low temperature (0.1) → More exploitation

2. **Training**: DQN with target network and prioritized replay
   - Successful episodes (positive reward) → priority memory
   - Failed episodes → regular memory
   - Dynamic sampling ratio based on success rate

3. **Sequential Navigation**: Agent must reach waypoints in order
   - Switches to next waypoint when within 20 pixels
   - Episode ends when all waypoints reached or collision occurs

## Results

Training demonstration and final performance: [YouTube Link](https://youtu.be/LySRaQBh7Ko)

## Assignment Questions

### What happens when boundary-signal is weak compared to the last reward?

The boundary signal (sensor reward weight, currently 20) guides obstacle avoidance. 
If weak relative to target reward (100):
- Agent prioritizes reaching targets over safety
- More crashes, especially in tight spaces
- Faster but riskier navigation

### What happens when Temperature is reduced?

Lower temperature → more greedy/exploitative behavior:
- Temperature = 50: Explores broadly, tries suboptimal actions
- Temperature = 5: Mostly picks best action, occasional exploration
- Temperature = 0.1: Nearly pure greedy, always best action

Reducing temperature is necessary for convergence - constant high temperature 
prevents the agent from fully exploiting its learned policy.

### What is the effect of reducing gamma?

Gamma (discount factor, currently 0.95) controls future reward valuation:
- High gamma (0.95-0.99): Values long-term rewards, plans ahead
- Low gamma (0.1-0.5): Only cares about immediate rewards, short-sighted
- Gamma → 0: Purely myopic, ignores future consequences

Low gamma would cause the agent to:
- Ignore distant waypoints
- Not plan paths through narrow passages
- Get stuck in local optima (e.g., getting closer but hitting walls)

## Known Issues / Notes

- Sensors detect brightness (white = road, dark = obstacle)
- Car uses center sensor (index 3) for path-ahead reward
- Collision detection checks car center point only
- Map should have white roads and dark obstacles
