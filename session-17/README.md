# Session 17 Assignment

## Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

TD3 agent that learns to navigate a city map and reach sequential waypoints using continuous control and reinforcement learning.

### Features

- Continuous action space (speed and steering control)
- Twin critic networks with delayed policy updates
- Target policy smoothing for stability
- Success-weighted experience replay
- Real-time visualization with sensor feedback
- Sequential waypoint navigation
- Configurable hyperparameters via YAML

### Quick Start

1. Run the application:
```bash
uv run ./src/session_17
```

2. Load a config (File dialog will open). Example configs are present in the `assets/` folder.

3. Right-click on map to:
   - Set start point
   - Add waypoints

4. Click "Start" to begin training

5. Use "Pause" and "Reset" as needed

### Controls

- **Right-click map**: Add start point or waypoint
- **Start**: Begin RL training loop
- **Pause/Resume**: Pause or continue training
- **Reset**: Reset simulation state

### Configuration

Training configurations are loaded from YAML files in the `assets/` folder.

### Architecture

**Actor Network:**
- Input: 9 features (7 sensors + angle to target + distance to target)
- Layers: 9 → 400 → 300 → 2 (continuous actions)
- Outputs: Speed [-1, 1], Turn angle [-1, 1]
- Activation: ReLU hidden layers, Tanh output

**Critic Networks (Twin Q-networks):**
- Input: State (9) + Action (2) = 11 features
- Layers: 11 → 400 → 300 → 1 (Q-value)
- Two independent critics to reduce overestimation bias

**Actions:**
- Speed: Continuous [-1, 1] × max_speed
- Turn angle: Continuous [-1, 1] × max_turn_angle

**Sensors:**
- 7 brightness sensors at [-45°, -30°, -15°, 0°, 15°, 30°, 45°]
- Detect road (bright) vs obstacles (dark)

### Reward Structure

The reward function balances multiple objectives:

1. **Centered Sensor Reward** (scale: 5-8)
   - Rewards staying centered on the road
   - Based on center sensor brightness (0.0 to 1.0)
   - Prevents wall-hugging behavior

2. **Distance Reward** (scale: 10-15)
   - Formula: `((prev_dist - curr_dist) / car_max_speed) × scale`
   - Rewards progress toward current waypoint
   - Normalized by car's movement capability (map-size independent)
   - Penalizes moving away from target

3. **Speed Reward** (scale: 2.5)
   - Formula: `speed × scale`
   - Encourages forward movement
   - Prevents hesitation and stalling

4. **Terminal Rewards:**
   - Collision: -100 (episode ends)
   - Reach waypoint: +100 (continues to next waypoint)
   - Base step penalty: -0.1

**Key Design Principles:**
- Distance reward dominates to ensure goal-directed behavior
- Speed reward breaks ties when distance is stable
- Sensor reward provides gentle lane-keeping guidance
- Rewards are normalized to be map-size independent

### TD3 Algorithm Details

**Key Components:**

1. **Twin Critics**: Two Q-networks prevent overestimation
   - Target Q-value uses minimum: `min(Q1, Q2)`
   - Reduces positive bias in value estimates

2. **Delayed Policy Updates**: Actor updates less frequently than critics
   - Critic update every step
   - Actor update every 2 steps (configurable)
   - Stabilizes training

3. **Target Policy Smoothing**: Adds noise to target actions
   - Noise: `N(0, 0.2)` clipped to [-0.5, 0.5]
   - Smooths Q-value estimates
   - Prevents exploitation of critic errors

4. **Polyak Averaging**: Slow target network updates
   - Formula: `target = τ × current + (1-τ) × target`
   - τ = 0.005 (very slow updates)
   - Provides stable learning targets

### Experience Replay

**Success-Weighted Replay Buffer:**
- Total capacity: 1,000,000 experiences
- Split into two buffers:
  - Priority buffer (300,000): Episodes that reached ≥1 waypoint
  - Regular buffer (700,000): Failed episodes

**Dynamic Sampling:**
```python
success_rate = len(priority) / len(total)
priority_ratio = 0.3 + (success_rate × 0.4)  # Range: 30% to 70%
```

- Early training: 30% priority samples (even with few successes)
- Late training: Up to 70% priority samples (when many successes available)

This prevents catastrophic forgetting and reinforces successful behaviors.

### Hyperparameters

### Hyperparameters

**Environment:**
- `car_max_speed`: Maximum speed in pixels per step. Higher values allow faster navigation but reduce control precision.
- `car_max_turn_angle`: Maximum turn angle in degrees per step. Higher values enable sharper turns but can cause instability.
- `sensor_dist`: Distance of sensors from car center in pixels. Affects early obstacle detection range.
- `collision_threshold`: Brightness threshold (0-1) for obstacle detection. Higher values are more sensitive to obstacles (stricter road detection).

**Reward Scales:**
- `centered_sensor_reward_scale`: Multiplier for staying centered on road. Higher values emphasize lane keeping.
- `distance_reward_scale`: Multiplier for progress toward waypoint. Higher values emphasize goal-directed behavior.
- `speed_reward_scale`: Multiplier for forward movement reward. Higher values encourage the agent to move forward rather than staying still, but don't directly force higher speeds.

**Agent:**
- `batch_size`: Number of experiences sampled per training step. Larger batches provide more stable gradients.
- `discount (γ)`: Discount factor for future rewards (0-1). Higher values make agent more forward-looking.
- `tau (τ)`: Target network update rate (0-1). Lower values make target networks update more slowly.
- `policy_noise`: Standard deviation of noise added to target actions. Smooths value estimates.
- `noise_clip`: Maximum absolute value of noise added to target actions. Prevents extreme perturbations.
- `policy_update_freq`: Number of critic updates per actor update. Higher values give critics more time to converge.

**Training:**
- `random_action_steps`: Number of initial steps with random actions. Fills replay buffer with diverse experiences.
- `exploration_noise`: Standard deviation of noise added to actions during training. Higher values increase exploration.

### How It Works

1. **Exploration Phase**: Random actions for first 10,000 steps
   - Fills replay buffer with diverse experiences
   - Agent hasn't learned anything yet

2. **Training Phase**: TD3 with exploration noise
   - Select action from actor network
   - Add Gaussian noise: `action + N(0, 0.01)`
   - Execute action, observe reward and next state
   - Store experience in appropriate buffer (priority or regular)
   - Sample batch and update networks

3. **Inference Phase**: Deterministic policy (no noise)
   - Use actor network directly
   - Smooth, consistent behavior
   - Used for evaluation and submission videos

4. **Sequential Navigation**: 
   - Agent must reach waypoints in order
   - Switches to next waypoint when within 20 pixels
   - Episode ends when all waypoints reached or collision occurs

### Results

Training demonstration and final performance: [YouTube Link](https://youtu.be/09adYiMIovc)

### Known Issues / Notes

- Sensors detect brightness (white = road, dark = obstacle)
- Collision detection uses brightness threshold (0.8 with orange walls)
- Distance normalization by `car_max_speed` makes rewards map-size independent
- TD3 is sensitive to reward scale - distance, speed, and sensor rewards must be balanced
- Training uses exploration noise, inference is deterministic (no noise)
- Success-weighted replay is critical - uniform sampling leads to catastrophic forgetting
- Map should have bright roads and dark obstacles for proper sensor detection
