# SmolLM2 to DeepSeek Architecture Conversion

## Overview

This project converts the SmolLM2-135M architecture to DeepSeek's architecture, incorporating two key innovations:
1. **Multi-Head Latent Attention (MLHA)** - Replaces Grouped Query Attention (GQA)
2. **Mixture of Experts (MoE)** - Replaces standard MLP layers with shared + routed experts
and training it on a dummy dataset.

## Architecture Changes

### Original SmolLM2 Architecture
```
TransformerBlock:
  ├── RMSNorm
  ├── Grouped Query Attention (GQA)
  │   ├── 9 query heads
  │   └── 3 key-value heads
  ├── RMSNorm
  └── MLP (SwiGLU)
      ├── gate_proj (576 → 1536)
      ├── up_proj (576 → 1536)
      └── down_proj (1536 → 576)
```

### SmolDeepSeek Architecture
```
TransformerBlock:
  ├── RMSNorm
  ├── Multi-Head Latent Attention (MLHA)
  │   ├── Compression: hidden_size → latent_dim (768 → 96)
  │   ├── Decompression: latent_dim → per-head dimensions
  │   ├── RoPE projections for positional encoding
  │   └── 9 attention heads (head_dim = 85)
  ├── RMSNorm
  └── MoE (Mixture of Experts)
      ├── 1 shared expert (always active)
      ├── 7 routed experts (top-2 selection)
      └── Loss-less load balancing via adaptive bias updates
```

## Training Logs

The exact log outputs can be found [here](./assets/training-stdout.txt).

### Loss Progress

| Step  | Loss   | Notes |
|-------|--------|-------|
| 1000  | 4.426  | Initial training, smooth descent |
| 2000  | 3.588  | Continuing to decrease |
| 3000  | 2.409  | Model learning patterns |
| 4000  | 2.127  | Steady progress |
| 5000  | 1.756  | Good convergence |
| 6000  | 1.587  | Near optimal for toy dataset |
| 6365  | -      | ⚠️ **Gradient norm spike: 121.52** |
| 7000  | 0.414  | **Best checkpoint** (before instability) |
| 8000  | 0.559  | Loss starting to oscillate |
| 8435  | NaN    | First NaN loss encountered |
| 9000  | 0.729  | Frequent NaN batches being skipped |
| 10000 | 0.389  | Training completed with many skipped batches |

### Expert Load Balancing

The MoE load balancing worked correctly throughout training:
- **Expected average load**: 2/7 = 0.2857 (28.57% per expert)
- **Observed**: All layers maintained average load of 0.2857
- **Load variation**: Min/max loads stayed close to average, indicating good balancing

## NaN Loss Issue Analysis

### Timeline
1. **Steps 1-6365**: Smooth training, loss decreasing normally
2. **Step 6365**: Large gradient norm detected (121.52)
3. **Steps 6365-8435**: Gradients building up, model becoming unstable
4. **Step 8435+**: NaN losses appearing frequently, many batches skipped

### Root Cause
- **Toy dataset overfitting** → Model memorizing, gradients become erratic
- **Gradient explosion** starting around step 6365 despite gradient clipping
- **bf16 precision instability** - Mixed precision can amplify numerical issues
- Learning rate (1e-4) was already conservative but insufficient for preventing explosion on overfit model

### Why Gradient Clipping Didn't Help
- Gradient clipping (1.0) clips the **norm** of gradients, not individual values
- When model is severely overfitting, many parameters have large gradients simultaneously
- The clipped gradient can still be problematic even if norm ≤ 1.0

### Why Training "Succeeded" Despite NaNs
- `return None` in training_step skipped NaN batches (safety mechanism)
- Training continued with valid batches when they occurred
- Final loss (0.389) achieved, but ~15-20% of batches skipped in final 1600 steps
- Model still learned from valid batches

## How to Avoid NaN Losses (For Future Training)

Since the issue occurred even with conservative learning_rate=1e-4, the root cause is **overfitting-induced instability on a small dataset**. Here are effective solutions:

### Option 1: Much Tighter Gradient Clipping
```yaml
trainer:
  gradient_clip_val: 0.3  # Down from 1.0, more aggressive
```
**Why**: Gradient norm of 121.52 was clipped to 1.0, but still too large. Clip to 0.3-0.5 for toy datasets.

### Option 2: Stop Training Earlier
```yaml
trainer:
  max_steps: 7000  # Instead of 11000
```
**Why**: Loss plateaued around 0.4-0.5 by step 7000. Further training only caused instability.

### Option 3: Learning Rate Warmdown (Decay)
```python
def lr_lambda(current_step):
    warmup_steps = int(0.1 * max_steps)  # 1100 steps
    
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    elif current_step < 6000:  # Before instability
        return 1.0
    else:
        # Aggressive decay after 6000 steps
        decay_steps = max_steps - 6000
        progress = float(current_step - 6000) / float(max(1, decay_steps))
        return max(0.01, 1.0 - 0.99 * progress)  # Decay to 1% of LR
```
**Why**: Reduce learning rate when model starts overfitting (around step 6000).

### Option 4: Use Full Precision (fp32) for Final Steps
```yaml
trainer:
  precision: 32  # Instead of bf16-mixed
```
**Why**: bf16 has limited precision range and can cause numerical instability when gradients are large. fp32 is more stable but slower and uses more memory.

### Option 5: Add Gradient Noise to Stabilize
```python
def on_before_optimizer_step(self, optimizer):
    # Add small noise to gradients to prevent overfitting
    if self.global_step > 6000:
        for param in self.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * 0.001
                param.grad.add_(noise)
```
**Why**: Small gradient noise can prevent model from finding sharp minima that lead to instability.

### Option 6: Exponential Moving Average (EMA) of Weights
```python
# Use EMA to smooth out weight updates
from torch_ema import ExponentialMovingAverage

ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

# After each step
ema.update()

# For inference, use EMA weights
with ema.average_parameters():
    generate(...)
```
**Why**: EMA weights are smoother and less prone to instability.

## Example Outputs

The example outputs were generated using [generate.py](./scripts/generate.py).

### Prompt: "Once upon a time"
```
I thank you, if you shall not be a match
Becomes it be a pen.

LEONTES:
A most grieve
But no more.

ANTIGONUS:
You are madam,
We'll not; I must not call me leave to
To have her.

ANTIGONUS:
I'll do my lord,
Do not displeased my lord, but not be in this,
To have
```

### Prompt: "The king said"
```
The king said, and the rest,
To set the seat in your favour,--

GLOUCESTER:
I do it you this day, my noble duke.

BUCKINGHAM:
What, you please you to our complices?

GLOUCESTER:
Then is the Tower, the Tower, the Tower.

BUCKINGHAM:
And look to the city we have it so?

GLOUC
```

### Prompt: "In a land far away"
```
In a land far away!

CLIFFORD:
I cannot my heart, to the heavens, and friends?

RICHARD:
Ay, Clifford, away!

RICHARD:
Why, wilt thou be that slew my life's blood
That makes these wounds the house of York,
And not let my fear the tide to the crown.

WARWICK:
This is the brat, and so much before.

KING HENRY VI
```

### Prompt: "Who goes there?"
Who goes there?

DUKE OF AUMERLE:
I brought my lord, no doubt, and I know;
Lest in your son, as thou know,
To have been still no cause to me
Of all the justice of his grace
Of Bolingbroke, and that, to speak:
I am your grace, and that for that
My father of England, and he shall quickly,
From the most love and the Englishman,
My father of

### Prompt: "The throne room was"
The throne room was,
The common of those o' the state; and so say
To think must be the man that have you
Of this my nature.

PROSPERO:
'Tis time
I do so dishonour.

MIRANDA:
'Tis time
To be dangerous: but in my state
To be content you; for, not think
To bate me, by oath-night and hang'd
Awaked an enemy, which

## Key Learnings

1. **Multi-Head Latent Attention (MLHA)**:
   - Successfully compresses KV to latent space (768 → 96)
   - Reduces memory footprint while maintaining performance
   - Requires careful handling of odd head dimensions (head_dim=85)

2. **Mixture of Experts (MoE)**:
   - Loss-less load balancing works well via adaptive bias updates
   - No auxiliary loss needed
   - Expert utilization remained balanced throughout training

3. **Training Stability**:
   - Initial learning rate (3e-4) too aggressive for extended training
   - Gradient clipping (1.0) insufficient for preventing explosion
   - Early stopping or LR decay crucial for toy datasets

4. **Model Scale**:
   - 916M parameters successfully trained on single GPU
   - bf16 mixed precision essential for memory efficiency
   - Checkpoint sizes manageable with `save_top_k=3`

## Notes

- Training completed in ~3.3 hours on single GPU
- NaN losses appeared after step 8435 but training continued with batch skipping
- Best usable checkpoint: step 7000 (loss=0.414)
- For production use, retrain with lower learning rate or implement early stopping
