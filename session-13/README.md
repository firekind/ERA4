# SmolLM2-135M: Reverse Engineering & Training

## Overview

This project reverse engineers the SmolLM2-135M model architecture from scratch by analyzing the official HuggingFace repository, configuration files, and pretrained weights. The model is then trained from scratch on a toy dataset and validated against the official pretrained weights.

## Architecture Discovery Process

### 1. Information Gathering
- **Source**: HuggingFace `HuggingFaceTB/SmolLM2-135M` repository
- **Key files analyzed**:
  - YAML training configuration
  - Model weights (state_dict inspection)
  - Config.json for hyperparameters

### 2. Architecture Details

SmolLM2-135M is a **Llama-style decoder-only transformer** with the following specifications:

```
Architecture: Llama-based (Pre-norm, RMSNorm, RoPE, SwiGLU MLP)

Model Hierarchy:
├── Token Embeddings (vocab_size=49,152 → hidden_size=576)
├── 30× Transformer Blocks
│   ├── RMSNorm (Pre-attention)
│   ├── Grouped Query Attention (GQA)
│   │   ├── Q: 9 query heads
│   │   ├── K,V: 3 key-value heads (GQA ratio 3:1)
│   │   └── RoPE positional encoding
│   ├── Residual Connection
│   ├── RMSNorm (Pre-MLP)
│   ├── SwiGLU MLP
│   │   ├── gate_proj: 576 → 1536
│   │   ├── up_proj: 576 → 1536
│   │   └── down_proj: 1536 → 576
│   └── Residual Connection
├── Final RMSNorm
└── LM Head (tied with embeddings)
```

### Key Configuration Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| vocab_size | 49,152 | Tokenizer: cosmo2-tokenizer |
| hidden_size | 576 | Base dimension |
| num_hidden_layers | 30 | Transformer blocks |
| num_attention_heads | 9 | Query heads |
| num_key_value_heads | 3 | KV heads (GQA) |
| head_dim | 64 | 576 / 9 |
| intermediate_size | 1,536 | MLP hidden dimension |
| max_position_embeddings | 2,048 | Context length |
| rope_theta | 100,000 | RoPE base frequency |
| tie_word_embeddings | True | Input/output embedding sharing |
| rms_norm_eps | 1e-5 | Normalization epsilon |

## Parameter Calculation

### Embedding Layer
- **Token Embeddings**: 49,152 × 576 = **28,311,552** params

### Single Transformer Block
- **Input LayerNorm**: 576 params
- **Attention**:
  - Q projection: 576 × 576 = 331,776
  - K projection: 576 × 192 = 110,592
  - V projection: 576 × 192 = 110,592
  - O projection: 576 × 576 = 331,776
  - **Subtotal**: 884,736 params
- **Post-Attention LayerNorm**: 576 params
- **MLP**:
  - gate_proj: 576 × 1,536 = 884,736
  - up_proj: 576 × 1,536 = 884,736
  - down_proj: 1,536 × 576 = 884,736
  - **Subtotal**: 2,654,208 params
- **Total per block**: 3,540,096 params

### All Layers
- **30 Transformer Blocks**: 30 × 3,540,096 = **106,202,880** params
- **Final LayerNorm**: 576 params
- **LM Head**: Tied with embeddings (0 additional params)

### Total Parameters
**28,311,552 + 106,202,880 + 576 = 134,515,008 ≈ 135M parameters**

### Model Size
- **fp32**: 135M × 4 bytes = 540 MB
- **fp16/bf16**: 135M × 2 bytes = 270 MB

## Reverse Engineering Validation

The model was validated using [verify.py](./scripts/verify.py)

### Weight Inspection Findings
```
Key observations from state_dict:
✓ embed_tokens.weight: [49152, 576]
✓ layers.*.self_attn.q_proj: [576, 576] - 9 heads confirmed
✓ layers.*.self_attn.k_proj: [192, 576] - 3 KV heads confirmed
✓ layers.*.self_attn.v_proj: [192, 576]
✓ layers.*.mlp.gate_proj: [1536, 576] - SwiGLU confirmed
✓ No bias terms in any linear layers
✓ No separate lm_head (tied embeddings confirmed)
```

## Training Logs

The model was trained for 5000 steps first, checkpointing when required. The logs are [here](./assets/first-run-stdout.txt).

## Checkpoint Save/Load Test 😂

**The Emoji Test**: Testing proper checkpoint mechanics!

The Emoji test (loading a previous test and training for 50 additional steps) matters since it tests critical production skills:
- **Optimizer state preservation**: Momentum/variance restored correctly
- **Scheduler state**: Learning rate continues from correct position
- **Global step tracking**: Step counter picks up at 5001, not reset to 0
- **Seamless resumption**: No loss spike or discontinuity

**Result**: ✅ Training resumed smoothly at step 5001 with no disruption!

The logs are [here](./assets/second-run-stdout.txt).
