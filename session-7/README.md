# Session 7: Advanced CNN Architectures for CIFAR-10

Alright, so the goal here was to build a CNN that hits 85% accuracy on CIFAR-10 with a bunch of constraints thrown in. No max pooling, gotta use dilated convs, depthwise separable convs, and keep it under 200k parameters. Oh, and the receptive field needs to be at least 44. Fun times.

## Assignment Requirements

- ✅ Works on CIFAR-10
- ✅ No MaxPooling (strided convolutions instead)
- ✅ Receptive Field ≥ 44
- ✅ Must use Depthwise Separable Convolution
- ✅ Must use Dilated Convolution
- ✅ Must use Global Average Pooling
- ✅ Augmentations: HorizontalFlip, ShiftScaleRotate, CoarseDropout
- ✅ Target: 85% accuracy
- ✅ Budget: <200k parameters

## The Journey

### Model 1: The Naive Baseline

**What I did:**
- Took the model from last session
- Swapped max pooling for strided convs
- Simple channel progression: 16 → 32 → 64
- No regularization (living dangerously)

**Stats:**
- Parameters: 92,874
- Train Acc: 84.2%
- Test Acc: 67.2%
- RF: 44 ✓

**What happened:**
Hit the RF target and stayed under 200k params, but test accuracy was all over the place. Classic overfitting - the model was basically memorizing the training data. Expected since I stripped out all regularization, but hey, at least the capacity seemed okay.

**Training Logs**:
```
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs

  | Name      | Type               | Params | Mode 
---------------------------------------------------------
0 | block1    | Sequential         | 3.9 K  | train
1 | block2    | Sequential         | 27.5 K | train
2 | block3    | Sequential         | 61.5 K | train
3 | train_acc | MulticlassAccuracy | 0      | train
4 | val_acc   | MulticlassAccuracy | 0      | train
---------------------------------------------------------
92.9 K    Trainable params
0         Non-trainable params
92.9 K    Total params
0.371     Total estimated model params size (MB)
43        Modules in train mode
0         Modules in eval mode
Training: |          | 0/? [00:00<?, ?it/s]                                
Epoch 0: 100%|██████████| 782/782 [00:06<00:00, 124.43it/s, val_loss=1.660, val_acc=0.380, train_loss=1.740, train_acc=0.333]
Epoch 1: 100%|██████████| 782/782 [00:05<00:00, 153.71it/s, val_loss=2.150, val_acc=0.351, train_loss=1.390, train_acc=0.483]
Epoch 2: 100%|██████████| 782/782 [00:04<00:00, 157.00it/s, val_loss=1.850, val_acc=0.399, train_loss=1.200, train_acc=0.564]
Epoch 3: 100%|██████████| 782/782 [00:04<00:00, 159.02it/s, val_loss=1.210, val_acc=0.561, train_loss=1.060, train_acc=0.617]
Epoch 4: 100%|██████████| 782/782 [00:05<00:00, 151.73it/s, val_loss=1.220, val_acc=0.566, train_loss=0.967, train_acc=0.652]
Epoch 5: 100%|██████████| 782/782 [00:05<00:00, 151.64it/s, val_loss=1.760, val_acc=0.458, train_loss=0.892, train_acc=0.682]
Epoch 6: 100%|██████████| 782/782 [00:05<00:00, 153.20it/s, val_loss=0.916, val_acc=0.669, train_loss=0.837, train_acc=0.701]
Epoch 7: 100%|██████████| 782/782 [00:05<00:00, 152.54it/s, val_loss=1.410, val_acc=0.552, train_loss=0.781, train_acc=0.722]
Epoch 8: 100%|██████████| 782/782 [00:05<00:00, 153.33it/s, val_loss=1.150, val_acc=0.623, train_loss=0.737, train_acc=0.739]
Epoch 9: 100%|██████████| 782/782 [00:05<00:00, 153.03it/s, val_loss=1.160, val_acc=0.619, train_loss=0.700, train_acc=0.753]
Epoch 10: 100%|██████████| 782/782 [00:05<00:00, 152.51it/s, val_loss=1.080, val_acc=0.632, train_loss=0.666, train_acc=0.764]
Epoch 11: 100%|██████████| 782/782 [00:04<00:00, 159.46it/s, val_loss=1.370, val_acc=0.579, train_loss=0.636, train_acc=0.777]
Epoch 12: 100%|██████████| 782/782 [00:05<00:00, 151.63it/s, val_loss=0.903, val_acc=0.695, train_loss=0.608, train_acc=0.787]
Epoch 13: 100%|██████████| 782/782 [00:05<00:00, 151.43it/s, val_loss=0.944, val_acc=0.687, train_loss=0.578, train_acc=0.796]
Epoch 14: 100%|██████████| 782/782 [00:05<00:00, 152.24it/s, val_loss=0.954, val_acc=0.690, train_loss=0.557, train_acc=0.808]
Epoch 15: 100%|██████████| 782/782 [00:05<00:00, 152.92it/s, val_loss=0.910, val_acc=0.703, train_loss=0.534, train_acc=0.812]
Epoch 16: 100%|██████████| 782/782 [00:05<00:00, 151.63it/s, val_loss=1.200, val_acc=0.639, train_loss=0.512, train_acc=0.822]
Epoch 17: 100%|██████████| 782/782 [00:05<00:00, 149.97it/s, val_loss=1.020, val_acc=0.682, train_loss=0.492, train_acc=0.828]
Epoch 18: 100%|██████████| 782/782 [00:05<00:00, 152.68it/s, val_loss=0.751, val_acc=0.746, train_loss=0.476, train_acc=0.832]
Epoch 19: 100%|██████████| 782/782 [00:05<00:00, 152.93it/s, val_loss=1.080, val_acc=0.672, train_loss=0.452, train_acc=0.842]
`Trainer.fit` stopped: `max_epochs=20` reached.
```

**Files:**
- [Notebook](./notebooks/model_1.ipynb)
- [Model Code](./src/session_7/model_1.py)
- [RF Calculation](https://docs.google.com/spreadsheets/d/1cOfIpeqaxG7mwUgDyGro2m8-6rP3tUsCOZQlD0vqVEU/edit?usp=sharing)

---

### Model 2: Adding Dropout

**What I did:**
- Same architecture as Model 1
- Added dropout to reduce overfitting

**Stats:**
- Parameters: 92,874 (no change)
- Train Acc: 73.7%
- Test Acc: 68.2%

**What happened:**
Dropout helped! The train-test gap shrunk a lot. Still not great, but at least the model wasn't memorizing everything anymore. The 64→10 channel drop at the end was still bugging me though - seemed like we were throwing away a lot of information there.

**Training Logs:**
```
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs

  | Name      | Type               | Params | Mode 
---------------------------------------------------------
0 | block1    | Sequential         | 3.9 K  | train
1 | block2    | Sequential         | 27.5 K | train
2 | block3    | Sequential         | 61.5 K | train
3 | train_acc | MulticlassAccuracy | 0      | train
4 | val_acc   | MulticlassAccuracy | 0      | train
---------------------------------------------------------
92.9 K    Trainable params
0         Non-trainable params
92.9 K    Total params
0.371     Total estimated model params size (MB)
44        Modules in train mode
0         Modules in eval mode
Training: |          | 0/? [00:00<?, ?it/s]                                
Epoch 0: 100%|██████████| 391/391 [00:06<00:00, 63.70it/s, val_loss=1.720, val_acc=0.338, train_loss=1.880, train_acc=0.284]
Epoch 1: 100%|██████████| 391/391 [00:02<00:00, 135.23it/s, val_loss=1.600, val_acc=0.392, train_loss=1.640, train_acc=0.369]
Epoch 2: 100%|██████████| 391/391 [00:02<00:00, 135.77it/s, val_loss=1.400, val_acc=0.476, train_loss=1.500, train_acc=0.436]
Epoch 3: 100%|██████████| 391/391 [00:02<00:00, 136.07it/s, val_loss=1.290, val_acc=0.525, train_loss=1.370, train_acc=0.491]
Epoch 4: 100%|██████████| 391/391 [00:03<00:00, 129.75it/s, val_loss=1.270, val_acc=0.532, train_loss=1.280, train_acc=0.526]
Epoch 5: 100%|██████████| 391/391 [00:02<00:00, 130.65it/s, val_loss=1.250, val_acc=0.546, train_loss=1.210, train_acc=0.559]
Epoch 6: 100%|██████████| 391/391 [00:03<00:00, 130.29it/s, val_loss=1.140, val_acc=0.586, train_loss=1.140, train_acc=0.586]
Epoch 7: 100%|██████████| 391/391 [00:02<00:00, 130.86it/s, val_loss=1.090, val_acc=0.607, train_loss=1.080, train_acc=0.607]
Epoch 8: 100%|██████████| 391/391 [00:02<00:00, 130.56it/s, val_loss=1.340, val_acc=0.513, train_loss=1.030, train_acc=0.629]
Epoch 9: 100%|██████████| 391/391 [00:03<00:00, 130.26it/s, val_loss=1.020, val_acc=0.635, train_loss=0.992, train_acc=0.641]
Epoch 10: 100%|██████████| 391/391 [00:02<00:00, 130.38it/s, val_loss=1.040, val_acc=0.625, train_loss=0.954, train_acc=0.655]
Epoch 11: 100%|██████████| 391/391 [00:02<00:00, 132.35it/s, val_loss=0.928, val_acc=0.666, train_loss=0.922, train_acc=0.669]
Epoch 12: 100%|██████████| 391/391 [00:02<00:00, 132.16it/s, val_loss=0.979, val_acc=0.648, train_loss=0.895, train_acc=0.676]
Epoch 13: 100%|██████████| 391/391 [00:02<00:00, 132.06it/s, val_loss=0.927, val_acc=0.666, train_loss=0.865, train_acc=0.689]
Epoch 14: 100%|██████████| 391/391 [00:02<00:00, 131.59it/s, val_loss=0.953, val_acc=0.661, train_loss=0.839, train_acc=0.696]
Epoch 15: 100%|██████████| 391/391 [00:02<00:00, 132.76it/s, val_loss=0.868, val_acc=0.691, train_loss=0.820, train_acc=0.707]
Epoch 16: 100%|██████████| 391/391 [00:02<00:00, 130.90it/s, val_loss=0.928, val_acc=0.675, train_loss=0.795, train_acc=0.716]
Epoch 17: 100%|██████████| 391/391 [00:02<00:00, 132.30it/s, val_loss=0.889, val_acc=0.684, train_loss=0.775, train_acc=0.723]
Epoch 18: 100%|██████████| 391/391 [00:02<00:00, 131.58it/s, val_loss=0.827, val_acc=0.703, train_loss=0.758, train_acc=0.729]
Epoch 19: 100%|██████████| 391/391 [00:02<00:00, 132.05it/s, val_loss=0.899, val_acc=0.682, train_loss=0.739, train_acc=0.737]
`Trainer.fit` stopped: `max_epochs=20` reached.
```

**Files:**
- [Notebook](./notebooks/model_2.ipynb)
- [Model Code](./src/session_7/model_2.py)
- [RF Calculation](https://docs.google.com/spreadsheets/d/1cOfIpeqaxG7mwUgDyGro2m8-6rP3tUsCOZQlD0vqVEU/edit?usp=sharing)

---

### Model 3: The Pyramid Experiment

**What I did:**
Changed to a pyramid-style architecture with expand-squeeze patterns:
- Block 1: 3 → 16 → 32
- Blocks 2 & 3: 32 → 48 → 48 → 32 (expand, process, squeeze with 1x1)
- Block 4: 32 → 10 → GAP

**Stats:**
- Parameters: 87,582 (5k fewer!)
- Train Acc: 76%
- Test Acc: 71.4%
- RF: 52

**Training Logs:**
```
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs

  | Name      | Type               | Params | Mode 
---------------------------------------------------------
0 | block1    | Sequential         | 9.4 K  | train
1 | block2    | Sequential         | 34.3 K | train
2 | block3    | Sequential         | 34.3 K | train
3 | block4    | Sequential         | 9.7 K  | train
4 | train_acc | MulticlassAccuracy | 0      | train
5 | val_acc   | MulticlassAccuracy | 0      | train
---------------------------------------------------------
87.6 K    Trainable params
0         Non-trainable params
87.6 K    Total params
0.350     Total estimated model params size (MB)
61        Modules in train mode
0         Modules in eval mode
Training: |          | 0/? [00:00<?, ?it/s]                                
Epoch 0: 100%|██████████| 391/391 [00:04<00:00, 80.33it/s, val_loss=1.840, val_acc=0.366, train_loss=2.030, train_acc=0.273]
Epoch 1: 100%|██████████| 391/391 [00:03<00:00, 100.40it/s, val_loss=1.680, val_acc=0.429, train_loss=1.780, train_acc=0.386]
Epoch 2: 100%|██████████| 391/391 [00:03<00:00, 100.41it/s, val_loss=1.500, val_acc=0.487, train_loss=1.610, train_acc=0.445]
Epoch 3: 100%|██████████| 391/391 [00:03<00:00, 100.56it/s, val_loss=1.420, val_acc=0.514, train_loss=1.490, train_acc=0.497]
Epoch 4: 100%|██████████| 391/391 [00:03<00:00, 100.19it/s, val_loss=1.320, val_acc=0.556, train_loss=1.380, train_acc=0.535]
Epoch 5: 100%|██████████| 391/391 [00:03<00:00, 100.24it/s, val_loss=1.300, val_acc=0.556, train_loss=1.290, train_acc=0.567]
Epoch 6: 100%|██████████| 391/391 [00:03<00:00, 99.90it/s, val_loss=1.160, val_acc=0.613, train_loss=1.210, train_acc=0.594] 
Epoch 7: 100%|██████████| 391/391 [00:03<00:00, 99.91it/s, val_loss=1.150, val_acc=0.613, train_loss=1.140, train_acc=0.620] 
Epoch 8: 100%|██████████| 391/391 [00:03<00:00, 99.79it/s, val_loss=1.080, val_acc=0.636, train_loss=1.080, train_acc=0.635] 
Epoch 9: 100%|██████████| 391/391 [00:03<00:00, 100.01it/s, val_loss=1.100, val_acc=0.625, train_loss=1.040, train_acc=0.655]
Epoch 10: 100%|██████████| 391/391 [00:03<00:00, 100.52it/s, val_loss=0.988, val_acc=0.669, train_loss=0.990, train_acc=0.671]
Epoch 11: 100%|██████████| 391/391 [00:03<00:00, 100.27it/s, val_loss=0.975, val_acc=0.668, train_loss=0.951, train_acc=0.683]
Epoch 12: 100%|██████████| 391/391 [00:03<00:00, 100.49it/s, val_loss=1.010, val_acc=0.658, train_loss=0.917, train_acc=0.693]
Epoch 13: 100%|██████████| 391/391 [00:03<00:00, 98.96it/s, val_loss=0.987, val_acc=0.665, train_loss=0.883, train_acc=0.705] 
Epoch 14: 100%|██████████| 391/391 [00:03<00:00, 97.98it/s, val_loss=0.924, val_acc=0.683, train_loss=0.855, train_acc=0.716] 
Epoch 15: 100%|██████████| 391/391 [00:03<00:00, 97.79it/s, val_loss=0.957, val_acc=0.677, train_loss=0.824, train_acc=0.726] 
Epoch 16: 100%|██████████| 391/391 [00:03<00:00, 99.63it/s, val_loss=0.922, val_acc=0.686, train_loss=0.802, train_acc=0.734] 
Epoch 17: 100%|██████████| 391/391 [00:03<00:00, 100.27it/s, val_loss=0.877, val_acc=0.702, train_loss=0.774, train_acc=0.742]
Epoch 18: 100%|██████████| 391/391 [00:03<00:00, 99.60it/s, val_loss=0.915, val_acc=0.692, train_loss=0.754, train_acc=0.748] 
Epoch 19: 100%|██████████| 391/391 [00:03<00:00, 99.84it/s, val_loss=0.845, val_acc=0.714, train_loss=0.728, train_acc=0.760] 
`Trainer.fit` stopped: `max_epochs=20` reached.
```

**What happened:**
This was interesting. Fewer parameters, bigger RF (52 vs 44), and more stable training. The expand-squeeze pattern seemed to work better than just throwing away channels at the end. Training converged faster and both accuracies were more stable across epochs. This became my new baseline.

**Files:**
- [Notebook](./notebooks/model_3.ipynb)
- [Model Code](./src/session_7/model_3.py)
- [RF Calculation](https://docs.google.com/spreadsheets/d/1cOfIpeqaxG7mwUgDyGro2m8-6rP3tUsCOZQlD0vqVEU/edit?usp=sharing)

---

### Model 4: The Final Boss

**What I did:**
Took Model 3 and added all the required bells and whistles:
- **Depthwise Separable Convolutions** in blocks 2 & 3 (saves a ton of params!)
- **Dilated Convolutions** (dilation=2) in blocks 2 & 3 for that RF boost
- Kept the pyramid structure
- Added all the required augmentations (HorizontalFlip, ShiftScaleRotate, CoarseDropout)
- Used **OneCycleLR scheduler** because training was struggling without it

Architecture:
- Block 1: 3→16→32, strided downsample
- Block 2: DSC(32→32), Dilated(32→48), strided, squeeze(48→32), dropout
- Block 3: DSC(32→32), Dilated(32→48), strided, squeeze(48→32), dropout
- Block 4: 32→32→10, GAP

**Stats:**
- Parameters: 71,838 (only 36% of budget used!)
- Train Acc: 80.7%
- Test Acc: **85.1%** 🎉
- RF: ≥44 ✓
- Epochs to target: 50

**Training Logs:**
```
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Loading `train_dataloader` to estimate number of stepping batches.

  | Name      | Type               | Params | Mode 
---------------------------------------------------------
0 | block1    | Sequential         | 9.4 K  | train
1 | block2    | Sequential         | 26.4 K | train
2 | block3    | Sequential         | 26.4 K | train
3 | block4    | Sequential         | 9.7 K  | train
4 | train_acc | MulticlassAccuracy | 0      | train
5 | val_acc   | MulticlassAccuracy | 0      | train
---------------------------------------------------------
71.8 K    Trainable params
0         Non-trainable params
71.8 K    Total params
0.287     Total estimated model params size (MB)
63        Modules in train mode
0         Modules in eval mode
Training: |          | 0/? [00:00<?, ?it/s]                                
Epoch 0: 100%|██████████| 391/391 [00:08<00:00, 47.13it/s, val_loss=1.540, val_acc=0.450, train_loss=1.900, train_acc=0.314]
Epoch 1: 100%|██████████| 391/391 [00:04<00:00, 92.95it/s, val_loss=1.410, val_acc=0.498, train_loss=1.580, train_acc=0.435]
Epoch 2: 100%|██████████| 391/391 [00:04<00:00, 92.19it/s, val_loss=1.270, val_acc=0.553, train_loss=1.410, train_acc=0.496]
Epoch 3: 100%|██████████| 391/391 [00:04<00:00, 92.99it/s, val_loss=1.130, val_acc=0.608, train_loss=1.300, train_acc=0.534]
Epoch 4: 100%|██████████| 391/391 [00:04<00:00, 92.63it/s, val_loss=1.080, val_acc=0.620, train_loss=1.220, train_acc=0.565]
Epoch 5: 100%|██████████| 391/391 [00:04<00:00, 92.91it/s, val_loss=1.020, val_acc=0.647, train_loss=1.140, train_acc=0.596]
Epoch 6: 100%|██████████| 391/391 [00:04<00:00, 92.97it/s, val_loss=0.947, val_acc=0.663, train_loss=1.070, train_acc=0.621]
Epoch 7: 100%|██████████| 391/391 [00:04<00:00, 93.13it/s, val_loss=0.929, val_acc=0.672, train_loss=1.030, train_acc=0.638]
Epoch 8: 100%|██████████| 391/391 [00:04<00:00, 92.65it/s, val_loss=0.808, val_acc=0.722, train_loss=0.978, train_acc=0.654]
Epoch 9: 100%|██████████| 391/391 [00:04<00:00, 91.43it/s, val_loss=0.792, val_acc=0.727, train_loss=0.949, train_acc=0.667]
Epoch 10: 100%|██████████| 391/391 [00:04<00:00, 91.32it/s, val_loss=0.780, val_acc=0.725, train_loss=0.922, train_acc=0.679]
Epoch 11: 100%|██████████| 391/391 [00:04<00:00, 92.25it/s, val_loss=0.804, val_acc=0.719, train_loss=0.898, train_acc=0.686]
Epoch 12: 100%|██████████| 391/391 [00:04<00:00, 91.79it/s, val_loss=0.704, val_acc=0.757, train_loss=0.876, train_acc=0.694]
Epoch 13: 100%|██████████| 391/391 [00:04<00:00, 92.32it/s, val_loss=0.681, val_acc=0.766, train_loss=0.853, train_acc=0.702]
Epoch 14: 100%|██████████| 391/391 [00:04<00:00, 91.70it/s, val_loss=0.665, val_acc=0.771, train_loss=0.836, train_acc=0.707]
Epoch 15: 100%|██████████| 391/391 [00:04<00:00, 92.12it/s, val_loss=0.632, val_acc=0.780, train_loss=0.818, train_acc=0.713]
Epoch 16: 100%|██████████| 391/391 [00:04<00:00, 91.76it/s, val_loss=0.610, val_acc=0.793, train_loss=0.805, train_acc=0.717]
Epoch 17: 100%|██████████| 391/391 [00:04<00:00, 91.73it/s, val_loss=0.762, val_acc=0.738, train_loss=0.788, train_acc=0.723]
Epoch 18: 100%|██████████| 391/391 [00:04<00:00, 91.98it/s, val_loss=0.661, val_acc=0.777, train_loss=0.772, train_acc=0.730]
Epoch 19: 100%|██████████| 391/391 [00:04<00:00, 91.09it/s, val_loss=0.580, val_acc=0.801, train_loss=0.772, train_acc=0.731]
Epoch 20: 100%|██████████| 391/391 [00:04<00:00, 91.48it/s, val_loss=0.585, val_acc=0.797, train_loss=0.763, train_acc=0.733]
Epoch 21: 100%|██████████| 391/391 [00:04<00:00, 91.25it/s, val_loss=0.590, val_acc=0.801, train_loss=0.743, train_acc=0.740]
Epoch 22: 100%|██████████| 391/391 [00:04<00:00, 91.73it/s, val_loss=0.599, val_acc=0.792, train_loss=0.742, train_acc=0.741]
Epoch 23: 100%|██████████| 391/391 [00:04<00:00, 91.33it/s, val_loss=0.592, val_acc=0.798, train_loss=0.737, train_acc=0.743]
Epoch 24: 100%|██████████| 391/391 [00:04<00:00, 91.84it/s, val_loss=0.599, val_acc=0.792, train_loss=0.723, train_acc=0.749]
Epoch 25: 100%|██████████| 391/391 [00:04<00:00, 91.49it/s, val_loss=0.539, val_acc=0.814, train_loss=0.705, train_acc=0.754]
Epoch 26: 100%|██████████| 391/391 [00:04<00:00, 91.28it/s, val_loss=0.541, val_acc=0.815, train_loss=0.705, train_acc=0.755]
Epoch 27: 100%|██████████| 391/391 [00:04<00:00, 91.52it/s, val_loss=0.552, val_acc=0.811, train_loss=0.695, train_acc=0.757]
Epoch 28: 100%|██████████| 391/391 [00:04<00:00, 91.67it/s, val_loss=0.553, val_acc=0.814, train_loss=0.696, train_acc=0.759]
Epoch 29: 100%|██████████| 391/391 [00:04<00:00, 91.64it/s, val_loss=0.517, val_acc=0.826, train_loss=0.688, train_acc=0.760]
Epoch 30: 100%|██████████| 391/391 [00:04<00:00, 91.52it/s, val_loss=0.529, val_acc=0.823, train_loss=0.672, train_acc=0.766]
Epoch 31: 100%|██████████| 391/391 [00:04<00:00, 91.34it/s, val_loss=0.527, val_acc=0.822, train_loss=0.678, train_acc=0.762]
Epoch 32: 100%|██████████| 391/391 [00:04<00:00, 91.32it/s, val_loss=0.513, val_acc=0.827, train_loss=0.667, train_acc=0.767]
Epoch 33: 100%|██████████| 391/391 [00:04<00:00, 90.99it/s, val_loss=0.514, val_acc=0.823, train_loss=0.661, train_acc=0.770]
Epoch 34: 100%|██████████| 391/391 [00:04<00:00, 87.53it/s, val_loss=0.486, val_acc=0.831, train_loss=0.655, train_acc=0.772]
Epoch 35: 100%|██████████| 391/391 [00:04<00:00, 85.10it/s, val_loss=0.488, val_acc=0.833, train_loss=0.651, train_acc=0.773]
Epoch 36: 100%|██████████| 391/391 [00:04<00:00, 82.43it/s, val_loss=0.487, val_acc=0.834, train_loss=0.636, train_acc=0.780]
Epoch 37: 100%|██████████| 391/391 [00:04<00:00, 78.27it/s, val_loss=0.470, val_acc=0.839, train_loss=0.636, train_acc=0.776]
Epoch 38: 100%|██████████| 391/391 [00:05<00:00, 75.34it/s, val_loss=0.480, val_acc=0.837, train_loss=0.626, train_acc=0.783]
Epoch 39: 100%|██████████| 391/391 [00:05<00:00, 72.99it/s, val_loss=0.474, val_acc=0.840, train_loss=0.613, train_acc=0.787]
Epoch 40: 100%|██████████| 391/391 [00:05<00:00, 73.50it/s, val_loss=0.464, val_acc=0.843, train_loss=0.610, train_acc=0.788]
Epoch 41: 100%|██████████| 391/391 [00:05<00:00, 69.69it/s, val_loss=0.456, val_acc=0.846, train_loss=0.601, train_acc=0.789]
Epoch 42: 100%|██████████| 391/391 [00:05<00:00, 68.41it/s, val_loss=0.452, val_acc=0.848, train_loss=0.593, train_acc=0.794]
Epoch 43: 100%|██████████| 391/391 [00:05<00:00, 67.61it/s, val_loss=0.445, val_acc=0.850, train_loss=0.584, train_acc=0.796]
Epoch 44: 100%|██████████| 391/391 [00:05<00:00, 68.94it/s, val_loss=0.440, val_acc=0.851, train_loss=0.579, train_acc=0.799]
Epoch 45: 100%|██████████| 391/391 [00:05<00:00, 69.02it/s, val_loss=0.440, val_acc=0.850, train_loss=0.578, train_acc=0.798]
Epoch 46: 100%|██████████| 391/391 [00:05<00:00, 69.15it/s, val_loss=0.442, val_acc=0.850, train_loss=0.571, train_acc=0.801]
Epoch 47: 100%|██████████| 391/391 [00:05<00:00, 76.70it/s, val_loss=0.436, val_acc=0.850, train_loss=0.575, train_acc=0.799]
Epoch 48: 100%|██████████| 391/391 [00:05<00:00, 78.05it/s, val_loss=0.438, val_acc=0.850, train_loss=0.566, train_acc=0.804]
Epoch 49: 100%|██████████| 391/391 [00:05<00:00, 77.53it/s, val_loss=0.437, val_acc=0.851, train_loss=0.559, train_acc=0.807]
`Trainer.fit` stopped: `max_epochs=50` reached.
```

**What happened:**
We did it! The depthwise separable convs saved a bunch of parameters (like 8-9x fewer than regular convs), and the dilated convs helped push the RF up without adding depth. The OneCycleLR scheduler was clutch - without it, the model was stuck in the low 70s. With aggressive augmentations, you really need that learning rate scheduling to help the model escape local minima.

The train-test gap is small (~4%), which means we're generalizing well. The model takes its time to converge (around 35-40 epochs), but it gets there.

**Files:**
- [Notebook](./notebooks/model_4.ipynb)
- [Model Code](./src/session_7/model_4.py)
- [RF Calculation](https://docs.google.com/spreadsheets/d/1cOfIpeqaxG7mwUgDyGro2m8-6rP3tUsCOZQlD0vqVEU/edit?usp=sharing)

## Key Takeaways

1. **Depthwise Separable Convolutions are magic** - Seriously, they save so many parameters with barely any accuracy loss. Use them.

2. **Dilated convolutions are sneaky good** - Want a bigger receptive field without going deeper? Dilation=2 turns a 3×3 conv into an effective 5×5. Neat trick.

3. **OneCycleLR is not optional** - At least not with aggressive augmentation. The model was stuck until I added the scheduler. Learning rate scheduling matters, people.

4. **Expand-squeeze > abrupt channel drops** - Don't go from 64→10 in one shot. Give the network a chance to compress information gradually.

5. **Aggressive augmentation = more epochs** - Those required augmentations are VERY aggressive. The model needs time (50 epochs) to learn through all that noise.

6. **Parameter efficiency is real** - Hit the target with only 71k params out of 200k budget. Efficiency for the win.

## Final Architecture Summary

- ✅ Total Parameters: 71,838
- ✅ Receptive Field: ≥44
- ✅ Validation Accuracy: 85.1%
- ✅ No MaxPooling
- ✅ Depthwise Separable Conv (blocks 2 & 3)
- ✅ Dilated Conv (blocks 2 & 3, dilation=2)
- ✅ Global Average Pooling
- ✅ All required augmentations
- ✅ <200k parameters
