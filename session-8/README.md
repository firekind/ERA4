# ResNet 50 trained on CIFAR-100

Trained ResNet 50 on CIFAR-100. Hit 79.4% accuracy after 100 epochs. The following changes was made to the default ResNet-50 architecture:

1. Changed first conv from 7x7 stride 2 to 3x3 stride 1
2. Removed maxpool (would kill spatial resolution on small images)

notebook used for training: [./notebooks/training.ipynb](./notebooks/training.ipynb)

huggingface space: https://huggingface.co/spaces/firekind/resnet-cifar100


## Training Logs:

```
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
HPU available: False, using: 0 HPUs
Loading `train_dataloader` to estimate number of stepping batches.

  | Name      | Type               | Params | Mode 
---------------------------------------------------------
0 | model     | ResNet             | 23.7 M | train
1 | loss_fn   | CrossEntropyLoss   | 0      | train
2 | train_acc | MulticlassAccuracy | 0      | train
3 | val_acc   | MulticlassAccuracy | 0      | train
---------------------------------------------------------
23.7 M    Trainable params
0         Non-trainable params
23.7 M    Total params
94.821    Total estimated model params size (MB)
154       Modules in train mode
0         Modules in eval mode
Training: |          | 0/? [00:00<?, ?it/s]                                
Epoch 0: 100%|██████████| 391/391 [02:20<00:00,  2.78it/s, v_num=1, val_loss=3.850, val_acc=0.105, train_loss=4.240, train_acc=0.0619]
Epoch 0, global step 391: 'val_acc' reached 0.10490 (best 0.10490), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=00-val_acc=0.10.ckpt' as top 3
Epoch 1: 100%|██████████| 391/391 [02:21<00:00,  2.77it/s, v_num=1, val_loss=3.370, val_acc=0.194, train_loss=3.610, train_acc=0.147] 
Epoch 1, global step 782: 'val_acc' reached 0.19450 (best 0.19450), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=01-val_acc=0.19.ckpt' as top 3

Epoch 2: 100%|██████████| 391/391 [02:20<00:00,  2.78it/s, v_num=1, val_loss=3.100, val_acc=0.242, train_loss=3.250, train_acc=0.207]
Epoch 2, global step 1173: 'val_acc' reached 0.24200 (best 0.24200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=02-val_acc=0.24.ckpt' as top 3

Epoch 3: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=2.810, val_acc=0.293, train_loss=2.960, train_acc=0.259]
Epoch 3, global step 1564: 'val_acc' reached 0.29290 (best 0.29290), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=03-val_acc=0.29.ckpt' as top 3

Epoch 4: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=2.540, val_acc=0.349, train_loss=2.640, train_acc=0.323]
Epoch 4, global step 1955: 'val_acc' reached 0.34850 (best 0.34850), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=04-val_acc=0.35.ckpt' as top 3

Epoch 5: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=2.440, val_acc=0.375, train_loss=2.360, train_acc=0.375]
Epoch 5, global step 2346: 'val_acc' reached 0.37450 (best 0.37450), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=05-val_acc=0.37.ckpt' as top 3

Epoch 6: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=2.250, val_acc=0.418, train_loss=2.110, train_acc=0.431]
Epoch 6, global step 2737: 'val_acc' reached 0.41780 (best 0.41780), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=06-val_acc=0.42.ckpt' as top 3
Epoch 7: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=2.120, val_acc=0.441, train_loss=1.920, train_acc=0.473]
Epoch 7, global step 3128: 'val_acc' reached 0.44100 (best 0.44100), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=07-val_acc=0.44.ckpt' as top 3

Epoch 8: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.980, val_acc=0.488, train_loss=1.770, train_acc=0.509]
Epoch 8, global step 3519: 'val_acc' reached 0.48780 (best 0.48780), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=08-val_acc=0.49.ckpt' as top 3

Epoch 9: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.910, val_acc=0.493, train_loss=1.630, train_acc=0.541]
Epoch 9, global step 3910: 'val_acc' reached 0.49260 (best 0.49260), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=09-val_acc=0.49.ckpt' as top 3

Epoch 10: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.820, val_acc=0.518, train_loss=1.530, train_acc=0.566]
Epoch 10, global step 4301: 'val_acc' reached 0.51760 (best 0.51760), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=10-val_acc=0.52.ckpt' as top 3

Epoch 11: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.880, val_acc=0.512, train_loss=1.450, train_acc=0.587]
Epoch 11, global step 4692: 'val_acc' reached 0.51240 (best 0.51760), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=11-val_acc=0.51.ckpt' as top 3

Epoch 12: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.910, val_acc=0.496, train_loss=1.370, train_acc=0.609]
Epoch 12, global step 5083: 'val_acc' reached 0.49640 (best 0.51760), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=12-val_acc=0.50.ckpt' as top 3

Epoch 13: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.930, val_acc=0.491, train_loss=1.320, train_acc=0.619]
Epoch 13, global step 5474: 'val_acc' was not in top 3

Epoch 14: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.650, val_acc=0.554, train_loss=1.260, train_acc=0.632]
Epoch 14, global step 5865: 'val_acc' reached 0.55410 (best 0.55410), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=14-val_acc=0.55.ckpt' as top 3

Epoch 15: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.610, val_acc=0.563, train_loss=1.220, train_acc=0.646]
Epoch 15, global step 6256: 'val_acc' reached 0.56340 (best 0.56340), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=15-val_acc=0.56.ckpt' as top 3

Epoch 16: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.540, val_acc=0.577, train_loss=1.180, train_acc=0.654]
Epoch 16, global step 6647: 'val_acc' reached 0.57720 (best 0.57720), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=16-val_acc=0.58.ckpt' as top 3

Epoch 17: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.580, val_acc=0.567, train_loss=1.150, train_acc=0.660]
Epoch 17, global step 7038: 'val_acc' reached 0.56670 (best 0.57720), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=17-val_acc=0.57.ckpt' as top 3

Epoch 18: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.560, val_acc=0.583, train_loss=1.120, train_acc=0.672]
Epoch 18, global step 7429: 'val_acc' reached 0.58340 (best 0.58340), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=18-val_acc=0.58.ckpt' as top 3
Epoch 19: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.600, val_acc=0.573, train_loss=1.110, train_acc=0.672]
Epoch 19, global step 7820: 'val_acc' reached 0.57290 (best 0.58340), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=19-val_acc=0.57.ckpt' as top 3

Epoch 20: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.430, val_acc=0.602, train_loss=1.080, train_acc=0.681]
Epoch 20, global step 8211: 'val_acc' reached 0.60200 (best 0.60200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=20-val_acc=0.60.ckpt' as top 3

Epoch 21: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.610, val_acc=0.559, train_loss=1.080, train_acc=0.682]
Epoch 21, global step 8602: 'val_acc' was not in top 3

Epoch 22: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.650, val_acc=0.567, train_loss=1.060, train_acc=0.690]
Epoch 22, global step 8993: 'val_acc' was not in top 3

Epoch 23: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.690, val_acc=0.554, train_loss=1.050, train_acc=0.690]
Epoch 23, global step 9384: 'val_acc' was not in top 3

Epoch 24: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.980, val_acc=0.511, train_loss=1.040, train_acc=0.691]
Epoch 24, global step 9775: 'val_acc' was not in top 3

Epoch 25: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.750, val_acc=0.541, train_loss=1.040, train_acc=0.692]
Epoch 25, global step 10166: 'val_acc' was not in top 3

Epoch 26: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.760, val_acc=0.546, train_loss=1.030, train_acc=0.699]
Epoch 26, global step 10557: 'val_acc' was not in top 3

Epoch 27: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.610, val_acc=0.573, train_loss=1.020, train_acc=0.698]
Epoch 27, global step 10948: 'val_acc' was not in top 3

Epoch 28: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.590, val_acc=0.580, train_loss=1.000, train_acc=0.702]
Epoch 28, global step 11339: 'val_acc' reached 0.58000 (best 0.60200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=28-val_acc=0.58.ckpt' as top 3

Epoch 29: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.540, val_acc=0.585, train_loss=0.986, train_acc=0.708]
Epoch 29, global step 11730: 'val_acc' reached 0.58460 (best 0.60200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=29-val_acc=0.58.ckpt' as top 3

Epoch 30: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.640, val_acc=0.567, train_loss=0.979, train_acc=0.710]
Epoch 30, global step 12121: 'val_acc' was not in top 3

Epoch 31: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.650, val_acc=0.560, train_loss=0.973, train_acc=0.712]
Epoch 31, global step 12512: 'val_acc' was not in top 3

Epoch 32: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.490, val_acc=0.600, train_loss=0.958, train_acc=0.713]
Epoch 32, global step 12903: 'val_acc' reached 0.60020 (best 0.60200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=32-val_acc=0.60.ckpt' as top 3

Epoch 33: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.640, val_acc=0.571, train_loss=0.948, train_acc=0.719]
Epoch 33, global step 13294: 'val_acc' was not in top 3

Epoch 34: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.920, val_acc=0.532, train_loss=0.940, train_acc=0.722]
Epoch 34, global step 13685: 'val_acc' was not in top 3

Epoch 35: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.600, val_acc=0.586, train_loss=0.928, train_acc=0.723]
Epoch 35, global step 14076: 'val_acc' reached 0.58560 (best 0.60200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=35-val_acc=0.59.ckpt' as top 3

Epoch 36: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.590, val_acc=0.588, train_loss=0.922, train_acc=0.726]
Epoch 36, global step 14467: 'val_acc' reached 0.58800 (best 0.60200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=36-val_acc=0.59.ckpt' as top 3

Epoch 37: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.610, val_acc=0.585, train_loss=0.913, train_acc=0.728]
Epoch 37, global step 14858: 'val_acc' was not in top 3

Epoch 38: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.710, val_acc=0.571, train_loss=0.908, train_acc=0.731]
Epoch 38, global step 15249: 'val_acc' was not in top 3

Epoch 39: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.670, val_acc=0.580, train_loss=0.894, train_acc=0.735]
Epoch 39, global step 15640: 'val_acc' was not in top 3

Epoch 40: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.620, val_acc=0.580, train_loss=0.886, train_acc=0.734]
Epoch 40, global step 16031: 'val_acc' was not in top 3

Epoch 41: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.610, val_acc=0.581, train_loss=0.870, train_acc=0.742]
Epoch 41, global step 16422: 'val_acc' was not in top 3

Epoch 42: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.640, val_acc=0.587, train_loss=0.881, train_acc=0.736]
Epoch 42, global step 16813: 'val_acc' was not in top 3

Epoch 43: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.550, val_acc=0.594, train_loss=0.847, train_acc=0.746]
Epoch 43, global step 17204: 'val_acc' reached 0.59440 (best 0.60200), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=43-val_acc=0.59.ckpt' as top 3

Epoch 44: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.490, val_acc=0.610, train_loss=0.855, train_acc=0.744]
Epoch 44, global step 17595: 'val_acc' reached 0.61010 (best 0.61010), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=44-val_acc=0.61.ckpt' as top 3

Epoch 45: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.560, val_acc=0.596, train_loss=0.839, train_acc=0.747]
Epoch 45, global step 17986: 'val_acc' was not in top 3

Epoch 46: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.530, val_acc=0.602, train_loss=0.827, train_acc=0.751]
Epoch 46, global step 18377: 'val_acc' reached 0.60220 (best 0.61010), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=46-val_acc=0.60.ckpt' as top 3

Epoch 47: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.460, val_acc=0.618, train_loss=0.828, train_acc=0.751]
Epoch 47, global step 18768: 'val_acc' reached 0.61790 (best 0.61790), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=47-val_acc=0.62.ckpt' as top 3

Epoch 48: 100%|██████████| 391/391 [02:17<00:00,  2.83it/s, v_num=1, val_loss=1.490, val_acc=0.610, train_loss=0.809, train_acc=0.757]
Epoch 48, global step 19159: 'val_acc' reached 0.60990 (best 0.61790), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=48-val_acc=0.61.ckpt' as top 3

Epoch 49: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.500, val_acc=0.609, train_loss=0.807, train_acc=0.759]
Epoch 49, global step 19550: 'val_acc' was not in top 3

Epoch 50: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.540, val_acc=0.601, train_loss=0.794, train_acc=0.760]
Epoch 50, global step 19941: 'val_acc' was not in top 3

Epoch 51: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.430, val_acc=0.627, train_loss=0.786, train_acc=0.763]
Epoch 51, global step 20332: 'val_acc' reached 0.62700 (best 0.62700), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=51-val_acc=0.63.ckpt' as top 3

Epoch 52: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.580, val_acc=0.593, train_loss=0.775, train_acc=0.765]
Epoch 52, global step 20723: 'val_acc' was not in top 3
Epoch 53: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.360, val_acc=0.634, train_loss=0.760, train_acc=0.769]
Epoch 53, global step 21114: 'val_acc' reached 0.63440 (best 0.63440), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=53-val_acc=0.63.ckpt' as top 3

Epoch 54: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.470, val_acc=0.615, train_loss=0.757, train_acc=0.771]
Epoch 54, global step 21505: 'val_acc' was not in top 3
Epoch 55: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.720, val_acc=0.576, train_loss=0.761, train_acc=0.770]
Epoch 55, global step 21896: 'val_acc' was not in top 3

Epoch 56: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.580, val_acc=0.601, train_loss=0.733, train_acc=0.777]
Epoch 56, global step 22287: 'val_acc' was not in top 3

Epoch 57: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.430, val_acc=0.636, train_loss=0.724, train_acc=0.777]
Epoch 57, global step 22678: 'val_acc' reached 0.63580 (best 0.63580), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=57-val_acc=0.64.ckpt' as top 3

Epoch 58: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.540, val_acc=0.613, train_loss=0.721, train_acc=0.780]
Epoch 58, global step 23069: 'val_acc' was not in top 3

Epoch 59: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.430, val_acc=0.630, train_loss=0.707, train_acc=0.786]
Epoch 59, global step 23460: 'val_acc' reached 0.62970 (best 0.63580), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=59-val_acc=0.63.ckpt' as top 3

Epoch 60: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.530, val_acc=0.609, train_loss=0.689, train_acc=0.790]
Epoch 60, global step 23851: 'val_acc' was not in top 3

Epoch 61: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.480, val_acc=0.621, train_loss=0.680, train_acc=0.792]
Epoch 61, global step 24242: 'val_acc' was not in top 3
Epoch 62: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.460, val_acc=0.626, train_loss=0.666, train_acc=0.797]
Epoch 62, global step 24633: 'val_acc' was not in top 3

Epoch 63: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.320, val_acc=0.649, train_loss=0.652, train_acc=0.801]
Epoch 63, global step 25024: 'val_acc' reached 0.64900 (best 0.64900), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=63-val_acc=0.65.ckpt' as top 3

Epoch 64: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.250, val_acc=0.668, train_loss=0.633, train_acc=0.804]
Epoch 64, global step 25415: 'val_acc' reached 0.66770 (best 0.66770), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=64-val_acc=0.67.ckpt' as top 3

Epoch 65: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.470, val_acc=0.628, train_loss=0.623, train_acc=0.810]
Epoch 65, global step 25806: 'val_acc' was not in top 3

Epoch 66: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.260, val_acc=0.664, train_loss=0.609, train_acc=0.812]
Epoch 66, global step 26197: 'val_acc' reached 0.66440 (best 0.66770), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=66-val_acc=0.66.ckpt' as top 3

Epoch 67: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=1.490, val_acc=0.628, train_loss=0.589, train_acc=0.818]
Epoch 67, global step 26588: 'val_acc' was not in top 3
Epoch 68: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.230, val_acc=0.679, train_loss=0.572, train_acc=0.823]
Epoch 68, global step 26979: 'val_acc' reached 0.67890 (best 0.67890), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=68-val_acc=0.68.ckpt' as top 3

Epoch 69: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.290, val_acc=0.664, train_loss=0.548, train_acc=0.831]
Epoch 69, global step 27370: 'val_acc' reached 0.66450 (best 0.67890), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=69-val_acc=0.66.ckpt' as top 3

Epoch 70: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.280, val_acc=0.670, train_loss=0.534, train_acc=0.835]
Epoch 70, global step 27761: 'val_acc' reached 0.66960 (best 0.67890), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=70-val_acc=0.67.ckpt' as top 3

Epoch 71: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.240, val_acc=0.675, train_loss=0.514, train_acc=0.841]
Epoch 71, global step 28152: 'val_acc' reached 0.67520 (best 0.67890), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=71-val_acc=0.68.ckpt' as top 3

Epoch 72: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.170, val_acc=0.691, train_loss=0.493, train_acc=0.847]
Epoch 72, global step 28543: 'val_acc' reached 0.69130 (best 0.69130), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=72-val_acc=0.69.ckpt' as top 3

Epoch 73: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=1.190, val_acc=0.690, train_loss=0.460, train_acc=0.856]
Epoch 73, global step 28934: 'val_acc' reached 0.68970 (best 0.69130), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=73-val_acc=0.69.ckpt' as top 3

Epoch 74: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.270, val_acc=0.676, train_loss=0.447, train_acc=0.862]
Epoch 74, global step 29325: 'val_acc' was not in top 3

Epoch 75: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.210, val_acc=0.693, train_loss=0.420, train_acc=0.868]
Epoch 75, global step 29716: 'val_acc' reached 0.69340 (best 0.69340), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=75-val_acc=0.69.ckpt' as top 3

Epoch 76: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=1.300, val_acc=0.671, train_loss=0.386, train_acc=0.879]
Epoch 76, global step 30107: 'val_acc' was not in top 3

Epoch 77: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=1.160, val_acc=0.709, train_loss=0.365, train_acc=0.887]
Epoch 77, global step 30498: 'val_acc' reached 0.70890 (best 0.70890), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=77-val_acc=0.71.ckpt' as top 3
Epoch 78: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=1.170, val_acc=0.703, train_loss=0.332, train_acc=0.896]
Epoch 78, global step 30889: 'val_acc' reached 0.70350 (best 0.70890), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=78-val_acc=0.70.ckpt' as top 3

Epoch 79: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=1.130, val_acc=0.712, train_loss=0.300, train_acc=0.908]
Epoch 79, global step 31280: 'val_acc' reached 0.71240 (best 0.71240), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=79-val_acc=0.71.ckpt' as top 3

Epoch 80: 100%|██████████| 391/391 [02:19<00:00,  2.80it/s, v_num=1, val_loss=1.090, val_acc=0.730, train_loss=0.264, train_acc=0.918]
Epoch 80, global step 31671: 'val_acc' reached 0.72970 (best 0.72970), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=80-val_acc=0.73.ckpt' as top 3

Epoch 81: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=1.170, val_acc=0.715, train_loss=0.222, train_acc=0.932]
Epoch 81, global step 32062: 'val_acc' reached 0.71530 (best 0.72970), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=81-val_acc=0.72.ckpt' as top 3

Epoch 82: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=1.140, val_acc=0.728, train_loss=0.186, train_acc=0.945]
Epoch 82, global step 32453: 'val_acc' reached 0.72840 (best 0.72970), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=82-val_acc=0.73.ckpt' as top 3

Epoch 83: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=1.060, val_acc=0.739, train_loss=0.169, train_acc=0.950]
Epoch 83, global step 32844: 'val_acc' reached 0.73920 (best 0.73920), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=83-val_acc=0.74.ckpt' as top 3

Epoch 84: 100%|██████████| 391/391 [02:19<00:00,  2.80it/s, v_num=1, val_loss=1.030, val_acc=0.744, train_loss=0.131, train_acc=0.963]
Epoch 84, global step 33235: 'val_acc' reached 0.74390 (best 0.74390), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=84-val_acc=0.74.ckpt' as top 3

Epoch 85: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=0.998, val_acc=0.754, train_loss=0.102, train_acc=0.973]
Epoch 85, global step 33626: 'val_acc' reached 0.75450 (best 0.75450), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=85-val_acc=0.75.ckpt' as top 3

Epoch 86: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=0.963, val_acc=0.760, train_loss=0.0739, train_acc=0.982]
Epoch 86, global step 34017: 'val_acc' reached 0.76040 (best 0.76040), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=86-val_acc=0.76.ckpt' as top 3

Epoch 87: 100%|██████████| 391/391 [02:19<00:00,  2.80it/s, v_num=1, val_loss=0.933, val_acc=0.773, train_loss=0.0534, train_acc=0.989]
Epoch 87, global step 34408: 'val_acc' reached 0.77320 (best 0.77320), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=87-val_acc=0.77.ckpt' as top 3

Epoch 88: 100%|██████████| 391/391 [02:19<00:00,  2.80it/s, v_num=1, val_loss=0.907, val_acc=0.778, train_loss=0.0384, train_acc=0.993]
Epoch 88, global step 34799: 'val_acc' reached 0.77760 (best 0.77760), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=88-val_acc=0.78.ckpt' as top 3

Epoch 89: 100%|██████████| 391/391 [02:19<00:00,  2.81it/s, v_num=1, val_loss=0.890, val_acc=0.782, train_loss=0.0303, train_acc=0.995]
Epoch 89, global step 35190: 'val_acc' reached 0.78190 (best 0.78190), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=89-val_acc=0.78.ckpt' as top 3

Epoch 90: 100%|██████████| 391/391 [02:18<00:00,  2.81it/s, v_num=1, val_loss=0.886, val_acc=0.783, train_loss=0.0231, train_acc=0.997]
Epoch 90, global step 35581: 'val_acc' reached 0.78260 (best 0.78260), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=90-val_acc=0.78.ckpt' as top 3

Epoch 91: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=0.856, val_acc=0.786, train_loss=0.0183, train_acc=0.998]
Epoch 91, global step 35972: 'val_acc' reached 0.78630 (best 0.78630), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=91-val_acc=0.79.ckpt' as top 3

Epoch 92: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=0.853, val_acc=0.787, train_loss=0.0163, train_acc=0.998]
Epoch 92, global step 36363: 'val_acc' reached 0.78720 (best 0.78720), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=92-val_acc=0.79.ckpt' as top 3

Epoch 93: 100%|██████████| 391/391 [02:18<00:00,  2.82it/s, v_num=1, val_loss=0.845, val_acc=0.790, train_loss=0.0146, train_acc=0.998]
Epoch 93, global step 36754: 'val_acc' reached 0.78980 (best 0.78980), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=93-val_acc=0.79.ckpt' as top 3

Epoch 94: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=0.841, val_acc=0.791, train_loss=0.0127, train_acc=0.999]
Epoch 94, global step 37145: 'val_acc' reached 0.79090 (best 0.79090), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=94-val_acc=0.79.ckpt' as top 3

Epoch 95: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=0.834, val_acc=0.791, train_loss=0.0123, train_acc=0.999]
Epoch 95, global step 37536: 'val_acc' reached 0.79150 (best 0.79150), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=95-val_acc=0.79.ckpt' as top 3
Epoch 96: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=0.838, val_acc=0.792, train_loss=0.0114, train_acc=0.999]
Epoch 96, global step 37927: 'val_acc' reached 0.79170 (best 0.79170), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=96-val_acc=0.79.ckpt' as top 3

Epoch 97: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=0.835, val_acc=0.794, train_loss=0.0112, train_acc=0.999]
Epoch 97, global step 38318: 'val_acc' reached 0.79420 (best 0.79420), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=97-val_acc=0.79.ckpt' as top 3

Epoch 98: 100%|██████████| 391/391 [02:18<00:00,  2.83it/s, v_num=1, val_loss=0.833, val_acc=0.792, train_loss=0.0111, train_acc=0.999]
Epoch 98, global step 38709: 'val_acc' reached 0.79220 (best 0.79420), saving model to '/Users/firekind/Projects/personal/ERA4/session-8/data/checkpoints/resnet50-cifar100-epoch=98-val_acc=0.79.ckpt' as top 3

Epoch 99: 100%|██████████| 391/391 [02:17<00:00,  2.84it/s, v_num=1, val_loss=0.836, val_acc=0.791, train_loss=0.0114, train_acc=0.999]
Epoch 99, global step 39100: 'val_acc' was not in top 3

`Trainer.fit` stopped: `max_epochs=100` reached.
```
