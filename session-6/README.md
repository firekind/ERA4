# MNIST Classification - Journey to 99.4%

## The Mission
Build a CNN that hits **99.4% test accuracy** consistently in the last few epochs, using **≤15 epochs** and **<8000 parameters**. Sounds simple? Let's see how it went.

## The Journey: 4 Models, 4 Lessons

### Model 1: Getting the Skeleton Right
**Target:** Build a solid foundation with proper layer structure and stay under 8k parameters.

**What I Did:**
- Started with the Session 5 skeleton and tweaked it
- Instead of jumping straight from 8→16 channels, took a gentler path: 8→12→16
- Added dropout to prevent overfitting

**Results:**
- Parameters: 6,174 ✅
- Train Accuracy: 87.67%
- Test Accuracy: 99.20%

[Here's the notebook](./notebooks/model_1.ipynb)

**The Reality Check:** Model was underfitting badly. Training at 87% while test was at 99%? That's not right. The 20% dropout was killing the learning capacity.

**Training Logs:**

```
epoch=01 loss=0.3856 batch_id=0468 accuracy=82.89%: 100%|██████████| 469/469 [00:05<00:00, 79.85it/s]
Test set: Average loss: 0.0823, Accuracy: 9773/10000 (97.73%)

epoch=02 loss=0.2334 batch_id=0468 accuracy=86.42%: 100%|██████████| 469/469 [00:04<00:00, 94.50it/s] 
Test set: Average loss: 0.0481, Accuracy: 9858/10000 (98.58%)

epoch=03 loss=0.2942 batch_id=0468 accuracy=86.71%: 100%|██████████| 469/469 [00:04<00:00, 94.24it/s]
Test set: Average loss: 0.0771, Accuracy: 9764/10000 (97.64%)

epoch=04 loss=0.3171 batch_id=0468 accuracy=87.06%: 100%|██████████| 469/469 [00:04<00:00, 96.43it/s] 
Test set: Average loss: 0.0352, Accuracy: 9883/10000 (98.83%)

epoch=05 loss=0.3700 batch_id=0468 accuracy=86.86%: 100%|██████████| 469/469 [00:04<00:00, 94.69it/s]
Test set: Average loss: 0.0338, Accuracy: 9890/10000 (98.90%)

epoch=06 loss=0.2911 batch_id=0468 accuracy=87.37%: 100%|██████████| 469/469 [00:05<00:00, 93.10it/s]
Test set: Average loss: 0.0241, Accuracy: 9924/10000 (99.24%)

epoch=07 loss=0.3265 batch_id=0468 accuracy=87.39%: 100%|██████████| 469/469 [00:04<00:00, 95.16it/s] 
Test set: Average loss: 0.0256, Accuracy: 9919/10000 (99.19%)

epoch=08 loss=0.3459 batch_id=0468 accuracy=87.41%: 100%|██████████| 469/469 [00:04<00:00, 97.07it/s] 
Test set: Average loss: 0.0247, Accuracy: 9920/10000 (99.20%)

epoch=09 loss=0.2328 batch_id=0468 accuracy=87.54%: 100%|██████████| 469/469 [00:04<00:00, 94.66it/s]
Test set: Average loss: 0.0308, Accuracy: 9895/10000 (98.95%)

epoch=10 loss=0.3972 batch_id=0468 accuracy=87.66%: 100%|██████████| 469/469 [00:04<00:00, 95.62it/s]
Test set: Average loss: 0.0281, Accuracy: 9905/10000 (99.05%)

epoch=11 loss=0.2075 batch_id=0468 accuracy=87.80%: 100%|██████████| 469/469 [00:05<00:00, 93.18it/s]
Test set: Average loss: 0.0271, Accuracy: 9918/10000 (99.18%)

epoch=12 loss=0.2350 batch_id=0468 accuracy=87.76%: 100%|██████████| 469/469 [00:05<00:00, 93.70it/s] 
Test set: Average loss: 0.0269, Accuracy: 9916/10000 (99.16%)

epoch=13 loss=0.3078 batch_id=0468 accuracy=87.55%: 100%|██████████| 469/469 [00:04<00:00, 94.76it/s] 
Test set: Average loss: 0.0254, Accuracy: 9921/10000 (99.21%)

epoch=14 loss=0.3115 batch_id=0468 accuracy=87.88%: 100%|██████████| 469/469 [00:05<00:00, 92.61it/s]
Test set: Average loss: 0.0229, Accuracy: 9922/10000 (99.22%)

epoch=15 loss=0.2397 batch_id=0468 accuracy=87.67%: 100%|██████████| 469/469 [00:04<00:00, 98.15it/s] 
Test set: Average loss: 0.0237, Accuracy: 9920/10000 (99.20%)
```

---

### Model 2: Fixing the Underfitting
**Target:** Close that weird gap between train and test accuracy.

**What I Did:**
- Kept the same architecture
- Reduced dropout from 20% to 10%

**Results:**
- Parameters: 6,174 ✅
- Train Accuracy: 95.47%
- Test Accuracy: 99.17%

[Here's the notebook](./notebooks/model_3.ipynb)

**The Reality Check:** Much better! The gap shrunk significantly. But we're still not hitting consistent 99.4%, and the accuracy is jumping around in the last epochs. Time to stabilize things.

**Training Logs:**

```
epoch=01 loss=0.2709 batch_id=0468 accuracy=90.57%: 100%|██████████| 469/469 [00:05<00:00, 91.21it/s] 
Test set: Average loss: 0.0682, Accuracy: 9814/10000 (98.14%)

epoch=02 loss=0.1380 batch_id=0468 accuracy=94.18%: 100%|██████████| 469/469 [00:05<00:00, 91.66it/s]
Test set: Average loss: 0.0492, Accuracy: 9853/10000 (98.53%)

epoch=03 loss=0.1142 batch_id=0468 accuracy=94.63%: 100%|██████████| 469/469 [00:04<00:00, 95.07it/s]
Test set: Average loss: 0.0402, Accuracy: 9882/10000 (98.82%)

epoch=04 loss=0.1273 batch_id=0468 accuracy=94.82%: 100%|██████████| 469/469 [00:05<00:00, 92.77it/s]
Test set: Average loss: 0.0358, Accuracy: 9899/10000 (98.99%)

epoch=05 loss=0.1636 batch_id=0468 accuracy=94.83%: 100%|██████████| 469/469 [00:05<00:00, 92.83it/s]
Test set: Average loss: 0.0285, Accuracy: 9913/10000 (99.13%)

epoch=06 loss=0.0826 batch_id=0468 accuracy=95.24%: 100%|██████████| 469/469 [00:04<00:00, 94.12it/s]
Test set: Average loss: 0.0299, Accuracy: 9894/10000 (98.94%)

epoch=07 loss=0.0976 batch_id=0468 accuracy=95.25%: 100%|██████████| 469/469 [00:04<00:00, 95.14it/s]
Test set: Average loss: 0.0242, Accuracy: 9922/10000 (99.22%)

epoch=08 loss=0.0523 batch_id=0468 accuracy=95.25%: 100%|██████████| 469/469 [00:05<00:00, 92.66it/s]
Test set: Average loss: 0.0319, Accuracy: 9905/10000 (99.05%)

epoch=09 loss=0.0249 batch_id=0468 accuracy=95.16%: 100%|██████████| 469/469 [00:05<00:00, 92.52it/s]
Test set: Average loss: 0.0217, Accuracy: 9928/10000 (99.28%)

epoch=10 loss=0.2502 batch_id=0468 accuracy=95.31%: 100%|██████████| 469/469 [00:05<00:00, 91.88it/s]
Test set: Average loss: 0.0245, Accuracy: 9924/10000 (99.24%)

epoch=11 loss=0.0742 batch_id=0468 accuracy=95.43%: 100%|██████████| 469/469 [00:04<00:00, 95.46it/s]
Test set: Average loss: 0.0247, Accuracy: 9927/10000 (99.27%)

epoch=12 loss=0.1375 batch_id=0468 accuracy=95.45%: 100%|██████████| 469/469 [00:04<00:00, 93.85it/s]
Test set: Average loss: 0.0289, Accuracy: 9911/10000 (99.11%)

epoch=13 loss=0.2064 batch_id=0468 accuracy=95.45%: 100%|██████████| 469/469 [00:04<00:00, 94.78it/s]
Test set: Average loss: 0.0193, Accuracy: 9934/10000 (99.34%)

epoch=14 loss=0.1354 batch_id=0468 accuracy=95.50%: 100%|██████████| 469/469 [00:05<00:00, 92.56it/s]
Test set: Average loss: 0.0232, Accuracy: 9931/10000 (99.31%)

epoch=15 loss=0.0939 batch_id=0468 accuracy=95.47%: 100%|██████████| 469/469 [00:05<00:00, 92.76it/s]
Test set: Average loss: 0.0249, Accuracy: 9917/10000 (99.17%)
```

---

### Model 3: Bringing in the Scheduler
**Target:** Stabilize those accuracies and push towards 99.4%.

**What I Did:**
- Same architecture as Model 2
- Added StepLR scheduler (step_size=10, gamma=0.1)
- This reduces learning rate after epoch 10, helping the model settle down

**Results:**
- Parameters: 6,174 ✅
- Train Accuracy: 95.79%
- Test Accuracy: 99.39% (but epochs 11-14 all hit 99.42%!)

[Here's the notebook](./notebooks/model_3.ipynb)

**The Reality Check:** So close! The scheduler worked beautifully - epochs 11-14 were solid 99.4%+. But epoch 15 dipped to 99.39%. The model has the capacity, it just needs a tiny push.

**Training Logs:**

```
epoch=01 loss=0.2709 batch_id=0468 accuracy=90.57%: 100%|██████████| 469/469 [00:05<00:00, 86.87it/s]
Test set: Average loss: 0.0682, Accuracy: 9814/10000 (98.14%)

epoch=02 loss=0.1380 batch_id=0468 accuracy=94.18%: 100%|██████████| 469/469 [00:04<00:00, 96.93it/s] 
Test set: Average loss: 0.0492, Accuracy: 9853/10000 (98.53%)

epoch=03 loss=0.1142 batch_id=0468 accuracy=94.63%: 100%|██████████| 469/469 [00:05<00:00, 91.29it/s]
Test set: Average loss: 0.0402, Accuracy: 9882/10000 (98.82%)

epoch=04 loss=0.1273 batch_id=0468 accuracy=94.82%: 100%|██████████| 469/469 [00:04<00:00, 97.37it/s] 
Test set: Average loss: 0.0358, Accuracy: 9899/10000 (98.99%)

epoch=05 loss=0.1636 batch_id=0468 accuracy=94.83%: 100%|██████████| 469/469 [00:04<00:00, 95.04it/s] 
Test set: Average loss: 0.0285, Accuracy: 9913/10000 (99.13%)

epoch=06 loss=0.0826 batch_id=0468 accuracy=95.24%: 100%|██████████| 469/469 [00:05<00:00, 93.10it/s] 
Test set: Average loss: 0.0299, Accuracy: 9894/10000 (98.94%)

epoch=07 loss=0.0976 batch_id=0468 accuracy=95.25%: 100%|██████████| 469/469 [00:04<00:00, 95.84it/s] 
Test set: Average loss: 0.0242, Accuracy: 9922/10000 (99.22%)

epoch=08 loss=0.0523 batch_id=0468 accuracy=95.25%: 100%|██████████| 469/469 [00:04<00:00, 94.89it/s]
Test set: Average loss: 0.0319, Accuracy: 9905/10000 (99.05%)

epoch=09 loss=0.0249 batch_id=0468 accuracy=95.16%: 100%|██████████| 469/469 [00:04<00:00, 94.00it/s]
Test set: Average loss: 0.0217, Accuracy: 9928/10000 (99.28%)

epoch=10 loss=0.2502 batch_id=0468 accuracy=95.31%: 100%|██████████| 469/469 [00:04<00:00, 95.41it/s] 
Test set: Average loss: 0.0245, Accuracy: 9924/10000 (99.24%)

epoch=11 loss=0.0588 batch_id=0468 accuracy=95.62%: 100%|██████████| 469/469 [00:05<00:00, 91.83it/s]
Test set: Average loss: 0.0180, Accuracy: 9942/10000 (99.42%)

epoch=12 loss=0.1221 batch_id=0468 accuracy=95.77%: 100%|██████████| 469/469 [00:05<00:00, 92.45it/s]
Test set: Average loss: 0.0181, Accuracy: 9942/10000 (99.42%)

epoch=13 loss=0.1471 batch_id=0468 accuracy=95.76%: 100%|██████████| 469/469 [00:04<00:00, 94.23it/s] 
Test set: Average loss: 0.0182, Accuracy: 9941/10000 (99.41%)

epoch=14 loss=0.1047 batch_id=0468 accuracy=95.83%: 100%|██████████| 469/469 [00:04<00:00, 94.71it/s] 
Test set: Average loss: 0.0180, Accuracy: 9942/10000 (99.42%)

epoch=15 loss=0.0804 batch_id=0468 accuracy=95.79%: 100%|██████████| 469/469 [00:04<00:00, 94.41it/s] 
Test set: Average loss: 0.0193, Accuracy: 9939/10000 (99.39%)
```

---

### Model 4: The Final Touch
**Target:** Get consistent 99.4%+ in the last 5 epochs. Lock it in.

**What I Did:**
- Same architecture and scheduler as Model 3
- Added data augmentation:
  - Random rotation: -10° to +10°
  - Random translation: 10%

**Results:**
- Parameters: 6,174 ✅
- Train Accuracy: 95.01%
- Test Accuracy: 99.47%
- **Last 5 epochs: 99.42%, 99.48%, 99.47%, 99.44%, 99.47%** 🎯

[Here's the notebook](./notebooks/model_4.ipynb)

**The Reality Check:** Mission accomplished! The augmentations gave the model enough variety to generalize better. Train accuracy dropped slightly (95% vs 95.8% in Model 3), but test accuracy became rock solid above 99.4%.

**Training Logs:**

```
epoch=01 loss=0.2531 batch_id=0468 accuracy=87.85%: 100%|██████████| 469/469 [00:13<00:00, 34.40it/s]
Test set: Average loss: 0.0880, Accuracy: 9738/10000 (97.38%)

epoch=02 loss=0.1234 batch_id=0468 accuracy=92.99%: 100%|██████████| 469/469 [00:09<00:00, 50.27it/s]
Test set: Average loss: 0.0452, Accuracy: 9863/10000 (98.63%)

epoch=03 loss=0.1608 batch_id=0468 accuracy=93.58%: 100%|██████████| 469/469 [00:09<00:00, 49.99it/s]
Test set: Average loss: 0.0587, Accuracy: 9821/10000 (98.21%)

epoch=04 loss=0.1358 batch_id=0468 accuracy=93.99%: 100%|██████████| 469/469 [00:09<00:00, 50.56it/s]
Test set: Average loss: 0.0341, Accuracy: 9891/10000 (98.91%)

epoch=05 loss=0.2339 batch_id=0468 accuracy=94.04%: 100%|██████████| 469/469 [00:09<00:00, 51.72it/s]
Test set: Average loss: 0.0298, Accuracy: 9905/10000 (99.05%)

epoch=06 loss=0.1302 batch_id=0468 accuracy=94.38%: 100%|██████████| 469/469 [00:09<00:00, 51.75it/s]
Test set: Average loss: 0.0277, Accuracy: 9924/10000 (99.24%)

epoch=07 loss=0.2253 batch_id=0468 accuracy=94.20%: 100%|██████████| 469/469 [00:09<00:00, 50.10it/s]
Test set: Average loss: 0.0259, Accuracy: 9920/10000 (99.20%)

epoch=08 loss=0.1130 batch_id=0468 accuracy=94.44%: 100%|██████████| 469/469 [00:09<00:00, 50.62it/s]
Test set: Average loss: 0.0397, Accuracy: 9877/10000 (98.77%)

epoch=09 loss=0.1194 batch_id=0468 accuracy=94.41%: 100%|██████████| 469/469 [00:09<00:00, 51.47it/s]
Test set: Average loss: 0.0263, Accuracy: 9923/10000 (99.23%)

epoch=10 loss=0.1101 batch_id=0468 accuracy=94.56%: 100%|██████████| 469/469 [00:09<00:00, 51.59it/s]
Test set: Average loss: 0.0292, Accuracy: 9906/10000 (99.06%)

epoch=11 loss=0.1033 batch_id=0468 accuracy=94.85%: 100%|██████████| 469/469 [00:09<00:00, 52.04it/s]
Test set: Average loss: 0.0194, Accuracy: 9942/10000 (99.42%)

epoch=12 loss=0.1362 batch_id=0468 accuracy=95.03%: 100%|██████████| 469/469 [00:09<00:00, 51.87it/s]
Test set: Average loss: 0.0186, Accuracy: 9948/10000 (99.48%)

epoch=13 loss=0.1360 batch_id=0468 accuracy=95.09%: 100%|██████████| 469/469 [00:09<00:00, 52.00it/s]
Test set: Average loss: 0.0182, Accuracy: 9947/10000 (99.47%)

epoch=14 loss=0.0578 batch_id=0468 accuracy=94.87%: 100%|██████████| 469/469 [00:08<00:00, 52.12it/s]
Test set: Average loss: 0.0182, Accuracy: 9944/10000 (99.44%)

epoch=15 loss=0.1090 batch_id=0468 accuracy=95.01%: 100%|██████████| 469/469 [00:09<00:00, 52.04it/s]
Test set: Average loss: 0.0185, Accuracy: 9947/10000 (99.47%)
```

---

## Key Takeaways

1. **Architecture matters, but it's not everything** - Got the structure right in Model 1, everything else was tuning
2. **Dropout is a double-edged sword** - Too much = underfitting, too little = overfitting
3. **LR schedulers are your friend** - That step down in learning rate stabilized everything
4. **Augmentation = generalization** - Small rotations and translations made all the difference

## Final Stats
- ✅ Consistent 99.4%+ accuracy in last 5 epochs
- ✅ Only 15 epochs needed
- ✅ Just 6,174 parameters (way under 8k limit)
- ✅ Clean, modular code structure
