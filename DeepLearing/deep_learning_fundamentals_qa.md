# Deep Learning Core Foundations + Optimization - Interview Q&A

## PART 1: DEEP LEARNING CORE FOUNDATIONS

### 1. What is a Neural Network and how does it work?

A neural network is a computational model inspired by biological neurons, consisting of interconnected nodes (neurons) organized in layers. It learns to map inputs to outputs through adjustable weights and biases.

**Structure:**
- **Input Layer**: Receives raw features
- **Hidden Layers**: Learn hierarchical representations
- **Output Layer**: Produces predictions

**Forward Pass Process:**
```
output = activation(weight * input + bias)
```

Each neuron computes a weighted sum of inputs, adds a bias, and applies a non-linear activation function to enable learning of complex patterns.

---

### 2. Explain the concept of Backpropagation. Why is it important?

Backpropagation is the algorithm used to train neural networks by computing gradients of the loss function with respect to each weight in the network.

**How it works:**
1. Forward pass: Compute predictions
2. Calculate loss: Measure prediction error
3. Backward pass: Propagate error gradients from output to input layers using the chain rule
4. Update weights: Adjust weights in the direction that reduces loss

**Mathematical insight:**
```
∂Loss/∂w = ∂Loss/∂output * ∂output/∂hidden * ∂hidden/∂w
```

**Why it's important:**
- Enables efficient gradient computation (O(n) instead of O(n²))
- Makes training deep networks practical
- Allows learning of complex patterns through chain rule of calculus

---

### 3. What are Activation Functions? Name common ones and explain their differences.

Activation functions introduce non-linearity into neural networks, enabling them to learn complex decision boundaries. Without them, stacking layers would just be equivalent to a single linear transformation.

**Common Activation Functions:**

**ReLU (Rectified Linear Unit)**
```
f(x) = max(0, x)
```
- **Pros**: Computationally efficient, helps with vanishing gradient problem
- **Cons**: Dying ReLU problem (dead neurons with zero gradient)

**Sigmoid**
```
f(x) = 1 / (1 + e^(-x))
```
- **Range**: [0, 1]
- **Use**: Binary classification output layer
- **Cons**: Vanishing gradient problem, not zero-centered

**Tanh (Hyperbolic Tangent)**
```
f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- **Range**: [-1, 1]
- **Pros**: Zero-centered, stronger gradients than sigmoid
- **Cons**: Still suffers from vanishing gradients

**Leaky ReLU**
```
f(x) = x if x > 0, else α*x (where α is small, like 0.01)
```
- **Advantage**: Prevents dying ReLU problem

**ELU (Exponential Linear Unit)**
- **Pros**: Mean output closer to zero, smooth gradient
- **Cons**: Computationally more expensive

**Choice depends on:**
- Hidden layers: ReLU/Leaky ReLU (faster, modern standard)
- Output layer: Sigmoid (binary), Softmax (multiclass), Linear (regression)

---

### 4. What is the Vanishing Gradient Problem? How do you solve it?

The vanishing gradient problem occurs when gradients become exponentially small as they backpropagate through many layers, making early layers learn very slowly.

**Why it happens:**
- When backpropagating through sigmoid/tanh, gradients are multiplied by derivatives (max 0.25 for sigmoid)
- After many layers: 0.25^n → 0 as n increases
- Weights in early layers barely update

**Solutions:**

1. **Use ReLU activations**: Derivative is 1 for positive values, preventing exponential decay

2. **Batch Normalization**: Normalizes layer inputs, controlling activation values and gradient flow

3. **Skip Connections (ResNets)**: 
```
y = f(x) + x  (instead of y = f(x))
```
Allows gradients to flow directly through the identity connection

4. **Careful initialization**: Use Xavier or He initialization to keep weights in appropriate range

5. **Gradient clipping**: Cap gradient values to prevent explosion

6. **Layer normalization**: Similar to batch norm but applied to individual samples

---

### 5. What is the difference between Batch Normalization and Layer Normalization?

Both normalize activations but differ in the normalization dimension:

**Batch Normalization (BatchNorm)**
```
x_normalized = (x - mean_across_batch) / sqrt(variance_across_batch + ε)
```
- Normalizes **across the batch dimension** for each feature
- **Shape**: For batch size B and features D, mean/var computed over B samples
- **Pros**: Reduces internal covariate shift, stabilizes training, allows higher learning rates
- **Cons**: Behaves differently during training vs inference; requires reasonable batch size; problematic for RNNs

**Layer Normalization (LayerNorm)**
```
x_normalized = (x - mean_across_features) / sqrt(variance_across_features + ε)
```
- Normalizes **across the features dimension** for each sample independently
- **Shape**: For batch size B and features D, mean/var computed over D features per sample
- **Pros**: Batch-size independent, consistent train/inference, works well with RNNs and Transformers
- **Cons**: Slightly slower, less effective for some CNN architectures

**When to use:**
- CNNs: BatchNorm (slightly better empirically)
- Transformers/RNNs: LayerNorm (more stable)
- Small batch sizes: LayerNorm

---

### 6. Explain Forward Pass vs. Backward Pass in detail.

**Forward Pass:**
1. Input flows through network layer by layer
2. Each layer computes: `z = w*x + b` and `a = activation(z)`
3. Continue until output layer produces prediction
4. Compute loss: `Loss = f(prediction, actual_label)`

**Purpose**: Get prediction and loss value

**Backward Pass (Backpropagation):**
1. Start from output layer with `dL/dOutput`
2. For each layer going backward:
   - Compute gradient w.r.t weights: `dL/dW = dL/dActivation * dActivation/dZ * dZ/dW`
   - Compute gradient w.r.t biases: `dL/dB = dL/dActivation * dActivation/dZ`
   - Propagate to previous layer: `dL/dInput = dL/dActivation * dActivation/dZ * dZ/dInput`
3. Chain rule applied throughout

**Purpose**: Compute gradients for weight updates

**Computational Complexity:**
- Forward pass: O(n) where n is number of operations
- Backward pass: O(n) where n is similar to forward pass (not O(n²))
- **Total training cost**: ~2x forward pass cost (forward + backward)

---

### 7. What is Weight Initialization and why does it matter?

Weight initialization sets the starting values of network weights before training. Poor initialization can severely impact training.

**Why it matters:**
- **All zeros**: All neurons learn identically (symmetry problem)
- **Too large**: Can cause gradient explosion or saturation
- **Too small**: Can cause vanishing gradients
- **Good initialization**: Helps convergence speed and final model quality

**Common strategies:**

**Random Normal**
```
w ~ N(0, σ²)
```
- Issue: Need careful σ selection

**Xavier/Glorot Initialization**
```
w ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]
```
- **Best for**: Sigmoid/Tanh activations
- **Goal**: Keep variance of activations constant across layers

**He Initialization**
```
w ~ N(0, 2/n_in)
```
- **Best for**: ReLU and variants
- **Motivation**: ReLU kills half of activations, so need 2x variance

**When to use:**
- ReLU networks: He initialization
- Sigmoid/Tanh networks: Xavier initialization
- Deep networks: Layer normalization reduces sensitivity to initialization

---

### 8. What is Dropout and how does it prevent overfitting?

Dropout is a regularization technique that randomly sets a fraction of activations to zero during training.

**How it works:**
```
During training: a = activation(z) * mask  (mask ~ Bernoulli(1-p))
During inference: a = activation(z)  (no dropout)
```
Where p is the dropout rate (typically 0.5).

**Why it prevents overfitting:**
1. Forces network to learn redundant representations
2. Prevents co-adaptation of neurons (neurons can't rely on specific neighbors)
3. Acts like training exponentially many different networks
4. Ensemble effect: At test time, use all neurons (scaled by 1-p)

**Key points:**
- Only applied during training, not inference
- Common dropout rates: 0.2-0.5
- Can be applied to any layer, but typically after dense/convolutional layers
- Modern alternative: Batch normalization often reduces need for dropout

**Inverted Dropout variant:**
```
During training: a = activation(z) * mask / (1-p)
During inference: a = activation(z)
```
More common in modern frameworks to avoid inference scaling.

---

## PART 2: OPTIMIZATION

### 9. Explain Gradient Descent and its variants (SGD, Mini-batch GD, Batch GD).

Gradient Descent is the fundamental optimization algorithm for training neural networks. It updates weights in the direction of negative gradient to minimize loss.

**Update rule:**
```
w_new = w_old - learning_rate * ∂Loss/∂w
```

**Three main variants:**

**1. Batch Gradient Descent**
- Uses **entire dataset** to compute gradient
- Update: `w = w - lr * gradient(entire_dataset)`
- **Pros**: Stable gradient, good for convex problems
- **Cons**: Very slow for large datasets, requires high memory

**2. Stochastic Gradient Descent (SGD)**
- Uses **single sample** per update
- Update: `w = w - lr * gradient(single_sample)`
- **Pros**: Fast, can escape local minima due to noise
- **Cons**: Very noisy, less stable, harder to parallelize

**3. Mini-batch Gradient Descent** (most common)
- Uses **small batch** of samples (32-256 typically)
- Update: `w = w - lr * gradient(batch)`
- **Pros**: Balances speed and stability, parallelizable, good memory efficiency
- **Cons**: Hyperparameter (batch size) to tune

**Comparison:**
| Aspect | Batch | SGD | Mini-batch |
|--------|-------|-----|-----------|
| Convergence Speed | Slow | Fast | Medium |
| Stability | High | Low | Medium |
| Memory | High | Low | Low-Medium |
| Parallelization | Good | Poor | Good |
| Generalization | Often worse | Often better | Good |

---

### 10. What is Momentum and why does it help optimization?

Momentum is an extension of SGD that accumulates gradients over time, allowing the optimizer to build up speed in consistent directions.

**Update rule:**
```
velocity = β * velocity + (1 - β) * gradient
weight = weight - learning_rate * velocity
```
Where β is typically 0.9 or 0.99.

**Intuition:**
Think of a ball rolling downhill. It builds up speed (momentum) as it goes, allowing it to:
1. **Speed up on consistent slopes**: Keep rolling faster in same direction
2. **Dampen oscillations**: Smooth out noisy gradient updates
3. **Escape shallow local minima**: Build up enough speed to roll over them

**Mathematical benefit:**
- **Exponential moving average** of gradients: Emphasizes recent gradients more
- **Effectively increases step size** in consistent directions
- **Reduces variance** from noisy gradient estimates

**Variants:**
- **Standard Momentum**: `v = β*v + g`
- **Nesterov Momentum**: `v = β*v + g` but compute gradient at lookahead position
  - Look ahead to where momentum would take us
  - Compute gradient there to be more responsive

**Effect:**
- Convergence is much faster, especially in ravines (narrow valleys)
- Better generalization
- Can handle larger learning rates

---

### 11. Explain Adam Optimizer. Why is it widely used?

Adam (Adaptive Moment Estimation) is one of the most popular optimizers in deep learning. It combines ideas from momentum and RMSprop.

**Algorithm:**
```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t           (1st moment: momentum)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²         (2nd moment: adaptive learning rate)
m_hat_t = m_t / (1 - β₁^t)                    (bias correction)
v_hat_t = v_t / (1 - β₂^t)                    (bias correction)
w = w - lr * m_hat_t / (sqrt(v_hat_t) + ε)
```

**Components:**
1. **First moment (m_t)**: Running average of gradients (like momentum)
2. **Second moment (v_t)**: Running average of squared gradients
3. **Bias correction**: Correct for initialization bias (important early in training)
4. **Adaptive learning rate**: Scale by 1/sqrt(v), larger for sparse gradients

**Default hyperparameters (usually work well):**
- β₁ = 0.9 (momentum decay)
- β₂ = 0.999 (second moment decay)
- ε = 1e-8 (small constant for numerical stability)
- lr = 0.001 (learning rate)

**Why widely used:**

1. **Adaptive learning rates**: Different learning rates per parameter based on gradient history
   - Sparse gradients get larger steps
   - Frequent parameters get smaller steps

2. **Works with momentum**: Combines acceleration and adaptive rates

3. **Robust**: Works well across diverse problems without much tuning

4. **Fast convergence**: Usually converges faster than SGD with momentum

5. **Handles sparse data**: Better for NLP, recommendation systems

**Comparison to SGD + Momentum:**
- Adam: Faster convergence, less hyperparameter tuning
- SGD + Momentum: Often better generalization, especially with proper learning rate decay
- Modern trend: Start with Adam, fine-tune with SGD if needed

**Disadvantages:**
- More memory (stores m and v for each parameter)
- Can sometimes generalize worse than SGD (though usually not significant)
- More hyperparameters to tune if you want optimal performance

---

### 12. What is Learning Rate and how do you choose it?

Learning rate (lr) is the step size for weight updates. It's one of the most critical hyperparameters.

**Update rule:**
```
w = w - learning_rate * gradient
```

**Effect of learning rate:**

**Too small:**
- Very slow convergence
- Takes forever to train
- May get stuck in local minima

**Too large:**
- Overshoots optima
- Divergence (loss increases)
- Oscillations, unstable training
- May never converge

**Just right:**
- Smooth, steady decrease in loss
- Good convergence speed
- Stable training

**How to choose:**

**1. Learning Rate Range Test**
```
Start with lr = 1e-4, train for few iterations
Gradually increase lr (10x every few iterations)
Plot loss vs learning rate
Choose lr where loss is still decreasing but not yet exploding
Typically 1-2 orders of magnitude below divergence point
```

**2. Learning Rate Scheduling**
Instead of fixed learning rate, decrease it during training:

**Step Decay:**
```
lr = initial_lr * decay^(epoch / decay_steps)
```
Reduces lr by constant factor periodically

**Cosine Annealing:**
```
lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```
Smoothly decreases lr following cosine curve

**Exponential Decay:**
```
lr = initial_lr * e^(-decay_rate * epoch)
```

**3. Typical ranges by optimizer:**
- **SGD**: 0.01 - 0.1
- **Adam**: 0.0001 - 0.001
- **RMSprop**: 0.0001 - 0.01

**4. Practical tips:**
- Start with optimizer's recommended default
- Use learning rate scheduling for better convergence
- Higher learning rates for larger batch sizes
- Reduce learning rate when loss plateaus
- Different lr for different layers (transfer learning): smaller for pretrained, larger for new

---

### 13. What is Regularization? Explain L1, L2, and Elastic Net.

Regularization prevents overfitting by adding a penalty term for model complexity to the loss function.

**Core idea:**
```
Total Loss = Data Loss + λ * Regularization Term
```
Where λ controls regularization strength.

**L2 Regularization (Ridge)**
```
Loss = MSE + λ * Σ(w²)
```
- **Penalty**: Sum of squared weights
- **Effect**: Encourages weights to be small
- **Weight update**: `w = w * (1 - λ) - lr * gradient`
  - Multiplicative decay, weights shrink toward zero but never reach it
- **Pros**: Smooth penalty, numerically stable
- **Cons**: Doesn't produce sparsity (all weights nonzero)
- **Use**: Most common, general purpose

**L1 Regularization (Lasso)**
```
Loss = MSE + λ * Σ(|w|)
```
- **Penalty**: Sum of absolute values
- **Effect**: Encourages sparsity (some weights become exactly zero)
- **Pros**: Feature selection (zeros out unimportant features)
- **Cons**: Non-differentiable at zero, can be numerically unstable
- **Use**: When feature selection is important, interpretability matters

**Elastic Net**
```
Loss = MSE + λ₁ * Σ(|w|) + λ₂ * Σ(w²)
```
- Combination of L1 and L2
- **Pros**: Gets benefits of both (sparsity + stability)
- **Cons**: Two hyperparameters to tune

**Comparison:**
| Property | L1 | L2 | Elastic Net |
|----------|-----|-----|-----------|
| Sparsity | Yes | No | Yes |
| Stability | Lower | Higher | High |
| Feature selection | Good | Fair | Good |
| Computational cost | Higher | Lower | Medium |

**Other regularization techniques:**
- **Dropout**: Random neuron deactivation
- **Early stopping**: Stop training when validation loss stops improving
- **Data augmentation**: Artificially increase dataset size
- **Batch normalization**: Also has regularization effect

---

### 14. What is Weight Decay and how does it differ from L2 Regularization?

Weight decay and L2 regularization are often used interchangeably, but they're technically different in how they're implemented in modern optimizers.

**Theoretical L2 Regularization:**
```
Loss = MSE + λ * Σ(w²)
∂Loss/∂w = ∂MSE/∂w + 2λ * w
w = w - lr * (∂MSE/∂w + 2λ * w)
```

**Weight Decay in optimizers (e.g., Adam):**
```
w = w - lr * (gradient + weight_decay * w)
```

**The difference:**
In adaptive learning rate optimizers like Adam:
- **L2 regularization**: Penalty is added to gradient before adaptation
- **Weight decay**: Penalty is added after adaptation (decoupled)

```
Adam with L2:         
m = β₁*m + (1-β₁)*(g + 2λw)      [penalty scaled by learning rate]
v = β₂*v + (1-β₂)*(g + 2λw)²

Adam with Weight Decay:
m = β₁*m + (1-β₁)*g              [no penalty here]
v = β₂*v + (1-β₂)*g²
w = w - lr * (m / √v + wd * w)   [penalty independent of learning rate]
```

**Why it matters:**
- Weight decay is **more effective** in practice with adaptive optimizers
- **Decoupling** means regularization strength is independent of learning rate
- Modern libraries (PyTorch, TensorFlow) use "weight_decay" parameter
- For SGD, they're nearly equivalent

**Practical implications:**
- Use `weight_decay` parameter in optimizer, not add L2 to loss
- Typical weight decay: 1e-4 to 1e-5
- Language models (transformers): Often use weight decay with Adam
- CNNs: Often use weight decay with SGD

---

### 15. Explain the concept of Learning Rate Warmup. When and why is it used?

Learning rate warmup gradually increases the learning rate from 0 to target value at the start of training.

**Common warmup schedule:**
```
for step in [0, warmup_steps]:
    lr_t = (step / warmup_steps) * target_lr

for step in [warmup_steps, total_steps]:
    lr_t = target_lr * (schedule_function)
```

**Why warmup helps:**

1. **Avoid bad initialization effects**:
   - At initialization, gradients can be unreliable
   - Large learning rate could cause divergence
   - Warmup allows settling to good gradient estimates

2. **Stabilize gradient estimates** (especially with batch norm):
   - Early batch statistics are noisy
   - Gradients are unreliable
   - Gradually increasing lr prevents instability

3. **Better convergence**:
   - Smoother loss curve
   - Higher final performance
   - More stable optimization

4. **Especially important for**:
   - **Transformers**: Standard in attention-based models
   - **Large batch training**: Noisy gradients early on
   - **Deep networks**: Gradient propagation needs time to stabilize
   - **Transfer learning**: When fine-tuning with high initial lr

**Typical warmup configurations:**

**Transformers (standard)**:
```
Warmup for ~10% of total steps
Then decay learning rate (cosine, polynomial, etc.)
```

**Example**: 100K total steps
- Steps 0-10K: Warmup to target lr
- Steps 10K-100K: Decay from target to small value

**When NOT needed**:
- Small networks on simple tasks
- SGD without momentum (less sensitive)
- When using lower learning rates anyway
- Fine-tuning with learning rate scheduling

**Interaction with other scheduling**:
```
Combined schedule = warmup + decay
lr_t = min(
    (t / warmup_steps) * target_lr,           [warmup part]
    target_lr * cosine_decay(t - warmup_steps) [decay part]
)
```

---

### 16. What is the difference between Batch Size and Learning Rate? How do they interact?

**Batch Size:**
- Number of samples used for one gradient computation
- Typical range: 32-512 for smaller models, up to 32K for large language models

**Learning Rate:**
- Magnitude of weight update per gradient step
- Typical range: 1e-4 to 0.1 depending on optimizer

**How they interact:**

**The Scaling Rule:**
```
When batch_size increases by factor k:
Learning rate should increase by ~√k
```

**Why:**
- Larger batch size → gradients are more stable (lower variance)
- Stable gradients can afford larger steps
- But don't scale linearly; only √k because:
  - Variance decreases as 1/√batch_size
  - Step size should scale with √variance

**Practical example:**
```
Original: batch_size=32, lr=0.001
New: batch_size=128 (4x increase)

New lr ≈ 0.001 * √4 = 0.001 * 2 = 0.002
```

**Detailed relationship:**

**Small batch size:**
- Noisy gradients (high variance)
- Can't use large learning rates (will diverge)
- Good for generalization (noise acts as regularization)
- Slower wall-clock time per epoch

**Large batch size:**
- Stable gradients (low variance)
- Can use larger learning rates
- Might hurt generalization (less noise regularization)
- Faster wall-clock time per epoch

**Batch size too large:**
- Loss plateaus at suboptimal solution
- Can't escape local minima
- Overfitting risk (less regularization)

**Practical strategies:**

**For better generalization:**
- Use smaller batch size if possible
- Add dropout/weight decay to compensate

**For faster training (distributed):**
- Use large batch size
- Scale learning rate carefully
- Use learning rate warmup
- Consider gradient accumulation
- May need additional regularization

**Modern trend (Large Language Models):**
- Very large batch sizes (1K-100K)
- Linear scaling rule: `lr_new = lr_old * (batch_size_new / batch_size_old)`
- Learning rate warmup and careful scheduling essential

---

## PART 3: COMMON INTERVIEW QUESTIONS

### 17. How do you debug a neural network that's not converging?

**Checklist approach:**

**1. Check Data**
- Verify data loading correctness
- Check data normalization (subtract mean, divide by std)
- Ensure labels are correct (not inverted or shifted)
- Look for NaN/Inf in data
- Verify train/test split

**2. Check Loss Function**
- Print loss value at initialization (should be ~ln(num_classes) for classification)
- Verify loss decreases initially
- Check if loss is Nan/Inf (indicates numerical instability)

**3. Learning Rate**
- Too small: Very slow convergence (like training is frozen)
- Too large: Loss diverges or oscillates wildly
- Try wide range: 10^{-5} to 10^{-1}
- Start with learning rate range test

**4. Batch Normalization**
- Can cause issues if features are unnormalized
- Try removing temporarily to debug
- Check running mean/variance in inference mode

**5. Activation Functions**
- Check for dead neurons (ReLU)
- Switch to Leaky ReLU if suspected
- Verify activation outputs aren't always near zero

**6. Weights & Biases**
- Print weight statistics (mean, std, min, max)
- Should start near zero, change during training
- Check for exploding/vanishing gradients
- Use gradient clipping if exploding

**7. Model Complexity**
- Start simple: single hidden layer
- Verify it can overfit on small batch
- Gradually add complexity
- Too complex with small data won't generalize

**8. Specific tests:**

**Overfit on single batch:**
```python
# Should achieve zero loss on single batch
# If not, implementation issue
for epoch in range(1000):
    loss = model(single_batch)
    loss.backward()
    optimizer.step()
```

**Initialize to zero loss:**
- Set weights to near-zero
- Loss should be near random guessing value
- ln(num_classes) for classification with softmax

**Gradient check:**
- Compute numerical gradients: `(f(x+h) - f(x-h)) / 2h`
- Compare with backpropagation gradients
- Should match to 1e-5 relative error

---

### 18. Explain Overfitting vs Underfitting and how to address each.

**Overfitting:**
- Model learns **training data too well**, including noise
- **Training loss**: Low ✓
- **Validation loss**: High ✗
- **Generalization**: Poor

**Underfitting:**
- Model is **too simple** to capture patterns
- **Training loss**: High
- **Validation loss**: High
- **Generalization**: Poor

**How to detect:**

```
Epoch | Train Loss | Val Loss | Status
------|-----------|----------|--------
1     | 0.50      | 0.52     | OK (both decreasing together)
10    | 0.10      | 0.15     | OK
50    | 0.02      | 0.25     | OVERFITTING (train low, val high)
100   | 0.01      | 0.35     | OVERFITTING (getting worse)

Epoch | Train Loss | Val Loss | Status
------|-----------|----------|--------
1     | 1.50      | 1.55     | Underfitting (both high)
50    | 1.45      | 1.50     | Underfitting (no improvement)
```

**Solutions for Overfitting:**

1. **More data**: Best solution if available
2. **Data augmentation**: Create variations of existing data
3. **Regularization**: L1, L2, dropout
4. **Reduce model size**: Fewer parameters
5. **Early stopping**: Stop when validation loss increases
6. **Batch normalization**: Also helps regularization
7. **Lower learning rate**: Prevent weights from growing too large
8. **Cross-validation**: Better evaluation of generalization

**Solutions for Underfitting:**

1. **Increase model capacity**: More hidden units, more layers
2. **Train longer**: More epochs (if not converged)
3. **Decrease regularization**: Less dropout, less weight decay
4. **Higher learning rate**: May help find better solutions
5. **Better features**: Feature engineering
6. **Reduce noise in data**: Data cleaning
7. **Use more complex architecture**: Better for problem

**Typical progression during training:**
```
Epoch 1-10:     Both train and val loss decrease (good)
Epoch 10-100:   Train loss decreases, val loss plateaus (good, maybe overfitting starting)
Epoch 100+:     Train continues decreasing, val increases (clear overfitting)
```

---

### 19. What are Hyperparameters vs Parameters? Give examples.

**Parameters:**
- **Learned during training**
- Weights (w) and biases (b)
- **Number depends on model architecture**
- Updated via backpropagation and optimizer

**Hyperparameters:**
- **Set before training** (not learned)
- Control **how model trains** and **model structure**
- Require manual tuning or automated search
- Examples:
  - Learning rate
  - Batch size
  - Number of epochs
  - Network architecture (layers, units)
  - Regularization (λ for L2, dropout rate)
  - Optimizer type and its parameters (momentum, β₁, β₂)
  - Activation function choices
  - Initialization scheme

**Examples table:**

| Aspect | Parameter | Hyperparameter |
|--------|-----------|-----------------|
| Example 1 | Weight matrix (10×20) | Learning rate = 0.001 |
| Example 2 | Bias vector (20,) | Number of hidden layers = 3 |
| Example 3 | Bias value = 0.5 | Dropout rate = 0.5 |
| Update | Via backprop | Manual tuning |
| Count | Thousands/millions | ~10-50 |
| Discovery | Automatic | Grid/random search |

**Hyperparameter tuning methods:**

1. **Grid Search**: Try all combinations
2. **Random Search**: Random sampling of space
3. **Bayesian Optimization**: Smart sampling based on previous results
4. **Hyperband**: Adaptive resource allocation

---

### 20. Explain the Bias-Variance Tradeoff.

The bias-variance tradeoff is fundamental to understanding model performance.

**Total Error decomposition:**
```
Total Error = Bias² + Variance + Irreducible Error

Bias²: Error from wrong assumptions (underfitting)
Variance: Sensitivity to training data changes (overfitting)
Irreducible: Noise inherent in problem
```

**High Bias:**
- Model makes **strong assumptions** about data
- **Underfits**: Cannot capture true patterns
- **Across datasets**: Consistent wrong predictions
- **Examples**: Linear model for nonlinear data
- **Solution**: More complex model

**High Variance:**
- Model is **too flexible**
- **Overfits**: Learns training data noise
- **Across datasets**: Very different predictions (unstable)
- **Examples**: Very deep network on small data
- **Solution**: Regularization, more data

**Visual intuition:**
```
Target: Bullseye center

High Bias, Low Variance:     Low Bias, High Variance:
X X X                        X  X  X
X   X                          X   
X X X                          X X
(Clustered but wrong spot)     (Scattered around target)

Low Bias, Low Variance:
  X
 X X
  X X
(Scattered around correct spot)
```

**Deep Learning Context:**

**Early epochs**: Low complexity → High bias, low variance
```
Training loss high, validation loss high
Model underfits
```

**Mid training**: Balanced
```
Both losses low and close
Good generalization
```

**Late epochs**: High complexity → Low bias, high variance
```
Training loss very low, validation loss high
Model overfits
```

**How to improve:**

**Reduce bias** (if high):
- Increase model capacity
- Train longer
- Use more complex architecture
- Remove regularization

**Reduce variance** (if high):
- Add regularization (L1, L2, dropout)
- Get more data
- Early stopping
- Reduce model complexity

**The key insight**: Often must accept some bias to reduce variance (and vice versa). The goal is **minimum total error**, not minimum bias or variance individually.

---

## INTERVIEW TIPS

1. **Always clarify assumptions**: Ask about data size, problem type, computational constraints
2. **Show your thinking**: Explain why you choose certain techniques
3. **Know the tradeoffs**: Every method has pros/cons, mention them
4. **Practical knowledge**: Know how to implement, not just theory
5. **Recent trends**: Mention modern techniques (Layer Norm, Weight Decay, Warmup)
6. **Code examples**: Be ready to sketch pseudocode
7. **Debugging mindset**: Show systematic approach to problem-solving
8. **Simplicity first**: Start simple, add complexity only if needed

---

## QUICK REFERENCE: Common Formulas

**Gradient Descent**: `w = w - lr × ∇L`

**Momentum**: `v = βv + ∇L`, `w = w - lr × v`

**Adam**: 
```
m = β₁m + (1-β₁)g
v = β₂v + (1-β₂)g²
w = w - lr × m / √(v + ε)
```

**L2 Loss**: `L = Σ(y - ŷ)²`

**Cross-Entropy**: `L = -Σ y log(ŷ)`

**ReLU**: `f(x) = max(0, x)`

**Sigmoid**: `f(x) = 1 / (1 + e^{-x})`

**Softmax**: `σ(x_i) = e^{x_i} / Σ_j e^{x_j}`

**Backpropagation Chain Rule**: `∂L/∂w = ∂L/∂y × ∂y/∂z × ∂z/∂w`
