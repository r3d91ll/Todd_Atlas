# Atlas Technical Specifications
**Extracted from arXiv:2505.23735v1**

## 1. Memory Update Equations

### Core Atlas Update Rule (Omega Rule with Muon)

```
ℳₜ = αₜℳₜ₋₁ - ηₜ NS-5(𝒮ₜ)
𝒮ₜ = θₜ𝒮ₜ₋₁ - ∇ℓ(ℳₜ₋₁; 𝐤ₜ, 𝐯ₜ)
```

### Loss Objective

```
min_ℳ Σᵢ₌ₜ₋ₒ₊₁ᵗ γᵢ⁽ᵗ⁾ ||ℳ(ϕ(𝐤ᵢ)) - 𝐯ᵢ||²₂
```

### Variable Definitions

- **ℳₜ**: Memory state at time step t (learnable memory matrix/MLP)
- **αₜ**: Dynamic decay coefficient (input-dependent, not fixed)
- **ηₜ**: Learning rate (adaptive schedule)
- **NS-5**: Normalized Symmetric rank-5 approximation (from Muon optimizer)
- **𝒮ₜ**: Momentum accumulator (stores gradient history)
- **θₜ**: Momentum decay coefficient
- **∇ℓ**: Gradient of loss with respect to memory parameters
- **𝐤ₜ, 𝐯ₜ**: Key-value pairs at time t
- **γᵢ⁽ᵗ⁾ ∈ [0,1]**: Input-dependent context gates (learned pruning weights)
- **ϕ(·)**: Polynomial feature mapping of degree p
- **c**: Context window length (number of past tokens to memorize)
- **i**: Time index within window [t-c+1, t]

### Polynomial Feature Mapping

```
ϕₚ(x) = [xᵝ]_{|β|≤p}
```

Where β represents multi-index notation for polynomial terms up to degree p.

**Capacity Implications**:
- Linear memory + Hebbian: O(dₖ) capacity
- With polynomial features: **O(dₖᵖ) capacity**
- Deep MLP memory: O(dₖdᵥΣᵢ₌₁^𝓛_ℳ min{dₕ⁽ʲ⁾}ⱼ≥ᵢ dₕ⁽ʲ⁺¹⁾)

## 2. What Atlas Adds vs. Titans/Miras

### Compared to Titans

**Titans**: First-order gradient descent for memory updates
**Atlas**: Approximated second-order optimization via Muon optimizer

**Key Innovation**: "Locally optimal" memory management through NS-5 transformation, which approximates Hessian information without explicit computation. This prevents convergence to spurious local minima in the memory optimization landscape.

### Compared to Miras Framework

**Miras**: Optimizes single (𝐤ₜ, 𝐯ₜ) pairs (online learning)
**Atlas**: Optimizes over entire context windows

**Key Innovation**: "Test-time memorization of context" rather than individual tokens. The loss function sums over window [t-c+1, t], enabling the model to learn contextual relationships rather than isolated associations.

### Unique Atlas Features

From Table 1 ablation comparison:
1. **Dynamic decay** (αₜ is input-dependent)
2. **Deep neural memory** (MLP with ≥1 layers + residuals)
3. **Non-linear capacity** (polynomial feature expansion)
4. **Locally optimal** (second-order approximation via Muon)
5. **Flexible context** (learned window pruning via γᵢ gates)

### Architectural Distinction

**Sliding Window Attention (SWA)**: Dense attention masks over fixed windows
**Atlas**: Learned sparse context selection through gradient-based optimization of γᵢ⁽ᵗ⁾ gates, enabling "in-context token pruning without increasing parameters proportionally"

## 3. Memory Initialization

### Deep Memory Networks

Atlas uses **standard deep initialization** for MLP-based memory:
- MLPs with ≥1 hidden layers
- Residual connections for gradient flow
- Standard Xavier/He initialization for weight matrices

### Polynomial Feature Coefficients

For polynomial feature mapping ϕₚ(x):

```
aᵢ = 1/i!
```

**Rationale**: Approximates Taylor expansion of exponential kernels, providing a theoretically-grounded initialization for polynomial features.

### Momentum Accumulator

```
𝒮₀ = 0  (zero initialization)
```

Standard practice for momentum-based optimizers.

### Context Gates

γᵢ⁽ᵗ⁾ gates are learned parameters, likely initialized near 1.0 to preserve full context initially, then learned through backpropagation.

## 4. Learning Rate Handling

### Adaptive Learning Rate Schedule

**ηₜ**: Adaptive (time-varying) learning rate

The paper mentions "adaptive learning rate schedule" but does not specify exact schedule (likely cosine annealing or similar, common in large-scale training).

### Muon Optimizer Integration

The NS-5 transformation acts as a **preconditioner** on gradients:

```
ηₜ NS-5(𝒮ₜ)
```

This combines:
- Global learning rate ηₜ
- Local curvature information from NS-5
- Momentum accumulation from 𝒮ₜ

### Second-Order Approximation

Muon's NS-5 provides "Hessian-free second-order updates" - the effective learning rate is modulated by approximated curvature without explicit Hessian computation.

**Benefit**: More stable convergence in non-convex memory optimization landscape.

## 5. Training Tricks & Stability

### Numerical Stability

1. **Momentum accumulation** smooths gradient updates, preventing oscillations
2. **Polynomial feature expansion** maintains bounded intermediate values through factorial initialization (aᵢ = 1/i!)
3. **Decay coefficients** (αₜ) keep memory values normalized over time
4. **NS-5 rank-5 approximation** limits dimensionality of second-order information, avoiding full Hessian costs

### Parallelization Strategy

**Section 3.3 key insight**: Unlike online updates (c=1), context windows enable **batch gradient computation**:

```
"Fast training without substantial overhead compared to the online version"
```

**Implementation**: Compute gradients for multiple tokens in window [t-c+1, t] simultaneously, then aggregate before memory update.

**Trade-off**: Enables parallelism while avoiding quadratic attention costs of full Transformers.

### Context Window Selection

```
c ∈ ℕ≥₁
```

- **c=1**: Reduces to online Delta rule (sequential, no context)
- **c=context_length**: Global optimization (memory-intensive)
- **Intermediate c**: Balances expressivity and computational efficiency

**Design choice**: Flexible c allows tuning memory-compute trade-off per deployment scenario.

### Input-Dependent Context Pruning

**γᵢ⁽ᵗ⁾ gates**: Learned sparse attention weights within context window

**Benefit**: "In-context token pruning without increasing parameters proportionally" - model learns which historical tokens are relevant for current update, avoiding dense computation.

## 6. Key Architectural Insights

### 1. Locally Optimal Memory Management

**Problem**: First-order gradient descent can converge to spurious local minima in memory optimization.

**Atlas Solution**: Approximated second-order information (Muon/NS-5) provides curvature awareness, enabling "locally optimal" updates that better navigate the non-convex loss surface.

**Impact**: More stable and effective memory consolidation during both training and test-time adaptation.

### 2. Context vs. Token Memorization

**Previous approaches (Titans/Miras)**: Optimize individual (𝐤ₜ, 𝐯ₜ) pairs sequentially.

**Atlas**: Optimizes over context windows Σᵢ₌ₜ₋ₒ₊₁ᵗ, enabling **relational memory** rather than isolated associations.

**Insight**: "Test-time memorization of context" allows the model to learn dependencies between tokens within a window, improving coherence and factual accuracy.

### 3. Capacity Through Non-Linearity

**Linear memory**: O(dₖ) capacity (limited by key dimensionality)

**Atlas with polynomial features**: **O(dₖᵖ) capacity**

**Insight**: Non-linear feature expansion (polynomial ϕₚ) exponentially increases memory capacity without proportional parameter growth. This is a **fundamentally different scaling law** than attention-based architectures.

### 4. Fixed-Size State vs. Growing KV Cache

**Transformer**: KV cache grows linearly with sequence length (O(n·d))

**Atlas**: Fixed-size memory state (O(d²) for matrix, O(Σdₕ) for MLP)

**Insight**: Atlas maintains **constant memory footprint** regardless of sequence length, making it suitable for long-context scenarios where Transformer KV caches become prohibitive.

### 5. Learned Sparse Context

**Dense attention**: O(n²) computation over all pairs

**Atlas**: O(c·d²) computation with learned sparsity via γᵢ gates

**Insight**: Rather than hand-crafting attention patterns (sliding windows, strided patterns), Atlas **learns** which context tokens matter through gradient-based optimization. This combines flexibility of full attention with efficiency of sparse patterns.

### 6. Gradient-Based Test-Time Adaptation

**Standard inference**: Fixed weights, no adaptation

**Atlas**: Memory parameters ℳ continue optimizing during inference via gradient descent on context windows

**Insight**: "Test-time memorization" enables the model to **adapt to distribution shifts** and **incorporate new information** without retraining. This is a form of meta-learning baked into the architecture.

### 7. Omega Rule as Principled Framework

**Ad-hoc memory updates**: Various heuristics (EMA, Hebbian, etc.)

**Atlas Omega Rule**: Derived from explicit loss minimization with momentum

**Insight**: Provides theoretical grounding for memory update mechanisms. The Omega rule isn't a heuristic - it's the **gradient descent solution** to the context memorization objective with momentum.

## Implementation Considerations

### Inference Requirements

1. **Cache past keys**: Store 𝐤ᵢ for i ∈ [t-c+1, t]
2. **Fixed memory state**: ℳₜ (constant size)
3. **Momentum buffer**: 𝒮ₜ (same size as ℳ)
4. **Context gates**: γᵢ⁽ᵗ⁾ values (c scalars per position)

**Memory footprint**: O(c·dₖ + |ℳ| + |𝒮|), where |ℳ| is memory parameter count.

### Training Requirements

1. **Batch gradient computation**: Compute ∇ℓ for all i ∈ [t-c+1, t] in parallel
2. **Momentum accumulation**: Update 𝒮ₜ ← θₜ𝒮ₜ₋₁ - Σ∇ℓᵢ
3. **NS-5 transformation**: Apply Muon's rank-5 approximation to 𝒮ₜ
4. **Memory update**: ℳₜ ← αₜℳₜ₋₁ - ηₜ NS-5(𝒮ₜ)

**Computational cost**: O(c·d²) per update (c gradient computations, each O(d²) for memory parameters).

### Hyperparameters to Tune

1. **c**: Context window length (impacts memory-compute trade-off)
2. **p**: Polynomial feature degree (impacts capacity vs. dimensionality)
3. **ηₜ schedule**: Learning rate decay (impacts adaptation speed)
4. **θₜ**: Momentum coefficient (impacts stability)
5. **αₜ function**: Decay schedule (impacts memory retention)
6. **γᵢ initialization**: Context gate starting values

## Comparison Summary

| Feature | Titans | Miras | Atlas |
|---------|--------|-------|-------|
| Optimization order | 1st (gradient) | 1st (gradient) | 2nd (Muon/NS-5) |
| Memorization unit | Single token | Single token | Context window |
| Decay | Fixed/dynamic | Fixed/dynamic | Dynamic (αₜ) |
| Capacity | Linear/polynomial | Linear/polynomial | Polynomial (O(dₖᵖ)) |
| Memory structure | Linear/MLP | Linear/MLP | Deep MLP + residuals |
| Context handling | Sequential | Sequential | Window-based with gates |
| Test-time adaptation | Yes | Yes | Yes (context-aware) |
| Parallelizable | Limited | Limited | Yes (batch gradients) |

## Key Takeaways for Implementation

1. **Start with deep MLP memory** (≥1 hidden layers + residuals)
2. **Implement Muon optimizer** or similar second-order approximation
3. **Use polynomial features** with aᵢ = 1/i! initialization
4. **Batch gradient computation** across context window for efficiency
5. **Learn context gates** (γᵢ) through backpropagation
6. **Dynamic decay** (αₜ) should be input-dependent
7. **Fixed memory footprint** enables long-context deployment
8. **Theoretical capacity scales as O(dₖᵖ)** - leverage non-linearity

## Open Questions for Implementation

1. **NS-5 details**: Exact algorithm for Normalized Symmetric rank-5 approximation?
2. **γᵢ architecture**: How are context gates computed from inputs?
3. **αₜ function**: Exact form of dynamic decay (attention-based? learned MLP?)?
4. **Training stability**: Gradient clipping? Warmup schedule?
5. **Polynomial degree**: Typical p values (2? 3? higher?)?
6. **Context window**: Optimal c for different sequence lengths?

---

**Generated**: 2025-12-08
**Source**: https://arxiv.org/html/2505.23735v1
**Model**: Atlas Memory Architecture
