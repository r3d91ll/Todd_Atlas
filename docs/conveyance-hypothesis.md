# The Conveyance Hypothesis
## A Mathematical Framework for Measuring Information Transfer Effectiveness

**Version:** 4.0 (December 2025)  
**Author:** Todd Bucy  
**Status:** Hypothesis under active investigation

---

## Executive Summary

The Conveyance Hypothesis proposes that **information transfer effectiveness between agents can be mathematically measured**. Unlike Shannon's information theory, which measures signal fidelity (did the bits arrive intact?), conveyance measures semantic effectiveness (did the meaning transform the receiver appropriately?).

**Core Insight:** Information is not a thing that flows—it's a transformation that occurs when data interacts with an agent's internal structure. Effective conveyance requires both parties to be modified through the interaction.

**Primary Equation:**
```
C_pair(i ↔ j) = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij
```

**Key Prediction:** Low-dimensional representations (128-256D) outperform high-dimensional ones for information transfer, and dimensional collapse (measured by β) negatively correlates with task performance.

---

## 1. Theoretical Foundation

### 1.1 Shannon's Deliberate Exclusion

In 1948, Claude Shannon explicitly excluded meaning from his theory:

> "The semantic aspects of communication are irrelevant to the engineering problem."

This wasn't naïve—Shannon understood meaning mattered, but 1948 technology couldn't measure it. His theory optimized for **signal fidelity**: how accurately can bits travel from sender to receiver?

**The Conveyance Hypothesis addresses what Shannon excluded**: the effectiveness of semantic transformation between agents.

| Aspect | Shannon Theory | Conveyance Hypothesis |
|--------|----------------|----------------------|
| Concern | Signal fidelity | Semantic effectiveness |
| Measures | Bit error rate, channel capacity | D_eff, bilateral C_pair |
| Question | Did the signal arrive intact? | Did meaning transform the receiver? |
| Agents | Sender → Channel → Receiver | Internal geometries interacting |

### 1.2 Actor-Network Theory: Translation, Not Transmission

Drawing from Bruno Latour's Actor-Network Theory, we recognize that information transfer is **translation**, not transmission:

> "There is no society, no social realm, and no social ties, but there exists translations between mediators that may generate traceable associations."

**Key insight:** Information doesn't flow like water through pipes. Every transfer is a creative transformation where **both sender and receiver are modified** through their interaction.

### 1.3 Boundary Objects

Following Susan Leigh Star's work, effective transfer between agents with different internal structures requires **boundary objects**—intermediate representations that maintain coherence across contexts.

In our framework, **C_ext** (external shared context) represents boundary objects that:
- Preserve sufficient structure for reconstruction
- Compress high-dimensional internal states into transferable form
- Expand appropriately in the receiver's distinct geometric space

### 1.4 The Data vs. Information Distinction

**Data** = Static patterns, boundary-preserving, exists without observers
**Information** = Dynamic transformation, boundary-crossing, requires agent interaction

Data becomes information only when it transforms an agent's internal state. A book on a shelf contains data; reading it creates information through the transformation of the reader's understanding.

---

## 2. The Mathematical Framework

### 2.1 Primary Conveyance Equation (v4.0)

Bilateral conveyance effectiveness between agents i and j:

```
C_pair(i ↔ j) = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij
```

**Components:**

| Symbol | Name | Meaning |
|--------|------|---------|
| C_pair | Bilateral Conveyance | Overall transfer effectiveness between two agents |
| Hmean | Harmonic Mean | Captures bilateral constraint (weakest link dominates) |
| C_out | Output Capacity | Sender's ability to encode meaning |
| C_in | Input Capacity | Receiver's ability to integrate meaning |
| f_dim(D_eff) | Dimensional Function | Richness of semantic representation |
| P_ij | Protocol Compatibility | How well agents' interfaces match [0,1] |

### 2.2 Why Harmonic Mean?

The harmonic mean is chosen deliberately:

```
Hmean(a, b) = 2ab / (a + b)
```

**Property:** The harmonic mean is dominated by the smaller value.

- If C_out = 0.9 and C_in = 0.1, then Hmean = 0.18
- Excellent sender + poor receiver = poor conveyance

This matches intuition: a brilliant lecturer teaching in a language students don't understand achieves low conveyance regardless of lecture quality.

### 2.3 Component Equations

Individual agent conveyance capacity decomposes as:

```
C_out(i → j) = (W × R × H) / T
C_in(j ← i) = (W × R × H) / T
```

**Variable Definitions:**

| Variable | Name | Concept | How to Measure |
|----------|------|---------|----------------|
| **W** | Semantic Investment | Computational allocation for semantic understanding | Hidden state activation patterns |
| **R** | Relational Discovery | Quality of geometric positioning in semantic space | Graph-theoretic embedding properties |
| **H** | Computational Frame | Processing throughput capability | Attention efficiency, layer utilization |
| **T** | Temporal Investment | Total computational budget expended | Token count, processing time |

### 2.4 Why Multiplicative?

The framework uses **multiplication** rather than addition because:

- **W = 0** (zero semantic content) → Nothing to transfer → Zero conveyance
- **R = 0** (zero relational structure) → No geometric organization → Zero conveyance
- **H = 0** (zero processing) → Cannot utilize signals → Zero conveyance
- **P_ij = 0** (zero protocol match) → Cannot communicate → Zero conveyance

**A single zero produces zero output.** This "zero-propagation" principle explains why communication failures are often catastrophic rather than gradual—one essential missing component collapses the entire transfer.

### 2.5 Dimensional Richness Function

The dimensional function scales conveyance by how much semantic structure is preserved:

```
f_dim(D_eff) = (D_eff / D_target)^α_dim

Where:
  D_eff = Effective dimensionality (measured via PCA)
  D_target = Target dimensionality for the layer/system
  α_dim ∈ [0.5, 1.0] (empirically determined scaling factor)
```

**Interpretation:**
- D_eff / D_target = 1.0 → Full dimensional preservation → f_dim = 1.0
- D_eff / D_target = 0.5 → Half dimensions preserved → f_dim ≈ 0.5-0.7
- D_eff / D_target → 0 → Dimensional collapse → f_dim → 0

---

## 3. Key Metrics

### 3.1 D_eff (Effective Dimensionality) — PRIMARY METRIC

**Definition:** The number of independent semantic dimensions preserved during processing, computed via PCA with 90% variance threshold.

```python
def compute_d_eff(embeddings, variance_threshold=0.90):
    """
    Count dimensions capturing 90% of variance.
    
    CRITICAL: L2 normalize embeddings first to prevent
    magnitude artifacts from dominating variance.
    """
    # Center and compute covariance
    centered = embeddings - embeddings.mean(axis=0)
    cov = centered.T @ centered / (len(embeddings) - 1)
    
    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(cov)[::-1]  # Descending
    
    # Cumulative variance ratio
    cumvar = np.cumsum(eigenvalues) / eigenvalues.sum()
    
    # Count dimensions below threshold
    d_eff = np.searchsorted(cumvar, variance_threshold) + 1
    
    return d_eff
```

**Why 90% threshold?**
- Established compromise between signal preservation and noise reduction
- Components beyond 90% typically capture noise/artifacts, not semantics
- Robust across diverse domains (neural activity, manifold learning, NLP)

**Target Values:**
| Nominal Dimension | Healthy D_eff | Collapse Warning |
|-------------------|---------------|------------------|
| 512D | ≥ 34 | < 20 |
| 256D | ≥ 20 | < 12 |
| 128D | ≥ 12 | < 8 |
| 64D | ≥ 8 | < 5 |
| 24D | ≥ 20 | < 12 |

### 3.2 β (Beta) — DIAGNOSTIC METRIC (NOT Optimization Target)

**Definition:** Collapse indicator measuring dimensional compression during processing.

```
β = D_eff_input / D_eff_output
```

**CRITICAL:** β is a **diagnostic warning signal**, not something to optimize. High β indicates information loss through dimensional collapse.

**Interpretation:**
| β Value | Status | Meaning |
|---------|--------|---------|
| < 2.0 | Healthy | Low collapse, good generalization expected |
| 2.0 - 2.5 | Warning | Moderate collapse, acceptable but monitor |
| 2.5 - 3.0 | Concerning | High collapse, likely overfitting |
| > 3.0 | Critical | Severe collapse, investigate immediately |

**Empirical Finding:** β shows strong negative correlation with task performance (r ≈ -0.92 in preliminary experiments). Lower β = better generalization.

### 3.3 Secondary Geometric Metrics

| Metric | Symbol | Target | Meaning |
|--------|--------|--------|---------|
| Mean k-NN Distance | d_nn | 0.10-0.15 | Moderate clustering (not too tight, not scattered) |
| Label Consistency | LC | ≥ 0.87 | Neighbors share semantic categories |
| Boundary Sharpness | σ_boundary | 0.40-0.50 | Balanced separation (not over-emphasized) |

### 3.4 Task Performance — VALIDATION METRICS

**Primary validation is always downstream task performance, not geometric metrics.**

| Metric | Target | Use Case |
|--------|--------|----------|
| F1 Score | ≥ 0.90 (strong), ≥ 0.85 (acceptable) | Classification |
| Recall@10 | ≥ 0.85 | Retrieval |
| Perplexity | Lower is better | Language modeling |

**Geometric metrics are diagnostic.** They help explain *why* task performance is good or bad, but task performance is ground truth.

---

## 4. Core Hypotheses (Under Investigation)

### 4.1 Low-Dimensional Hypothesis

**Prediction:** External shared context (C_ext) performs better at 128-256 dimensions than higher dimensions.

**Rationale:** 
- Forcing low dimensions makes geometric relationships carry semantic meaning
- High dimensions allow magnitude artifacts to dominate
- BDH architecture independently arrived at d=256 as optimal bottleneck

**Falsification:** High-dimensional representations (512D+) consistently outperform low-dimensional across tasks.

### 4.2 β-Overfitting Hypothesis

**Prediction:** β ∈ [1.5, 2.0] correlates with better generalization; higher β indicates overfitting.

**Rationale:**
- Dimensional collapse destroys information needed for generalization
- Over-compressed representations memorize rather than generalize

**Preliminary Evidence:** r(β, F1) ≈ -0.92 from limited experiments

**Falsification:** β shows positive correlation with task performance across domains.

### 4.3 Attention-Only Hypothesis

**Prediction:** Attention mechanisms outperform rigid boundary scaffolding for dimensional preservation.

**Preliminary Evidence:**
- Attention-only: D_eff = 34 (preserved)
- Boundary scaffolding: D_eff = 6 (collapsed, -83% loss)

**Falsification:** Boundary scaffolding systematically wins in controlled A/B tests.

### 4.4 Bilateral Asymmetry Hypothesis

**Prediction:** In adversarial or misaligned contexts, C_A→B ≠ C_B→A (asymmetric conveyance).

**Implication:** Misaligned AI systems might show high C_AI→Human (they understand us) but low C_Human→AI (we don't understand them).

**Falsification:** Bilateral measurements show no predictive value for alignment detection.

---

## 5. Context Amplification

### 5.1 Original Formulation (Superseded)

Earlier versions used exponential context amplification:

```
C_pair = Hmean(C_out, C_in) × C_ext^α × P_ij

Where α ∈ [1.5, 2.0] (super-linear amplification)
```

This predicted that context quality has super-linear effects—doubling context quality more than doubles conveyance effectiveness.

### 5.2 Current Formulation (v4.0)

Framework v4.0 replaces exponential C_ext with dimensional preservation function:

```
C_pair = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij
```

**Key insight:** Context amplification occurs through **dimensional preservation**, not exponential scaling. Good context maintains D_eff; bad context causes dimensional collapse.

### 5.3 Geometric Extension (For Advanced Analysis)

When considering manifold structure, the complete formulation includes curvature and geodesic effects:

```
C_pair^geometric = Hmean(C_out, C_in) × 
                   f_dim(D_eff) × 
                   exp(-λ/τ²) ×           [curvature penalty]
                   exp(-dist_M²/(2σ²)) ×  [geodesic distance decay]
                   P_ij
```

Where:
- τ = local reach (inverse maximum curvature)
- λ = curvature sensitivity parameter
- dist_M = geodesic distance on semantic manifold

**Interpretation:** Information flows efficiently through low-curvature regions (within semantic categories) and less efficiently across high-curvature boundaries (between categories).

---

## 6. Temporal Dynamics

### 6.1 Multi-Turn Context Evolution

In multi-turn interactions, context evolves over time:

```
C_ext(t) = f(C_ext(t-1), B_t)

Where B_t = boundary object at turn t
```

**Quality Trajectory:**
```
quality_trajectory(t) = ∏_{k=1}^{t} q(B_k)

Where q(B_k) ∈ [0, 1] = quality of boundary object k
```

**Critical insight:** Quality is multiplicative across turns. A single low-quality exchange (q ≈ 0.3) can severely degrade the entire trajectory.

### 6.2 Self-Reinforcing Cycles

**Positive cycle:** Good context → better responses → improved context → ...
**Negative cycle:** Poor context → confused responses → degraded context → ...

This explains why early interactions disproportionately determine outcomes—they set the trajectory's initial slope.

### 6.3 Threshold-Based Management

To prevent catastrophic trajectory degradation:

```
if gap(B_t, expected) < θ_refine:
    # Minor gap: refine and continue
    B_t' = refine(B_t, feedback)
    
elif gap(B_t, expected) > θ_reset:
    # Major gap: reset to last good checkpoint
    context = restore_checkpoint(t_checkpoint)
```

---

## 7. Zero-Propagation Principle

### 7.1 Definition

**Zero-propagation** occurs when any essential component of conveyance equals zero:

```
If W = 0 OR R = 0 OR H = 0 OR P_ij = 0 OR D_eff → 0:
    C_pair = 0 (categorical failure)
```

### 7.2 Implications

Zero-propagation is **categorical failure**, distinct from "very poor" conveyance:

| Condition | Result | Nature |
|-----------|--------|--------|
| All components > 0 but low | Low C_pair | Degraded but possible |
| Any component = 0 | C_pair = 0 | Impossible transfer |

**Example:** A perfect lecture (W=1.0, R=1.0, H=1.0) in a language no student speaks (P_ij=0) achieves zero conveyance—not poor conveyance, but zero.

### 7.3 Dimensional Collapse as Zero-Propagation

When D_eff collapses to near-zero, effective conveyance becomes impossible even with good W, R, H, P_ij values:

```
D_eff < threshold → f_dim(D_eff) → 0 → C_pair → 0
```

This explains why memory poisoning attacks with only ~10% contamination can cause ~95% task failure—small contamination triggers dimensional collapse, which cascades to zero-propagation.

---

## 8. Falsification Criteria

A hypothesis must be falsifiable to be scientific. The Conveyance Hypothesis would be **falsified** by:

### 8.1 Strong Falsification Evidence

1. **β shows consistent positive correlation with performance across domains**
   - Current observation: r ≈ -0.92 (negative)
   - Falsifying observation: r > +0.5 replicated across tasks

2. **High-dimensional C_ext systematically outperforms low-dimensional**
   - Current hypothesis: 128-256D optimal
   - Falsifying observation: 2048D+ consistently superior

3. **Boundary scaffolding beats attention-only in rigorous A/B tests**
   - Current observation: Attention-only preserves D_eff = 34; scaffolding collapses to D_eff = 6
   - Falsifying observation: Scaffolding wins majority of comparisons

4. **Bilateral conveyance measurements show zero predictive validity**
   - Current hypothesis: Asymmetric C_pair predicts misalignment
   - Falsifying observation: No correlation between C_pair asymmetry and outcomes

5. **P_ij compatibility shows no relationship to transfer success**
   - Current hypothesis: Protocol match enables transfer
   - Falsifying observation: Incompatible agents transfer equally well

### 8.2 Weak Falsification Evidence

- Single counterexamples (might be domain-specific)
- Mixed results without clear patterns
- Inability to measure proposed constructs reliably

---

## 9. Relationship to Existing Theories

### 9.1 Shannon's Information Theory

| Shannon | Conveyance |
|---------|------------|
| Channel capacity | Agent capacity (W × R × H) |
| Noise | Protocol mismatch (1 - P_ij) |
| Encoding | Boundary object creation |
| Decoding | Integration into receiver geometry |
| Bit error rate | Dimensional collapse (β) |

**Conveyance extends Shannon** by adding semantic effectiveness to signal fidelity.

### 9.2 Rogers' Innovation Diffusion Theory

| Rogers | Conveyance |
|--------|------------|
| Adoption curves | Conveyance effectiveness over time |
| Opinion leaders | High-conveyance nodes in networks |
| Compatibility | P_ij protocol coefficient |
| Complexity | Inverse of D_eff preservation |

**Conveyance mathematizes Rogers'** qualitative descriptions of how innovations spread.

### 9.3 Kolmogorov Complexity

| Kolmogorov | Conveyance |
|------------|------------|
| Minimum description length | Optimal boundary object compression |
| Incompressibility | Essential semantic structure |
| Algorithmic probability | Transfer success probability |

**Conveyance operationalizes** complexity concepts for agent-to-agent transfer.

---

## 10. Practical Applications (If Validated)

### 10.1 AI Development

- **Optimization targets:** Maximize D_eff rather than arbitrary metrics
- **Diagnostic tools:** Detect dimensional collapse before deployment
- **Architecture guidance:** Prefer attention-only over scaffolding
- **Training monitoring:** Watch geometric health during learning

### 10.2 AI Safety

- **Alignment detection:** Misaligned agents may show asymmetric C_pair
- **Early warning:** Geometric anomalies before behavioral symptoms
- **Interpretability:** Ground-truth about what information actually transferred

### 10.3 Memory Systems

- **Poisoning detection:** Dimensional collapse indicates contamination
- **Quality maintenance:** Monitor D_eff trajectory across interactions
- **Defense mechanisms:** Reset when D_eff drops below threshold

### 10.4 Human Communication (Speculative)

If the framework validates in AI systems, it may inform:
- Educational theory (why some teaching works)
- Organizational communication (why information gets lost in hierarchies)
- Cross-cultural understanding (how meaning transforms across contexts)

---

## 11. Current Evidence Status

### 11.1 Validated (Tier 1 Evidence)

✅ Dimensional richness correlates positively with utility  
✅ β anti-correlates with utility (r = -0.92)  
✅ Attention-only architecture preserves D_eff = 34  
✅ Boundary scaffolding collapses to D_eff = 6 (-83% loss)  
✅ L2 normalization prevents magnitude artifacts  

### 11.2 Preliminary Observations (Require Validation)

⚠️ Low-dimensional (128-256D) outperforms high-dimensional  
⚠️ β ∈ [1.5, 2.0] optimal range  
⚠️ BDH's d=256 bottleneck validates independently  

### 11.3 Unvalidated (Theoretical)

❓ Bilateral asymmetry predicts misalignment  
❓ Curvature-modulated conveyance  
❓ Temporal amplification (T^β term)  
❓ Human communication applications  

### 11.4 Falsified (Revised in v3.9+)

❌ Boundary-first approach (v3.7-3.8) — produced anti-utility  
❌ β as optimization target — now diagnostic only  
❌ φ (conductance) and κ (curvature) as primary metrics — deprecated  

---

## 12. Conclusion

The Conveyance Hypothesis proposes that **semantic transfer effectiveness is mathematically measurable**. We offer:

1. **A core equation:** C_pair = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij
2. **Measurable variables:** W, R, H, T, D_eff, β, P_ij
3. **Primary metric:** D_eff (effective dimensionality via PCA)
4. **Diagnostic metric:** β (dimensional collapse indicator)
5. **Falsification criteria:** Clear conditions that would disprove the hypothesis
6. **Preliminary validation:** β-utility anti-correlation, attention-only superiority

**Status:** This is a **hypothesis under investigation**, not a validated theory. The equations are mathematically coherent but require systematic empirical validation.

**Core claim:** Clean math ≠ empirical reality. We have elegant equations that need rigorous testing. The Low-Dimensional Hypothesis, β-overfitting correlation, and bilateral asymmetry predictions all await experimental validation.

Shannon's exclusion of meaning was wise for 1948. In 2025, with transformer architectures providing measurable embedding spaces, we may finally have the tools to include what he deliberately left out.

---

## Appendix A: Quick Reference

### Primary Equation
```
C_pair(i ↔ j) = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij
```

### Component Equations
```
C_out = (W × R × H) / T
C_in = (W × R × H) / T
f_dim(D_eff) = (D_eff / D_target)^α_dim
```

### Variables
| Symbol | Name | Range |
|--------|------|-------|
| W | Semantic Investment | [0, 1] |
| R | Relational Discovery | [0, 1] |
| H | Computational Frame | [0, 1] |
| T | Temporal Investment | [0, ∞) |
| D_eff | Effective Dimensionality | [1, D_nominal] |
| β | Collapse Indicator | [1, ∞) |
| P_ij | Protocol Compatibility | [0, 1] |
| α_dim | Dimensional Scaling | [0.5, 1.0] |

### Target Values
| Metric | Target | Warning |
|--------|--------|---------|
| D_eff (512D) | ≥ 34 | < 20 |
| D_eff (256D) | ≥ 20 | < 12 |
| β | < 2.0 | > 2.5 |
| d_nn | 0.10-0.15 | < 0.05 or > 0.25 |
| LC | ≥ 0.87 | < 0.70 |
| F1 | ≥ 0.90 | < 0.85 |

### Critical Rules
1. **ALWAYS** L2 normalize embeddings before geometric analysis
2. **NEVER** optimize for β—it's diagnostic only
3. **PRIMARY** validation is task performance, not geometric metrics
4. **WATCH** for dimensional collapse (D_eff dropping rapidly)

---

## Appendix B: Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| v1.0 | 2024 | Initial formulation with C_ext^α |
| v3.7-3.8 | Oct 2025 | Boundary-first approach (later falsified) |
| v3.9 | Oct 2025 | D_eff as primary metric, β inversion discovered |
| v4.0 | Nov 2025 | f_dim(D_eff) replaces C_ext^α, attention-only validated |

---

*"The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point."* — Claude Shannon, 1948

*"The fundamental problem of conveyance is that of transforming at one agent a semantic structure that appropriately reorganizes another agent's internal geometry."* — The Conveyance Hypothesis, 2025
