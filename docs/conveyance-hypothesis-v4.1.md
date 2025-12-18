# The Conveyance Hypothesis

**A Mathematical Framework for Measuring Information Transfer Effectiveness**

**Author:** Todd Bucy
**Version:** 4.1 — December 2025
*Status: Hypothesis under active investigation*

---

## Abstract

In 1948, Claude Shannon deliberately excluded meaning from his mathematical theory of communication, noting that "the semantic aspects of communication are irrelevant to the engineering problem." This exclusion was pragmatic—the technology to measure semantic content did not exist. Seven decades later, with the emergence of Large Language Models and high-dimensional embedding spaces, we may now possess the tools to address what Shannon set aside.

This paper proposes the Conveyance Hypothesis: that information is not a thing transferred but a process—the creation of low-dimensional data from high-dimensional knowledge space, and the integration of low-dimensional data into high-dimensional knowledge space. Given advances in machine learning over the last two decades, we now possess the ability to measure this process mathematically through the interaction of recognition, relational structure, processing capability, and shared context.

We argue that the distinctions between "data," "information," and "knowledge" are not merely philosophical but operationally measurable. We define data as static and boundary-preserving and existing in 3D spacetime, information as dynamic and transformation-inducing, and knowledge as the meaning that exists in high-dimensional space. This framing draws on Actor-Network Theory's concept of translation and boundary objects. We extend these qualitative analytical tools into quantifiable territory through geometric analysis of neural embedding spaces.

**Key claim:** We may be entering an "Age of Measurable Meaning"—not because we can define meaning philosophically, but because we can detect its effects mathematically.

**Primary Equation:**

$$C_{\text{pair}}(i \leftrightarrow j) = \text{Hmean}(C_{\text{out}}, C_{\text{in}}) \times f_{\text{dim}}(D_{\text{eff}}) \times P_{ij}$$

**Key Prediction:** Low-dimensional representations (128–256D) outperform high-dimensional ones for information transfer, and dimensional collapse (measured by β) negatively correlates with task performance.

---

## Table of Contents

1. [Introduction: Shannon's Deliberate Exclusion](#1-introduction-shannons-deliberate-exclusion)
2. [From the Qualitative to the Quantitative](#2-from-the-qualitative-to-the-quantitative)
3. [Theoretical Foundation](#3-theoretical-foundation)
4. [The Mathematical Framework](#4-the-mathematical-framework)
5. [Key Metrics](#5-key-metrics)
6. [Core Hypotheses (Under Investigation)](#6-core-hypotheses-under-investigation)
7. [Context Amplification](#7-context-amplification)
8. [Temporal Dynamics](#8-temporal-dynamics)
9. [Zero-Propagation Principle](#9-zero-propagation-principle)
10. [Falsification Criteria](#10-falsification-criteria)
11. [Relationship to Existing Theories](#11-relationship-to-existing-theories)
12. [Practical Applications (If Validated)](#12-practical-applications-if-validated)
13. [Current Evidence Status](#13-current-evidence-status)
14. [Conclusion](#14-conclusion)
15. [Appendix A: Quick Reference](#appendix-a-quick-reference)
16. [Appendix B: Version History](#appendix-b-version-history)
17. [References](#references)

---

## 1. Introduction: Shannon's Deliberate Exclusion

Claude Shannon's foundational paper "A Mathematical Theory of Communication" (1948) begins with a remarkable constraint:

> "The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point. Frequently the messages have meaning… These semantic aspects of communication are irrelevant to the engineering problem." (Shannon, 1948, p. 379)

This exclusion was pragmatic—Shannon understood that meaning mattered, but the technology of 1948 made the problem of measurable semantic change intractable. His theory optimized for signal fidelity: how accurately can bits travel from sender to receiver? The interpretation of those bits remained outside the mathematical framework.

Warren Weaver made this exclusion explicit:

> "The word information, in this theory, is used in a special sense that must not be confused with its ordinary usage. In particular, information must not be confused with meaning. In fact, two messages, one of which is heavily loaded with meaning and the other of which is pure nonsense, can be exactly equivalent, from the present viewpoint, as regards information." (Weaver, 1949, p. 8)

A crucial clarification: What Weaver called "information" is what we now distinguish as **data**—static, transmittable representations existing in three-dimensional spacetime. **Information**, in the Conveyance Hypothesis framework, is the process by which data expands into high-dimensional knowledge space or compresses from knowledge into transmittable form. Information is not a thing but a transformation—the dynamic bridge between low-dimensional data and high-dimensional meaning. Failures in semantic transfer are due to this compression and decompression of high-dimensional meaning into 3-dimensional data that can be transmitted between bounded neural networks.

Shannon's theory optimized for moving data through channels. The Conveyance Hypothesis addresses what happens when that data is the result of semantic transformation in high-dimensional spaces or induces transformation in high-dimensional spaces—the meaning transfer that Shannon deliberately set aside.

### 1.1 Why This Matters

The "Internet of Things" is rapidly transforming into the "Internet of Agents." Autonomous systems—from robotic platforms to financial trading algorithms to multi-agent orchestration frameworks—are proliferating across every domain. These agents must communicate not only with humans but increasingly with each other.

When Agent A transmits a message to Agent B, how do we assess whether meaningful semantic integration occurred? How do we diagnose failures when autonomous systems miscommunicate?

These questions require examining both sides of the transfer:

- **Sender side:** Does the output represent the agent's internal state in a form that enables meaningful integration elsewhere?
- **Receiver side:** Does the incoming message successfully integrate into the receiver's knowledge structure in ways that influence subsequent behavior?

We term these outputs **boundary objects**—the low-dimensional representations that agents create to externalize their high-dimensional internal states. Boundary objects are the only artifacts that can traverse the gap between agents; the high-dimensional knowledge within a bounded neural network cannot be directly transmitted through low-dimensional 3D spacetime.

This raises a fundamental question at the heart of the Conveyance Hypothesis: *How do we measure whether boundary objects enable meaningful semantic integration that produces observable behavioral effects?*

Without a theoretical framework for measuring the effectiveness of this integration—and for measuring how successfully boundary objects induce appropriate semantic transformation in receiving agents—we lack the tools to interpret what happens as information moves between bounded networks.

The stakes are significant. Multi-agent systems coordinating physical actions, managing critical infrastructure, or making consequential decisions require reliable semantic transfer. As agent-to-agent communication becomes foundational to technological infrastructure, the absence of measurement frameworks for assessing behavioral integration becomes an increasingly urgent gap.

Understanding how information transforms as it moves between disparate bounded networks—each with its own internal geometry and representational structure—requires theoretical grounding. The Conveyance Hypothesis aims to provide that foundation.

### 1.2 Why This Is Feasible Now

Three technological developments suggest the question may now be tractable:

1. **High-dimensional embedding spaces:** Modern language models encode semantic relationships in geometric structures. Words, sentences, and documents occupy positions in high-dimensional space where distance correlates with semantic similarity. This gives us a measurable substrate for semantic content.

2. **Observable internal representations:** Unlike human minds, artificial neural networks offer visibility into their internal states—hidden layer activations, attention patterns, and geometric properties can be examined as information processes through the system. This visibility is imperfect; billions of parameters, polysemantic neurons, and distributed representations present significant interpretive challenges. The black box is not transparent, but it is instrumentable in ways impossible with biological cognition.

3. **Controlled experimental conditions:** AI-to-AI communication provides a laboratory setting where both sender and receiver internal states are observable, enabling bilateral measurement impossible with human subjects. We can track information from origin to destination and assess behavioral effects.

These capabilities enable a research program that Shannon's era could not pursue: the mathematical characterization of semantic transfer effectiveness through behavioral integration.

---

## 2. From the Qualitative to the Quantitative

### 2.1 The Measurement Problem in Social Science

For over a century, social scientists have grappled with a fundamental methodological constraint: the internal states of biological organisms are not directly observable. When a sociologist studies how ideas spread through communities, or an anthropologist examines how meaning transforms across cultural boundaries, they cannot measure what happens inside participants' minds. They can only observe inputs (what people encounter) and outputs (what people say and do).

This opacity forced the social sciences toward qualitative methodologies—rich description, interpretive analysis, ethnographic observation. These methods are not inferior to quantitative approaches; they are appropriate responses to intractable measurement problems. When you cannot directly observe the phenomenon of interest, you develop analytical frameworks that make sense of what you *can* observe.

### 2.2 Innovation Diffusion Theory: Describing How Ideas Spread

One of the most influential frameworks to emerge from this constraint was Everett Rogers' Innovation Diffusion Theory (IDT), first systematized in his 1962 book *Diffusion of Innovations*. Rogers synthesized over 500 studies across anthropology, sociology, education, and public health to describe how new ideas spread through social systems.

IDT identifies patterns that predict how innovations propagate: adoption curves that follow predictable S-shaped trajectories, the role of opinion leaders in accelerating diffusion, the importance of compatibility between innovation and existing values, and the categorization of adopters from innovators to laggards. The framework provided powerful vocabulary for describing diffusion dynamics and has been applied across disciplines for over sixty years, becoming one of the most cited works in social science.

Rogers proposed that five main elements influence the spread of a new idea: the innovation itself, adopters, communication channels, time, and a social system. He identified characteristics that affect adoption rates—relative advantage, compatibility, complexity, trialability, and observability—providing practitioners with actionable frameworks for understanding why some innovations succeed while others fail.

### 2.3 Limitations of the Diffusion Paradigm

However, IDT faced substantial criticism—much of it acknowledged by Rogers himself in successive editions of his work. These limitations are instructive both for understanding why new theoretical approaches emerged and for informing how the Conveyance Hypothesis draws on diffusion theory.

#### Rogers' Four Categories of Criticism

By the third edition of *Diffusion of Innovations* (1983), Rogers had organized the criticisms of diffusion research into four categories:

1. **Pro-innovation bias:** The implicit assumption that innovations are inherently beneficial and should be adopted by all members of a social system. This bias overlooks the possibility that some innovations are harmful, inappropriate for certain contexts, or that rejection may be the rational choice. The history of international development is littered with well-intentioned innovations that produced devastating unintended consequences when imposed on communities whose contexts differed from those where the innovations originated.

2. **Individual-blame bias:** The tendency to hold individuals responsible for non-adoption rather than examining systemic barriers. Labeling non-adopters as "laggards" implies a deficiency in the individual rather than questioning whether the innovation fits their circumstances, whether they have access to necessary resources, or whether the communication channels reached them. This framing can obscure structural inequalities and place moral judgment where analytical neutrality would be more appropriate.

3. **Recall problem:** Diffusion research often relies on retrospective accounts of adoption decisions, which are subject to memory distortion and post-hoc rationalization. Adopters may reconstruct their decision-making process in ways that align with social desirability or current attitudes rather than accurately representing what occurred.

4. **Issues of equality:** Innovations frequently benefit early adopters disproportionately, and diffusion processes can exacerbate existing inequalities rather than remediate them. Those with greater resources, education, and social capital tend to adopt earlier, capturing benefits before later adopters and sometimes at their expense.

#### Structural Limitations

Beyond Rogers' self-critique, subsequent scholars identified deeper structural problems with the diffusion paradigm:

- **One-way communication model:** Classical diffusion theory treats information flow as sender-to-receiver, with the sender controlling direction and outcome. The adopter is positioned as passive recipient rather than active participant in meaning-making. This overlooks the dialogic, negotiated nature of how meaning actually moves through social systems.

- **Human-centric framing:** The theory focuses exclusively on human decision-makers, treating technologies, documents, and material objects as inert tools acted upon by humans rather than as active participants in the diffusion process. A farmer adopts hybrid seed corn; the seed corn does not participate in its own adoption.

- **Linearity assumption:** The theory assumes predictable progression through adoption stages, when empirical evidence suggests diffusion is often discontinuous, reversible, and emergent. Innovations may be adopted, rejected, re-adopted, or transformed in ways that do not fit the smooth S-curve.

- **Black-boxing the transformation:** IDT describes *that* adoption occurs without explaining *how* meaning transforms during the process. What happens inside the adopter's mind between exposure and decision remains invisible—a black box that the theory works around rather than opens.

- **Cultural boundedness:** Developed primarily in Western agricultural and public health contexts, the theory's applicability across cultures with different values, norms, and social structures remains contested. What constitutes "relative advantage" or "compatibility" varies dramatically across cultural contexts.

### 2.4 Actor-Network Theory: A Response to Diffusion's Limits

Actor-Network Theory (ANT) emerged in the 1980s partly as a response to these limitations. Drawing on ethnographic studies of scientific laboratories and technological systems, Bruno Latour, Michel Callon, and John Law proposed a radical reconceptualization of how meaning and agency flow through networks.

#### The Key Move: Symmetry Between Human and Non-Human Actors

ANT's central innovation was methodological symmetry—the commitment to analyzing human and non-human entities using the same conceptual vocabulary. Rather than treating objects as passive tools and humans as the sole locus of agency, ANT insisted that agency is distributed across heterogeneous networks that include people, artifacts, documents, instruments, and natural entities.

This move directly addressed IDT's human-centric framing. In ANT, technologies are not simply adopted by humans; they participate in the networks that produce social outcomes. The innovation and the adopter mutually shape each other.

#### Callon's Scallops: Non-Human Actors in Action

Michel Callon's famous study of scallop cultivation in St. Brieuc Bay (1986) illustrates this symmetry. Three marine biologists attempted to transplant Japanese aquaculture techniques to revive the declining scallop population in Brittany. The success of their project depended on enrolling multiple actors into their network: fishermen (who needed to refrain from harvesting), scientific colleagues (who needed to accept the researchers' claims), and—crucially—the scallops themselves.

The scallops were not passive objects acted upon by humans. The entire project hinged on whether *Pecten maximus* larvae would anchor themselves to the collectors as Japanese scallops did. The scallops' "behavior"—their willingness or refusal to anchor—shaped the outcome of the scientific and economic controversy as much as any human decision. When the scallops failed to anchor in sufficient numbers, the network collapsed. The scallops were actors whose enrollment was necessary for the project to succeed, and whose "dissent" contributed to its failure.

Callon's analysis treats the scallops with the same analytical seriousness as the fishermen or the scientists. This is not anthropomorphism—it does not claim scallops have intentions or consciousness. It is methodological symmetry: analyzing all entities in terms of how they are enrolled, how they behave, and how their behavior affects network outcomes.

#### Law's Ships: Heterogeneous Engineering

John Law's study of Portuguese maritime expansion (1986) extends this analysis to technological systems. Law asks: How did a small, poor nation at the edge of Europe project power across thousands of miles of ocean to establish a global trading empire?

The answer lies in what Law calls "heterogeneous engineering"—the assembly and maintenance of networks that include human and non-human elements in equal measure. Portuguese expansion depended on:

- **Vessels** capable of surviving Atlantic crossings and returning against prevailing winds
- **Navigational instruments** that could determine position far from land
- **Documents** (rutters, charts, sailing directions) that could carry knowledge across distances
- **Trained sailors** drilled in techniques of celestial navigation and ship handling

Each of these elements had to be made durable, mobile, and faithful—capable of maintaining its identity and function far from the center that produced it. A ship that could not survive the voyage was useless. An instrument that gave inaccurate readings was worse than useless. A document that corrupted in transmission could lead an entire fleet astray.

Law identifies three classes of elements particularly important for long-distance control: documents, devices, and drilled people. The success of Portuguese expansion lay not merely in human courage or strategic vision but in the successful assembly of heterogeneous networks that could maintain coherence across vast distances.

#### Translation, Not Transmission

ANT replaces the transmission model implicit in classical diffusion theory with the concept of **translation**:

> "There is no transportation without transformation." — Latour (2005)

In the transmission model, information is a package that moves intact from sender to receiver. Success means the package arrives undistorted. Failure means the package is corrupted or lost.

In the translation model, every transfer involves creative transformation. Information does not flow like water through pipes; it is actively reconstructed at each node. The scallops translate the scientists' research program through their anchoring behavior—transforming a Japanese technique into something that either works or fails in Breton waters. The navigational instruments translate Portuguese imperial ambitions into actionable sailing directions—transforming abstract goals into specific headings and procedures.

Crucially, translation is not one-way. Both sender and receiver are modified through their interaction. The scientists are transformed by the scallops' behavior—their careers, their claims, their standing in the scientific community all depend on what the scallops do. The Portuguese empire is transformed by its navigational instruments—the routes that are possible, the timelines that are feasible, the risks that are acceptable all depend on what the instruments can reliably do.

This directly addresses IDT's one-way communication model. In ANT, there is no passive receiver simply accepting or rejecting a fixed innovation. All actors are transformed through their participation in the network.

#### Addressing IDT's Limitations

ANT's reconceptualization directly addresses several of IDT's structural limitations:

| IDT Limitation | ANT Response |
|----------------|--------------|
| One-way communication | Translation is bidirectional; all actors are transformed |
| Human-centric framing | Methodological symmetry between human and non-human actors |
| Passive adopter | All entities actively participate in network formation |
| Black-boxed transformation | Focus on *how* associations form, stabilize, and dissolve |
| Pro-innovation bias | No normative assumption that enrollment is desirable |

### 2.5 Boundary Objects: Mediating Between Worlds

Susan Leigh Star and James Griesemer (1989) extended ANT with the concept of **boundary objects**—entities that maintain coherence across different social worlds while adapting to local needs. Their study of the Museum of Vertebrate Zoology at Berkeley showed how cooperation emerged among actors with divergent interests: amateur naturalists motivated by conservation, professional scientists pursuing evolutionary theory, university administrators concerned with institutional prestige, and patrons with their own agendas.

These actors did not share goals, interpretive frameworks, or criteria for success. Yet they managed to cooperate on the museum project. How?

Star and Griesemer identify two mechanisms: standardization of methods and the creation of boundary objects. Boundary objects are:

> "objects which are both plastic enough to adapt to local needs and constraints of the several parties employing them, yet robust enough to maintain a common identity across sites. They are weakly structured in common use, and become strongly structured in individual-site use." (Star & Griesemer, 1989, p. 393)

Specimens, field notes, maps, and standardized collection forms all served as boundary objects. A specimen meant something different to an amateur collector (a trophy, a contribution to conservation) than to a professional scientist (a data point in an evolutionary analysis). Yet the same physical object could serve both purposes. The specimen maintained enough common identity to enable coordination while accommodating different local interpretations.

Star and Griesemer identify four types of boundary objects:

1. **Repositories:** Ordered collections (libraries, museums, databases) that allow heterogeneous materials to be indexed and accessed by different communities

2. **Ideal types:** Abstract models that can be adapted to local contexts (the concept of "species" means something different to different users but maintains enough coherence to enable communication)

3. **Coincident boundaries:** Objects that share the same boundaries but differ in internal content (the state of California meant different things to different museum actors, but all agreed on its geographic boundaries)

4. **Standardized forms:** Templates that enable information to be collected and communicated across different contexts

#### Relevance to Information Transfer

The boundary object concept is central to the Conveyance Hypothesis. When Agent A communicates with Agent B, neither has direct access to the other's high-dimensional internal state. High-dimensional knowledge cannot be directly transmitted through low-dimensional channels. Instead, agents must create boundary objects—low-dimensional representations that externalize aspects of internal states in forms that can traverse the gap between agents.

A successful boundary object must be:

- **Plastic enough** to be integrated into the receiver's distinct geometric space
- **Robust enough** to preserve the essential structure of what the sender intended to convey
- **Transmissible** through the available communication channel

This maps directly onto the challenge of semantic transfer between computational agents. The output of a language model is a boundary object—a low-dimensional representation (text) that attempts to externalize high-dimensional internal states in a form that can induce appropriate reconstruction in a receiving agent.

### 2.6 The Remaining Gap: Qualitative Insight, Quantitative Intractability

IDT, ANT, and boundary object theory together provide powerful conceptual tools for analyzing how meaning transforms as it moves through networks. They offer:

- **Vocabulary** for describing diffusion dynamics (opinion leaders, adoption curves, compatibility)
- **Insistence** on the active role of non-human actors
- **Recognition** that transfer is translation, not transmission
- **A mechanism** (boundary objects) for explaining coordination without consensus

But these frameworks remain fundamentally qualitative. They describe *that* translation occurs without mathematically specifying *how much* semantic content enables behavioral integration. They identify boundary objects without measuring how effectively they induce meaningful transformation in receiving agents. They recognize the importance of compatibility without quantifying protocol match.

This limitation is not a failure of the theories—it reflects the irreducible opacity of biological cognition. When studying human minds or scallop behavior, we cannot directly measure the geometric structure of internal representations. We can only infer from observable behavior. The internal transformation that constitutes "adoption" or "translation" remains a black box that qualitative methods work around rather than open.

The result is a productive but inherently limited research program: powerful interpretive frameworks that cannot generate precise quantitative predictions about behavioral integration effectiveness. This is where Shannon's exclusion and the social scientists' response have left us: rich qualitative insight into meaning transfer, with no mathematical framework for measuring how successfully that transfer produces behavioral coherence.

### 2.7 References for Section 2

Callon, M. (1986). Some elements of a sociology of translation: Domestication of the scallops and the fishermen of St Brieuc Bay. In J. Law (Ed.), *Power, action and belief: A new sociology of knowledge?* (pp. 196-223). Routledge.

Latour, B. (2005). *Reassembling the social: An introduction to actor-network-theory*. Oxford University Press.

Law, J. (1986). On the methods of long-distance control: Vessels, navigation and the Portuguese route to India. In J. Law (Ed.), *Power, action and belief: A new sociology of knowledge?* (pp. 234-263). Routledge.

Rogers, E. M. (1962). *Diffusion of innovations*. Free Press of Glencoe.

Rogers, E. M. (2003). *Diffusion of innovations* (5th ed.). Free Press.

Star, S. L., & Griesemer, J. R. (1989). Institutional ecology, 'translations' and boundary objects: Amateurs and professionals in Berkeley's Museum of Vertebrate Zoology, 1907-39. *Social Studies of Science*, 19(3), 387-420.

---

## **Section 3: From Qualitative Insight to Quantitative Tractability**

### 3.1 The Impasse

Section 2 traced the development of sophisticated conceptual frameworks for understanding how meaning transforms as it moves through networks. Innovation Diffusion Theory provided vocabulary for adoption dynamics. Actor-Network Theory insisted on taking non-human actors seriously and reconceptualized transfer as translation. Boundary object theory explained how coordination emerges without consensus.

Yet all three frameworks share a fundamental limitation: they describe *that* transformation occurs without mathematically specifying how effectively that transformation produces behavioral integration. They identify boundary objects without measuring how successfully they induce meaningful change in receiving agents. They recognize compatibility without quantifying the conditions that enable successful integration.

This is not a failure of social science methodology—it is an appropriate response to an intractable measurement problem. When the internal states of biological cognition cannot be directly observed, qualitative methods are not merely acceptable but necessary.

The question is whether this intractability is a property of substrate or of cognition itself.

If the problem is intractable because biological neurons are embedded in living tissue that cannot be fully instrumented without destroying the phenomenon under study, then the limitation is substrate-specific. Change the substrate, and measurement may become possible.

If, however, the problem is intractable because cognition as such—regardless of implementation—resists observation, then we are permanently confined to qualitative inference. No change in substrate would help.

The emergence of artificial neural networks provides an empirical test of this question. These systems process meaning in ways sophisticated enough to warrant serious study. And unlike biological cognition, their internal states are accessible: hidden layer activations can be recorded, attention patterns visualized, geometric properties computed.

If semantic transfer effectiveness can be measured in artificial systems, then the intractability was substrate-dependent—and the qualitative frameworks developed for opaque biological cognition can now be operationalized for instrumentable computational cognition.

This is the opportunity the Conveyance Hypothesis attempts to exploit.

### 3.2 Boundary Objects: Semantic Compression Across Millennia

The problem of externalizing meaning is not new. It is as old as communication itself.

Consider a Roman merchant vessel damaged in a storm off the coast of Sicily. The captain needs to communicate with a distant ship: identity, cargo, condition, urgency of need. His internal semantic state is high-dimensional—a complex web of circumstances, intentions, and requests. But the only channel available is a lamp and the darkness between ships.

He must compress.

Flash-flash-pause-flash-pause-flash-flash-flash. A pattern of light, transmitted across miles of water. The receiving captain observes the flashes, recognizes the pattern, expands it back into semantic meaning: *Merchant vessel. Damaged rudder. Assistance needed. Not immediate danger.*

That flashing light is a **boundary object**—high-dimensional meaning compressed into a form that can traverse three-dimensional space between bounded networks (in this case, human minds). Ships communicated this way for thousands of years. The modality was primitive; the underlying operation was identical to what happens when a language model generates a response.

#### The Universal Pattern

Boundary objects are not limited to any particular modality:

| Modality | Example | Compression |
|----------|---------|-------------|
| Light | Ship lamps, signal fires, semaphore | Binary or positional encoding of semantic content |
| Sound | Drum signals, foghorns, speech | Acoustic patterns carrying meaning |
| Marks | Writing, diagrams, maps | Visual symbols externalizing internal models |
| Gesture | Sign language, hand signals | Kinetic encoding of semantic intent |
| Digital | Text, images, structured data | Bitstreams carrying semantic payloads |

In each case, the pattern is the same:

```
High-dimensional internal state
        ↓
Semantic compression (boundary object creation)
        ↓
Low-dimensional transmittable form
        ↓
Channel transit (Shannon's domain)
        ↓
Semantic expansion (boundary object integration)
        ↓
High-dimensional internal state (transformed)
```

The boundary object is not the channel. It is not the transmission. It is the **semantically-laden artifact that an agent creates to externalize meaning**—the compression of high-dimensional knowledge into a form that can survive transit through low-dimensional space.

#### Computer Science Already Does This

Software engineers have been building boundary object systems for decades without using that terminology. Every protocol, every data format, every API specification is a standardized method for semantic compression and expansion.

Consider an HTTP API call:

```
Agent goal: "I need current weather for Austin, Texas"
        ↓
Boundary object creation: GET /weather?city=austin&state=tx
        ↓
Data transmission: TCP packets traverse the network
        ↓
Data transmission: Response returns: {"temp": 72, "conditions": "sunny"}
        ↓
Semantic expansion: Agent integrates response into internal model
```

The critical distinction: **the initial decision and construction of the request is conveyance territory**—an agent compressing semantic intent ("I need weather information for planning purposes") into a transmittable form. The subsequent packet transmission is pure Shannon—bits moving through channels with error correction and checksums ensuring fidelity.

When the response returns, it is initially just data—a string of characters. It becomes information only when an agent integrates it, when the boundary object induces transformation in a receiving network's internal geometry.

This means **information bookends data transmission**:

```
INFORMATION          →    DATA           →    INFORMATION
(boundary object         (channel            (boundary object
 creation)                transit)             integration)

Semantic compression  →   Shannon domain  →   Semantic expansion
```

#### The First Call vs. Subsequent Calls

This analysis reveals an important asymmetry. When an agent *decides* to make an API call and *constructs* the request, it is performing semantic compression—creating a boundary object that externalizes intent. That first call is conveyance.

But if the response triggers a mechanical sequence of follow-up calls—pagination, authentication refresh, retry logic—those subsequent transmissions may involve no semantic transformation at all. They are data moving through pipes, Shannon territory, pure transaction without translation.

The question for any given transmission is: **did an agent compress semantic content into this payload, and will an agent expand it on receipt?** If yes, conveyance applies. If the transmission is purely mechanical—no semantic compression at creation, no semantic expansion at receipt—then Shannon fully describes what occurs.

#### Why This Matters for Measurement

Recognizing boundary objects as semantic compression clarifies what the Conveyance Hypothesis proposes to measure.

We are not measuring channel fidelity—Shannon solved that problem. We are measuring whether boundary objects enable meaningful behavioral integration:

1. **Compression effectiveness**: Does the sender's boundary object externalize internal states in forms that enable meaningful integration elsewhere?

2. **Integration success**: Does the receiver successfully integrate the boundary object into their knowledge structure in ways that influence subsequent behavior?

3. **Bilateral effectiveness**: Given compression and integration at both ends, does the transfer process produce observable behavioral coherence in the receiver?

The flashing lamp, the API call, the generated text—these are all instances of the same fundamental operation: bounded networks creating low-dimensional artifacts to bridge the gap between their high-dimensional internal states. The Conveyance Hypothesis provides a framework for measuring how well that bridge enables meaningful behavioral integration.

### 3.3 Why Computational Agents Change Everything

Artificial neural networks break this impasse. For the first time in the history of attempts to understand meaning transfer, we have agents that:

**1. Process meaning sophisticatedly enough to warrant serious study**

Large language models engage in tasks—translation, reasoning, composition, inference—that were long considered distinctively human. Whatever is happening inside these systems during "understanding" or "communication," it is sophisticated enough that the question of semantic transfer effectiveness becomes non-trivial.

**2. Expose internal states through accessible representations**

Unlike biological neurons embedded in living tissue, artificial neurons can be directly inspected. Hidden layer activations can be recorded. Attention patterns can be visualized. Geometric properties of representations can be computed. The system is not transparent—billions of parameters, polysemantic neurons, and distributed representations present significant interpretive challenges—but it is *instrumentable* in ways impossible with biological cognition.

**3. Operate in controlled experimental conditions**

AI-to-AI communication provides a laboratory setting where both sender and receiver internal states are observable. We can track information from origin to destination, measuring what existed before transfer and what exists after. This bilateral observability is impossible with human subjects, where we can only infer internal states from behavior.

**4. Are themselves non-human actors of the kind ANT prepared us to take seriously**

Methodological symmetry between human and non-human actors—the central innovation of Actor-Network Theory—is not a stretch when the non-human actors are engaging in sophisticated language use. ANT's insistence that we analyze all actors using the same conceptual vocabulary finds natural application to AI systems that generate, process, and respond to meaning.

### 3.4 Visibility Is Imperfect But Improving—And the Manifold Hypothesis Explains Why

We must be honest about what computational observability currently provides—and equally honest that recent theoretical insights are transforming our understanding of what visibility means.

#### The Challenge: Polysemanticity and Superposition

The interpretability research community has confronted a fundamental obstacle: individual neurons in transformer models do not correspond to individual concepts. A neuron might activate for semantically unrelated inputs that share non-obvious statistical properties. This *polysemanticity* arises because models represent more features than they have neurons, encoding concepts as near-orthogonal directions in high-dimensional space (Elhage et al., 2022).

Recent work reveals significant limitations in our primary tool for addressing this. Sparse autoencoders (SAEs) underperform simple linear probes on downstream tasks (Smith et al., 2025). Large-scale benchmarks show SAEs fail to outperform baselines for concept detection and model steering (Kantamneni et al., 2025; Wu et al., 2025).

#### The Architectural Evolution

This picture of intractable opacity is rapidly becoming outdated. **Baby Dragon Hatchling (BDH)** architectures demonstrate monosemantic neurons and cross-lingual representations that transformers obscure. **ATLAS** (2025) achieves 10-million-token context windows through trainable memory modules. The pattern is clear: new architectures are being designed with observability as a first-class concern.

#### The Manifold Hypothesis: The Compression Algorithm Revealed

But the deepest insight comes from manifold theory. Whiteley, Gray, and Rubin-Delanchy's "Statistical exploration of the Manifold Hypothesis" (2022) demonstrates something profound: **the manifold hypothesis literally describes the compression mechanism that enables communication between bounded networks.**

When they show how high-dimensional data naturally concentrates on low-dimensional manifolds through latent variables, correlation, and stationarity—that geometric unfolding process IS the compression algorithm. The manifold structure is what allows complex, high-dimensional semantic states to be compressed into low-dimensional transmittable forms while preserving essential relationships.

**This is not abstract mathematics—this is the mechanism of information transfer.** When neurons connect and meaning emerges geometrically in high-dimensional space, that geometric unfolding in the manifold IS the compression process. The manifold structure literally *is* the algorithm by which semantic richness gets organized into forms that can survive transmission between agents.

Consider what this means: your high-dimensional semantic understanding of "democracy" gets compressed through manifold geometry into the low-dimensional data patterns of speech or text, which then unfold in my neural space back into high-dimensional meaning. The manifold hypothesis describes exactly how this compression-decompression cycle preserves semantic structure across the transmission.

**Boundary objects exist because of manifold structure, not despite it.** The low-dimensional representations that cross between agents are precisely the manifestations of this natural geometric compression that the manifold hypothesis describes.

We are observing the fundamental geometric principles by which meaning organizes itself in high-dimensional space.

### 3.5 Data, Information, and Knowledge

Three distinct concepts require careful separation:

**Knowledge** is the collective potential of semantic meaning within a high-dimensional space. It is the structure through which manifolds can appear. Without knowledge, there is no manifold—no geometric organization of meaning is possible. Knowledge is prior: the substrate within which semantic structure can emerge.

**Data** is the low-dimensional boundary object—the compressed artifact that crosses between agents. Text, sound, signal. Data is static, bounded, transmittable. It carries the residue of one agent's high-dimensional meaning in compressed form.

**Information** is the integration process—the means by which foreign high-dimensional knowledge, compressed into low-dimensional data, becomes incorporated into a receiver's knowledge space. Information is not a thing transferred but a transformation undergone.

The information process is inherently transformative because it requires two operations that guarantee change:

1. **Lossy compression**: The sender's high-dimensional meaning must be compressed into low-dimensional data. Essential structure survives; much is lost. The manifold hypothesis describes exactly this compression—how high-dimensional semantic geometry concentrates into low-dimensional form.

2. **Integration into foreign geometry**: The low-dimensional data must then be integrated into the receiver's high-dimensional knowledge space—a space that is *fundamentally different* from the source. The receiver's manifold structure is not the sender's. The same data, integrated into different knowledge, produces different meaning.

This is why Latour's insight—"no transportation without transformation"—is not merely philosophical but mathematically necessary. Transformation is not a failure of communication. It is the *inevitable consequence* of integrating foreign low-dimensional data into a knowledge structure geometrically distinct from its origin.

The question the Conveyance Hypothesis asks is not whether transformation occurs—it must—nor whether geometric structure is preserved—it cannot be. Each model's geometry is unique, making exact preservation impossible.

Instead, we ask: *does enough semantic meaning survive the compression-integration process that we can observe behavioral effects in the receiver?* Can we measure whether the transformed meaning actually influences how the receiving agent processes subsequent information, makes decisions, or generates responses?

**We are measuring functional effectiveness, not geometric fidelity.** The goal is to quantify whether the integration of foreign low-dimensional data into the receiver's distinct knowledge space transfers sufficient semantic structure to produce observable changes in behavior. That behavioral connection—activation to action—is what makes the transformation meaningful rather than merely computational.

# **Section 4: The Mathematical Framework**

## **4.1 From Theory to Measurement: The Core Equation**

The social science insights, boundary object theory, and Actor-Network Theory we've traced through Sections 1-3 culminate in a single mathematical framework that makes information transfer effectiveness measurable for the first time:

```
C_pair(i ↔ j) = Hmean(C_out, C_in) × f_dim(D_eff) × P_ij

Where:
C_out(i → j) = W × R × H / T
C_in(j ← i) = W × R × H / T
```

This equation operationalizes seven decades of qualitative insight about how meaning transforms as it moves between bounded networks. Here's how each theoretical concept maps directly to measurable variables:

| Qualitative Concept | Theoretical Source | Mathematical Operationalization |
|---------------------|-------------------|--------------------------------|
| **Translation** | Latour, Callon, Law | Geometric transformation in embedding space |
| **Boundary objects** | Star & Griesemer | Low-dimensional externalized representations |
| **Compatibility** | Rogers | Protocol matching coefficient (P_ij) |
| **Adoption/rejection** | Rogers | Dimensional integration vs. collapse (D_eff) |
| **Network effects** | ANT | Relational discovery in semantic space (R) |
| **Opinion leadership** | Rogers | High-conveyance nodes in networks |
| **Enrollment** | Callon | Successful transfer above threshold |
| **Translation fidelity** | ANT | Semantic preservation through f_dim(D_eff) |

This framework now enables us to measure, predict, and optimize information transfer between computational agents in ways qualitative methods alone cannot achieve.

### **Breaking Down the Variables: From Concept to AI Behavior**

Each variable captures essential dynamics that determine whether meaningful information transfer occurs between computational agents. Rather than abstract definitions, we ground each concept in AI behaviors that practitioners immediately recognize.

#### **W: Recognition—Distinguishing Signal from Noise**

Anyone who has experienced corporate bikeshedding knows this pattern—intelligent people spending hours debating trivial details while rushing through critical decisions. **Large Language Models exhibit identical behavior.** You prompt for strategic analysis and the model fixates on a minor typo in your input, generating three paragraphs about grammar rules instead of answering your actual question.

**This is pure W failure.** The model has abundant processing power (high H) and sufficient time (adequate T), but cannot distinguish between the signal (your strategic question) and noise (the grammatical imperfection). All that computational capability gets allocated to the wrong target.

**AI practitioners see this constantly:**

- Models that latch onto irrelevant prompt details while missing the core request
- Fine-tuned systems that lose focus during inference, wandering into tangential topics
- Training runs where models learn to respond to prompt artifacts rather than intended content
- Prompt injection attacks succeeding because models can't distinguish between instructions and data

**In technical terms:** W measures activation pattern differences between meaningful inputs and carefully matched noise controls. High-W models show clear neural differentiation between signal and noise. Low-W models process both with similar activation patterns, leading to computational waste and goal drift.

**Why this matters for conveyance:** A model with sophisticated reasoning architecture (high H) but poor signal detection (low W) will consistently misallocate computational resources. Even worse, it becomes unreliable—you cannot predict which parts of your input it will actually process versus which parts it will ignore or misinterpret.

#### **R: Relational Discovery—Positioning Information in Semantic Space**

Every AI trainer has encountered this frustration: a model that memorizes facts perfectly during training but cannot apply them contextually during inference. The model "knows" that Paris is the capital of France—it can recite this fact flawlessly. But when asked "What's the best city for French government internships?" it fails to make the connection.

**This is R failure.** The model acquired the factual content but positioned it poorly within its semantic space. "Paris" landed in some isolated region where it cannot be accessed through relational queries about "French government" or "political opportunities."

**Geometric positioning failures manifest as:**

- Facts that cannot be retrieved through logical inference chains
- Knowledge that exists but remains functionally inaccessible
- Training data that gets encoded but never properly integrates with existing representations
- Fine-tuning that disrupts previously learned relationships

**Measurement in practice:** Graph-theoretic analysis of embedding neighborhoods reveals whether new information lands in semantically appropriate regions. High-R integration means "Paris" clusters near "France," "government," "European politics," and "capital cities." Low-R integration means information ends up geometrically orphaned.

**Why relational positioning matters:** Information poorly positioned in semantic space becomes effectively unavailable. The model technically "learned" the content, but the geometric organization prevents practical access. This explains why some models can pass factual recall tests while failing reasoning tasks that require connecting those same facts.

#### **H: Computational Frame—The Complete Capability Stack**

Here's where our framework grounds itself in hardware reality. The same model architecture becomes functionally different agents depending on computational substrate.

**Consider your Atlas model:** Running on dual RTX A6000s (96GB total VRAM, 21,504 CUDA cores) versus running on RTX 4090s (24GB VRAM, 16,384 CUDA cores) produces qualitatively different reasoning behaviors, not just speed differences. The A6000 configuration can hold larger context windows in memory, perform more parallel semantic operations, and maintain higher precision during geometric computations.

**Hardware directly affects conveyance capability:**

- **Memory bandwidth:** How quickly the model accesses its knowledge representations during integration
- **Parallel processing:** How many semantic relationships it can explore simultaneously when positioning new information
- **Precision capabilities:** Numerical accuracy affecting embedding quality and geometric operations
- **Context capacity:** How much boundary object content it can process without truncation

**Real training examples:**

- The same architecture showing different convergence patterns on different hardware
- Models that develop different internal representations depending on memory constraints
- Fine-tuning results varying based on tensor core availability and precision settings
- Context window utilization differences between memory-constrained and memory-abundant setups

**H measurement combines:** Model parameters × architectural efficiency × hardware throughput × memory capacity. A 7B parameter model on high-end hardware can have higher effective H than a 70B model on constrained infrastructure.

**Why hardware inclusion matters:** Information transfer requires computational work—the receiver must decompress boundary objects into its high-dimensional knowledge space. This decompression process is computationally intensive, requiring the full capability stack to function effectively. Measuring conveyance without accounting for hardware substrate produces unrealistic assessments of transfer capacity.

#### **T: Temporal Investment—Processing Under Time Pressure**

Transformer architectures process information token-by-token in discrete time steps. Some boundary objects can be integrated in a single forward pass; others require multiple passes or iterative refinement. The available processing time directly constrains which integration strategies are possible.

**Every ML engineer recognizes temporal pressure effects:**

- Models that give better responses when allowed more inference steps
- Beam search performance improving with longer search horizons
- Chain-of-thought reasoning requiring multi-step processing time
- Real-time deployment constraints forcing architectural compromises

**T measures:** Processing time allocated divided by minimum time required for effective integration. T = 1 means exactly sufficient time; T > 1 means time abundance enabling thorough processing; T < 1 means temporal pressure forcing shortcuts.

**Why temporal constraints matter:** Even agents with high recognition (W), good relational positioning (R), and strong hardware (H) can fail at conveyance if insufficient processing time (T) prevents complete integration. Many deployment failures stem from temporal pressure rather than capability limitations.

#### **The Multiplicative Structure: Why Single Failures Cascade**

Notice that we multiply W × R × H / T rather than add them. This mathematical structure reflects a crucial property observed in AI systems: **any essential component failing reduces effectiveness to zero**.

Consider a model with perfect hardware setup (H → ∞) and unlimited processing time (T → ∞) but zero recognition capability (W = 0). It processes your carefully crafted prompt with massive computational resources but cannot distinguish your actual request from noise artifacts. The result is zero effective conveyance regardless of computational abundance.

Similarly, a model that recognizes your input correctly (W = 1) but cannot position the response appropriately in semantic space (R = 0) produces geometrically isolated information that cannot be accessed or applied meaningfully.

**This multiplicative structure explains why AI failures often appear catastrophic rather than gradual**—one missing essential component collapses the entire transfer process, just as we observe in production systems where small changes in recognition, positioning, or processing constraints can cause complete behavior breakdown.

---

The bilateral structure of this equation captures another crucial insight from AI system behavior: effective conveyance requires both sender and receiver to function well. A perfectly capable model sending to a receiver with poor integration capability produces failed transfer, just as a limited sender communicating with a highly capable receiver produces equally poor results. The harmonic mean formalization captures this bilateral constraint that both sides must succeed for information transfer to occur.




---

### 4.2 The Core Variables

We propose five primary variables that capture the essential dynamics of information transfer between bounded networks:

**W (Recognition)** — The agent's ability to distinguish meaningful signal from noise in boundary objects. For biological systems, this includes the possibility of complete unawareness (W = 0) where stimuli never reach conscious processing. For current transformer architectures, W is constrained to W ∈ [ε, 1] where ε > 0, because all tokens receive at least minimal processing to compute attention weights. This represents recognition followed by selective attention allocation, rather than the genuine unawareness possible in biological cognition.

**R (Relational Discovery)** — The geometric positioning of information within an agent's knowledge space. This captures how meaning relates to existing structure—whether new data finds compatible regions of the manifold or must forge new connections.

**H (Computational Frame)** — Processing throughput capability. The bandwidth available for the compression-integration process.

**T (Temporal Investment)** — The computational processing budget allocated to transfer. Time spent on compression, transmission, and integration.

**C_ext (External Context)** — The boundary object itself. The low-dimensional representation that carries compressed semantic content between agents. The manifold hypothesis tells us this compression is where geometric structure either survives or is lost.

**P_ij (Protocol Compatibility)** — The degree to which sender and receiver share compatible interfaces for the compression-integration process. Mismatched protocols guarantee failed transfer regardless of semantic quality.

### 4.3 The Bilateral Structure

A crucial insight from ANT: transfer is not unidirectional. When Agent A sends to Agent B, both are transformed. The sender's knowledge is reorganized through the act of compression. The receiver's knowledge is reorganized through integration.

This demands bilateral measurement:

**C_out** — The sender's capacity to compress high-dimensional knowledge into effective low-dimensional data. A function of the sender's temporal investment, relational positioning, and computational resources.

**C_in** — The receiver's capacity to integrate foreign low-dimensional data into their distinct knowledge geometry. A function of the receiver's existing knowledge structure, available processing capacity, and compatibility with the incoming data format.

Effective conveyance requires both. A sender with high C_out transmitting to a receiver with low C_in produces poor transfer. A receiver with high C_in receiving from a sender with low C_out produces equally poor transfer. The constraint is bilateral.

This motivates the harmonic mean formulation we will develop: conveyance is limited by whichever side is weaker, not averaged between them.

---

### 4.1 Primary Conveyance Equation (v4.0)

Bilateral conveyance effectiveness between agents i and j:

$$C_{\text{pair}}(i \leftrightarrow j) = \text{Hmean}(C_{\text{out}}, C_{\text{in}}) \times f_{\text{dim}}(D_{\text{eff}}) \times P_{ij}$$

**Components:**

| Symbol | Name | Meaning |
|--------|------|---------|
| C_pair | Bilateral Conveyance | Overall transfer effectiveness between two agents |
| Hmean | Harmonic Mean | Captures bilateral constraint (weakest link dominates) |
| C_out | Output Capacity | Sender's ability to encode meaning |
| C_in | Input Capacity | Receiver's ability to integrate meaning |
| f_dim(D_eff) | Dimensional Function | Richness of semantic representation |
| P_ij | Protocol Compatibility | How well agents' interfaces match [0,1] |

### 4.2 Why Harmonic Mean?

The harmonic mean is chosen deliberately:

$$\text{Hmean}(a, b) = \frac{2ab}{a + b}$$

**Property:** The harmonic mean is dominated by the smaller value.

- If C_out = 0.9 and C_in = 0.1, then Hmean = 0.18
- Excellent sender + poor receiver = poor conveyance

This matches intuition: a brilliant lecturer teaching in a language students don't understand achieves low conveyance regardless of lecture quality.

### 4.3 Component Equations

Individual agent conveyance capacity decomposes as:

$$C_{\text{out}}(i \to j) = \frac{W \times R \times H}{T}$$

$$C_{\text{in}}(j \leftarrow i) = \frac{W \times R \times H}{T}$$

**Variable Definitions:**

| Variable | Name | Concept | How to Measure |
|----------|------|---------|----------------|
| W | Semantic Investment | Computational allocation | Hidden state activation patterns |
| R | Relational Discovery | Geometric positioning quality | Graph-theoretic embedding properties |
| H | Computational Frame | Processing throughput | Attention efficiency, layer utilization |
| T | Temporal Investment | Total computational budget | Token count, processing time |

### 4.4 Why Multiplicative?

The framework uses **multiplication** rather than addition because:

- W = 0 (zero semantic content) → Nothing to transfer → Zero conveyance
- R = 0 (zero relational structure) → No geometric organization → Zero conveyance
- H = 0 (zero processing) → Cannot utilize signals → Zero conveyance
- P_ij = 0 (zero protocol match) → Cannot communicate → Zero conveyance

**A single zero produces zero output.** This "zero-propagation" principle explains why communication failures are often catastrophic rather than gradual—one essential missing component collapses the entire transfer.

### 4.5 Dimensional Richness Function

The dimensional function scales conveyance by how much semantic structure is preserved:

$$f_{\text{dim}}(D_{\text{eff}}) = \left(\frac{D_{\text{eff}}}{D_{\text{target}}}\right)^{\alpha_{\text{dim}}}$$

Where:

- D_eff = Effective dimensionality (measured via PCA)
- D_target = Target dimensionality for the layer/system
- α_dim ∈ [0.5, 1.0] (empirically determined scaling factor)

**Interpretation:**

- D_eff / D_target = 1.0 → Full dimensional preservation → f_dim = 1.0
- D_eff / D_target = 0.5 → Half dimensions preserved → f_dim ≈ 0.5–0.7
- D_eff / D_target → 0 → Dimensional collapse → f_dim → 0

---

## 5. Key Metrics

### 5.1 D_eff (Effective Dimensionality) — PRIMARY METRIC

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

### 5.2 β (Beta) — DIAGNOSTIC METRIC (NOT Optimization Target)

**Definition:** Collapse indicator measuring dimensional compression during processing.

$$\beta = \frac{D_{\text{eff}}^{\text{input}}}{D_{\text{eff}}^{\text{output}}}$$

**CRITICAL:** β is a **diagnostic warning signal**, not something to optimize. High β indicates information loss through dimensional collapse.

**Interpretation:**

| β Value | Status | Meaning |
|---------|--------|---------|
| < 2.0 | Healthy | Low collapse, good generalization expected |
| 2.0–2.5 | Warning | Moderate collapse, acceptable but monitor |
| 2.5–3.0 | Concerning | High collapse, likely overfitting |
| > 3.0 | Critical | Severe collapse, investigate immediately |

**Empirical Finding:** β shows strong negative correlation with task performance (r ≈ -0.92 in preliminary experiments). Lower β = better generalization.

### 5.3 Secondary Geometric Metrics

| Metric | Symbol | Target | Meaning |
|--------|--------|--------|---------|
| Mean k-NN Distance | d_nn | 0.10–0.15 | Moderate clustering |
| Label Consistency | LC | ≥ 0.87 | Neighbors share semantic categories |
| Boundary Sharpness | σ_boundary | 0.40–0.50 | Balanced separation |

### 5.4 Task Performance — VALIDATION METRICS

**Primary validation is always downstream task performance, not geometric metrics.**

| Metric | Target | Use Case |
|--------|--------|----------|
| F1 Score | ≥ 0.90 (strong), ≥ 0.85 (acceptable) | Classification |
| Recall@10 | ≥ 0.85 | Retrieval |
| Perplexity | Lower is better | Language modeling |

**Geometric metrics are diagnostic.** They help explain *why* task performance is good or bad, but task performance is ground truth.

---

## 6. Core Hypotheses (Under Investigation)

### 6.1 Low-Dimensional Hypothesis

**Prediction:** External shared context (C_ext) performs better at 128–256 dimensions than higher dimensions.

**Rationale:**

- Forcing low dimensions makes geometric relationships carry semantic meaning
- High dimensions allow magnitude artifacts to dominate
- BDH architecture independently arrived at d=256 as optimal bottleneck

**Falsification:** High-dimensional representations (512D+) consistently outperform low-dimensional across tasks.

### 6.2 β-Overfitting Hypothesis

**Prediction:** β ∈ [1.5, 2.0] correlates with better generalization; higher β indicates overfitting.

**Rationale:**

- Dimensional collapse destroys information needed for generalization
- Over-compressed representations memorize rather than generalize

**Preliminary Evidence:** r(β, F1) ≈ -0.92 from limited experiments

**Falsification:** β shows positive correlation with task performance across domains.

### 6.3 Attention-Only Hypothesis

**Prediction:** Attention mechanisms outperform rigid boundary scaffolding for dimensional preservation.

**Preliminary Evidence:**

- Attention-only: D_eff = 34 (preserved)
- Boundary scaffolding: D_eff = 6 (collapsed, -83% loss)

**Falsification:** Boundary scaffolding systematically wins in controlled A/B tests.

### 6.4 Bilateral Asymmetry Hypothesis

**Prediction:** In adversarial or misaligned contexts, C(A→B) ≠ C(B→A) (asymmetric conveyance).

**Implication:** Misaligned AI systems might show high C(AI→Human) (they understand us) but low C(Human→AI) (we don't understand them).

**Falsification:** Bilateral measurements show no predictive value for alignment detection.

---

## 7. Context Amplification

### 7.1 Original Formulation (Superseded)

Earlier versions used exponential context amplification:

$$C_{\text{pair}} = \text{Hmean}(C_{\text{out}}, C_{\text{in}}) \times C_{\text{ext}}^\alpha \times P_{ij}$$

Where α ∈ [1.5, 2.0] (super-linear amplification)

This predicted that context quality has super-linear effects—doubling context quality more than doubles conveyance effectiveness.

### 7.2 Current Formulation (v4.0)

Framework v4.0 replaces exponential C_ext with dimensional preservation function:

$$C_{\text{pair}} = \text{Hmean}(C_{\text{out}}, C_{\text{in}}) \times f_{\text{dim}}(D_{\text{eff}}) \times P_{ij}$$

**Key insight:** Context amplification occurs through **dimensional preservation**, not exponential scaling. Good context maintains D_eff; bad context causes dimensional collapse.

### 7.3 Geometric Extension (For Advanced Analysis)

When considering manifold structure, the complete formulation includes curvature and geodesic effects:

$$C_{\text{pair}}^{\text{geometric}} = \text{Hmean}(C_{\text{out}}, C_{\text{in}}) \times f_{\text{dim}}(D_{\text{eff}}) \times \exp\left(-\frac{\lambda}{\tau^2}\right) \times \exp\left(-\frac{\text{dist}_M^2}{2\sigma^2}\right) \times P_{ij}$$

Where:

- τ = local reach (inverse maximum curvature)
- λ = curvature sensitivity parameter
- dist_M = geodesic distance on semantic manifold

**Interpretation:** Information flows efficiently through low-curvature regions (within semantic categories) and less efficiently across high-curvature boundaries (between categories).

---

## 8. Temporal Dynamics

### 8.1 Multi-Turn Context Evolution

In multi-turn interactions, context evolves over time:

$$C_{\text{ext}}(t) = f(C_{\text{ext}}(t-1), B_t)$$

Where B_t = boundary object at turn t.

**Quality Trajectory:**

$$\text{quality\_trajectory}(t) = \prod_{k=1}^{t} q(B_k)$$

Where q(B_k) ∈ [0, 1] = quality of boundary object k.

**Critical insight:** Quality is multiplicative across turns. A single low-quality exchange (q ≈ 0.3) can severely degrade the entire trajectory.

### 8.2 Self-Reinforcing Cycles

- **Positive cycle:** Good context → better responses → improved context → …
- **Negative cycle:** Poor context → confused responses → degraded context → …

This explains why early interactions disproportionately determine outcomes—they set the trajectory's initial slope.

### 8.3 Threshold-Based Management

To prevent catastrophic trajectory degradation:

```python
if gap(B_t, expected) < theta_refine:
    # Minor gap: refine and continue
    B_t_prime = refine(B_t, feedback)

elif gap(B_t, expected) > theta_reset:
    # Major gap: reset to last good checkpoint
    context = restore_checkpoint(t_checkpoint)
```

---

## 9. Zero-Propagation Principle

### 9.1 Definition

**Zero-propagation** occurs when any essential component of conveyance equals zero:

$$\text{If } W = 0 \text{ OR } R = 0 \text{ OR } H = 0 \text{ OR } P_{ij} = 0 \text{ OR } D_{\text{eff}} \to 0: \quad C_{\text{pair}} = 0 \text{ (categorical failure)}$$

### 9.2 Implications

Zero-propagation is **categorical failure**, distinct from "very poor" conveyance:

| Condition | Result | Nature |
|-----------|--------|--------|
| All components > 0 but low | Low C_pair | Degraded but possible |
| Any component = 0 | C_pair = 0 | Impossible transfer |

**Example:** A perfect lecture (W=1.0, R=1.0, H=1.0) in a language no student speaks (P_ij=0) achieves zero conveyance—not poor conveyance, but zero.

### 9.3 Dimensional Collapse as Zero-Propagation

When D_eff collapses to near-zero, effective conveyance becomes impossible even with good W, R, H, P_ij values:

$$D_{\text{eff}} < \text{threshold} \to f_{\text{dim}}(D_{\text{eff}}) \to 0 \to C_{\text{pair}} \to 0$$

This explains why memory poisoning attacks with only ~10% contamination can cause ~95% task failure—small contamination triggers dimensional collapse, which cascades to zero-propagation.

---

## 10. Falsification Criteria

A hypothesis must be falsifiable to be scientific. The Conveyance Hypothesis would be **falsified** by:

### 10.1 Strong Falsification Evidence

1. **β shows consistent positive correlation with performance across domains**
   - Current observation: r ≈ -0.92 (negative)
   - Falsifying observation: r > +0.5 replicated across tasks

2. **High-dimensional C_ext systematically outperforms low-dimensional**
   - Current hypothesis: 128–256D optimal
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

### 10.2 Weak Falsification Evidence

- Single counterexamples (might be domain-specific)
- Mixed results without clear patterns
- Inability to measure proposed constructs reliably

---

## 11. Relationship to Existing Theories

### 11.1 Shannon's Information Theory

| Shannon | Conveyance |
|---------|------------|
| Channel capacity | Agent capacity (W × R × H) |
| Noise | Protocol mismatch (1 - P_ij) |
| Encoding | Boundary object creation |
| Decoding | Integration into receiver geometry |
| Bit error rate | Dimensional collapse (β) |

**Conveyance extends Shannon** by adding semantic effectiveness to signal fidelity.

### 11.2 Rogers' Innovation Diffusion Theory

| Rogers | Conveyance |
|--------|------------|
| Adoption curves | Conveyance effectiveness over time |
| Opinion leaders | High-conveyance nodes in networks |
| Compatibility | P_ij protocol coefficient |
| Complexity | Inverse of D_eff preservation |

**Conveyance mathematizes Rogers'** qualitative descriptions of how innovations spread.

### 11.3 Kolmogorov Complexity

| Kolmogorov | Conveyance |
|------------|------------|
| Minimum description length | Optimal boundary object compression |
| Incompressibility | Essential semantic structure |
| Algorithmic probability | Transfer success probability |

**Conveyance operationalizes** complexity concepts for agent-to-agent transfer.

---

## 12. Practical Applications (If Validated)

### 12.1 AI Development

- **Optimization targets:** Maximize D_eff rather than arbitrary metrics
- **Diagnostic tools:** Detect dimensional collapse before deployment
- **Architecture guidance:** Prefer attention-only over scaffolding
- **Training monitoring:** Watch geometric health during learning

### 12.2 AI Safety

- **Alignment detection:** Misaligned agents may show asymmetric C_pair
- **Early warning:** Geometric anomalies before behavioral symptoms
- **Interpretability:** Ground-truth about what information actually transferred

### 12.3 Memory Systems

- **Poisoning detection:** Dimensional collapse indicates contamination
- **Quality maintenance:** Monitor D_eff trajectory across interactions
- **Defense mechanisms:** Reset when D_eff drops below threshold

### 12.4 Human Communication (Speculative)

If the framework validates in AI systems, it may inform:

- Educational theory (why some teaching works)
- Organizational communication (why information gets lost in hierarchies)
- Cross-cultural understanding (how meaning transforms across contexts)

---

## 13. Current Evidence Status

### 13.1 Validated (Tier 1 Evidence)

- ✓ Dimensional richness correlates positively with utility
- ✓ β anti-correlates with utility (r = -0.92)
- ✓ Attention-only architecture preserves D_eff = 34
- ✓ Boundary scaffolding collapses to D_eff = 6 (-83% loss)
- ✓ L2 normalization prevents magnitude artifacts

### 13.2 Preliminary Observations (Require Validation)

- △ Low-dimensional (128–256D) outperforms high-dimensional
- △ β ∈ [1.5, 2.0] optimal range
- △ BDH's d=256 bottleneck validates independently

### 13.3 Unvalidated (Theoretical)

- ? Bilateral asymmetry predicts misalignment
- ? Curvature-modulated conveyance
- ? Temporal amplification (T^β term)
- ? Human communication applications

### 13.4 Falsified (Revised in v3.9+)

- ✗ Boundary-first approach (v3.7–3.8) — produced anti-utility
- ✗ β as optimization target — now diagnostic only
- ✗ φ (conductance) and κ (curvature) as primary metrics — deprecated

---

## 14. Conclusion

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

$$C_{\text{pair}}(i \leftrightarrow j) = \text{Hmean}(C_{\text{out}}, C_{\text{in}}) \times f_{\text{dim}}(D_{\text{eff}}) \times P_{ij}$$

### Component Equations

$$C_{\text{out}} = \frac{W \times R \times H}{T}$$

$$C_{\text{in}} = \frac{W \times R \times H}{T}$$

$$f_{\text{dim}}(D_{\text{eff}}) = \left(\frac{D_{\text{eff}}}{D_{\text{target}}}\right)^{\alpha_{\text{dim}}}$$

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
| d_nn | 0.10–0.15 | < 0.05 or > 0.25 |
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
| v3.7–3.8 | Oct 2025 | Boundary-first approach (later falsified) |
| v3.9 | Oct 2025 | D_eff as primary metric, β inversion discovered |
| v4.0 | Nov 2025 | f_dim(D_eff) replaces C_ext^α, attention-only validated |
| v4.1 | Dec 2025 | Expanded introduction with Shannon/Weaver citations, "Why This Matters" and "Why This Is Feasible Now" sections |

---

> *"The fundamental problem of communication is that of reproducing at one point either exactly or approximately a message selected at another point."* — Claude Shannon, 1948

> *"The fundamental problem of conveyance is that of transforming at one agent a semantic structure that appropriately reorganizes another agent's internal geometry."* — The Conveyance Hypothesis, 2025

---

## References

Shannon, C. E. (1948). A mathematical theory of communication. *The Bell System Technical Journal*, 27(3):379–423.

Weaver, W. (1949). Recent contributions to the mathematical theory of communication. In Shannon, C. E. and Weaver, W., editors, *The Mathematical Theory of Communication*, pages 1–28. University of Illinois Press.
