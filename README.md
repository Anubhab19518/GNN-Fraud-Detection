# Fraud Detection using Causal Temporal Heterogeneous Graph Neural Networks (HTGNN)

## Overview

This project implements a research-oriented fraud detection pipeline on the IEEE-CIS Fraud Detection dataset using a Causal Temporal Heterogeneous Graph Neural Network (HTGNN) built with:

* PyTorch
* PyTorch Geometric (PyG)
* Transformer-based Graph Neural Networks
* Temporal Graph Learning
* Contrastive Fraud Representation Learning

The system models fraud detection as a dynamic heterogeneous graph learning problem, where:

* Transactions become graph nodes
* Cards and devices become entity nodes
* Temporal transaction relationships become causal graph edges

The core idea is:

> Fraud is not an isolated transaction anomaly.
> Fraud emerges through temporal propagation, behavioral reuse, and structural interaction patterns.

---

## Key Highlights

### Implemented Features

* Heterogeneous graph construction
* Causal temporal transaction graph
* Temporal K-nearest neighbor propagation
* TransformerConv-based HTGNN
* Temporal edge attributes
* Time-aware sampling
* Temporal contrastive learning
* Temporal memory modules
* Temporal positional encoding (Time2Vec)
* Energy-based anomaly regularization
* Hard-negative analysis
* Strict anti-leakage chronological evaluation

---

## Dataset

Dataset Used:

* IEEE-CIS Fraud Detection Dataset

Main files:

* `train_transaction.csv`
* `train_identity.csv`

The project heavily engineers both:

* transactional features
* identity/device features

before constructing the graph.

---

## Problem Formulation

Traditional fraud detection treats transactions independently.

This project instead formulates fraud detection as:

### Temporal Heterogeneous Graph Learning

Where:

* transactions interact through:

	* shared cards
	* shared devices
	* temporal reuse
	* causal propagation

This allows the model to learn:

* fraud bursts
* suspicious reuse patterns
* coordinated activity
* evolving fraud behavior

---

## Graph Construction

### Node Types

The graph contains three primary node types:

| Node Type   | Description                                            |
| ----------- | ------------------------------------------------------ |
| Transaction | Individual transaction records                         |
| Card        | Shared payment card entities                           |
| Device      | Device fingerprints constructed from identity features |

---

## Transaction Node Features

Each transaction node contains:

### Continuous Features

Examples:

* TransactionAmt
* PCA-transformed V columns
* Engineered temporal features
* Identity numerical features
* Missingness masks

Total:

* ~265 continuous features

---

### Categorical Features

Examples:

* DeviceType
* Browser
* Operating System
* Identity categories

Total:

* ~25 categorical features

These are embedded using trainable embedding layers.

---

## Device Fingerprint Engineering

A strong device fingerprint is constructed using:

* DeviceInfo
* id_30
* id_31
* id_33
* id_19
* id_20

This creates highly informative device entities that significantly improve graph quality.

---

## Edge Construction

### 1. Transaction ↔ Card Edges

Represents:

* which card was used in a transaction

Relations:

* `has_card`
* `rev_has_card`

---

### 2. Transaction ↔ Device Edges

Represents:

* which device generated the transaction

Relations:

* `has_device`
* `rev_has_device`

---

## Temporal Transaction Graph

The most important component of the project.

Transactions are connected temporally if:

* they share the same:

	* card
	* device
* AND occur within a fixed causal time window

---

## Causal Temporal Propagation

Edges flow strictly:

```text
Past Transaction → Future Transaction
```

This prevents:

* temporal leakage
* future information contamination

Reverse temporal edges were intentionally removed after experiments showed they harmed generalization.

---

## Temporal Edge Types

| Edge Type                    | Meaning                                         |
| ---------------------------- | ----------------------------------------------- |
| temporal_precedes_via_card   | Transactions linked through shared card usage   |
| temporal_precedes_via_device | Transactions linked through shared device usage |

---

## Temporal K-Nearest Neighbor Sampling

To avoid dense temporal cliques:

Only the:

* K nearest previous temporal neighbors

are connected.

Example:

* Card edges → K=5
* Device edges → K=3

This significantly improves:

* generalization
* scalability
* anti-overfitting behavior

---

## Edge Attributes

Each temporal edge contains:

| Feature            | Description                |
| ------------------ | -------------------------- |
| decay_weight       | Exponential temporal decay |
| delta_t            | Time difference            |
| transaction_amount | Source transaction amount  |

Edge dimension:

```text
EDGE_DIM = 3
```

---

## Graph Characteristics

Approximate graph size:

| Component    | Count |
| ------------ | ----- |
| Transactions | ~590K |
| Cards        | ~12K  |
| Devices      | ~26K  |

Temporal edges:

* ~367K via cards
* ~52K via devices

---

## Temporal Homophily

The graph exhibits extremely high local temporal homophily.

| Relation              | Homophily |
| --------------------- | --------- |
| Card Temporal Edges   | ~96%      |
| Device Temporal Edges | ~99%      |

This confirms:

* fraud propagates strongly through local temporal neighborhoods.

---

## Chronological Data Splitting

The dataset is split chronologically:

| Split      | Percentage |
| ---------- | ---------- |
| Train      | 70%        |
| Validation | 15%        |
| Test       | 15%        |

This ensures:

* realistic fraud evaluation
* out-of-time generalization testing
* no future leakage

---

## NeighborLoader Sampling

The graph is trained using:

```python
NeighborLoader
```

with temporal sampling enabled.

This allows:

* scalable mini-batch training
* memory-efficient execution on Colab T4 GPUs
* causal neighbor sampling

---

## Model Architecture

### Core Architecture

The model uses:

```text
HeteroConv + TransformerConv
```

instead of:

* GCN
* GraphSAGE
* HGTConv

---

## Why TransformerConv?

TransformerConv allows:

* relation-aware attention
* edge attribute usage
* temporal weighting
* stronger message propagation

while remaining compatible with:

* PyG 2.7.0
* NeighborLoader batching

---

## Main Components

### 1. Feature Projection

Projects:

* continuous features
* categorical embeddings

into a unified hidden representation.

---

### 2. Entity Embeddings

Learnable embeddings for:

* cards
* devices

---

### 3. Temporal Transformer Layers

Each graph relation uses:

```text
TransformerConv
```

with:

* edge-aware attention
* residual connections
* LayerNorm
* dropout

---

## Temporal Representation Learning Upgrades

The notebook later integrates:

### Time2Vec Encoding

Learnable temporal positional encoding.

Helps model:

* periodicity
* temporal drift
* time dynamics

---

### Temporal Memory Module

Maintains lightweight temporal memory slots to improve:

* sequential fraud reasoning
* evolving behavior tracking

---

### Relation-Specific Scaling

Different relations:

* card propagation
* device propagation

receive learnable scaling behavior.

---

## Temporal Regularization

To prevent temporal memorization:

### Temporal Edge Dropout

Randomly removes temporal edges during training.

This improves:

* robustness
* out-of-time generalization

---

## Training Strategy

### Loss Function

Main supervised loss:

```text
Focal Loss
```

Used because:

* fraud detection is highly imbalanced

---

## Additional Objectives

The notebook later experiments with:

* Temporal contrastive learning
* Temporal consistency regularization
* Energy-based anomaly losses

---

## Optimization

| Component         | Value             |
| ----------------- | ----------------- |
| Optimizer         | AdamW             |
| Scheduler         | ReduceLROnPlateau |
| Early Stopping    | Validation PR-AUC |
| Gradient Clipping | Enabled           |

---

## Evaluation Metrics

Primary metric:

# PR-AUC (Average Precision)

Because:

* fraud datasets are highly imbalanced

Secondary metrics:

* ROC-AUC
* F1-score
* Precision
* Recall
* ECE (Calibration)

---

## Key Experimental Findings

### Important Discoveries

### 1. Temporal propagation is extremely powerful

Temporal graph edges contribute more than:

* static heterogeneous semantics

---

### 2. Reverse temporal edges hurt performance

Future-to-past propagation causes:

* leakage
* overfitting
* poor generalization

---

### 3. Strong device fingerprints matter significantly

Better device identity construction dramatically improves:

* graph quality
* fraud localization

---

### 4. Dense temporal graphs overfit badly

Sparse causal KNN temporal graphs generalize much better.

---

### 5. The system enters an information-limited regime

Even after:

* temporal contrastive learning
* memory modules
* Time2Vec
* anomaly regularization

PR-AUC plateaus around:

```text
~0.45
```

This suggests:

* the graph captures local temporal propagation well
* but misses higher-order structural fraud patterns

---

## Hardware & Environment

Tested on:

* Google Colab T4 GPU

Main libraries:

* PyTorch
* PyTorch Geometric (PyG)
* Scikit-learn
* Pandas
* NumPy

---

## Current Best Results

Approximate performance:

| Metric       | Score      |
| ------------ | ---------- |
| ROC-AUC      | ~0.84–0.86 |
| PR-AUC       | ~0.40–0.45 |
| Fraud Recall | ~0.50–0.58 |

