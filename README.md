## Architectural Implementations

### 1. AlexNet (2012)
* **Paradigm Shift:** The birth of deep learning for computer vision, breaking away from hand-crafted features.
* **Key Innovations:** Overlapping Max Pooling, ReLU Activation, Dropout Regularization, Local Response Normalization (LRN).
* **Core Mechanics:** 
  * Replaced traditional Sigmoid or Tanh activations with **ReLU** (\(\max(0, x)\)) to mitigate vanishing gradients.
  * Utilized large convolutional filters (11 × 11 with stride 4, and 5 × 5) in early layers to extract broad spatial structures from high-resolution inputs.
  * Integrated **Dropout layers** (p=0.5) in the fully connected dense blocks to break up co-adaptation and prevent severe overfitting.

### 2. ResNet-16 (2015)
* **Paradigm Shift:** Overcame the degradation problem, proving that networks can be trained much deeper by learning residual mappings.
* **Key Innovations:** Identity Shortcut Connections (Skip Connections), Heavy reliance on Batch Normalization.
* **Core Mechanics:**
  * Uses **Identity Skip Connections** to pass the input directly to the output of a block (F(x) + x), creating an uninterrupted highway for gradient flow during backpropagation.
  * Formed by standard **Basic Blocks** containing two sequential 3 × 3 convolutional layers.
  * Placed **Batch Normalization (BN)** immediately after every convolution to stabilize training distributions and accelerate convergence.

### 3. ResNet-64 (2015)
* **Paradigm Shift:** Scaling network depth efficiently without a massive explosion in computational parameters.
* **Key Innovations:** Bottleneck Blocks, Deep Feature Hierarchy.
* **Core Mechanics:**
  * Transitions from Basic Blocks to high-efficiency **Bottleneck Blocks** (1 × 1 reduce → 3 × 3 convolve → 1 × 1 restore).
  * The 1 × 1 projections dramatically compress and restore channel dimensions, minimizing parameter count and computational footprint at an expanded depth of **64 layers**.
  * Demonstrates how identity mappings allow networks to scale deep while maintaining stable gradient flow and preventing performance plateaus.

### 4. ResNeXt (2016)
* **Paradigm Shift:** Moving beyond simply making networks deeper or wider by introducing structural split-transform-merge mechanics.
* **Key Innovations:** Aggregated Residual Transformations, Cardinality over Depth/Width.
* **Core Mechanics:**
  * Replaces standard Bottleneck paths with a multi-branch design using **Grouped Convolutions**. 
  * Introduces a new hyperparameter dimension called **Cardinality** (the number of parallel, independent paths within a single block), improving accuracy while keeping parameter counts and computational costs stable.

### 5. MobileNet V2 / V3 (2018–2019)
* **Paradigm Shift:** Redesigning convolutions from scratch to support high-performance deep learning on mobile and embedded hardware.
* **Key Innovations:** Depthwise Separable Convolutions, Inverted Residuals, Linear Bottlenecks.
* **Core Mechanics:**
  * Splits standard convolutions into a spatial phase (**Depthwise Convolution**) and a channel-mixing phase (**Pointwise 1 × 1 Convolution**), reducing computation by roughly 8x to 9x.
  * Uses **Inverted Residual Blocks** that expand features to high dimensions inside the block and compress them at the skip connection to preserve information.
  * Replaces non-linear activations with **Linear Bottlenecks** at the end of narrow layers to prevent the ReLU activation from destroying vital feature data.

### 6. SENet (Squeeze-and-Excitation Networks, 2018)
* **Paradigm Shift:** Transitioning from purely spatial modeling to explicit, dynamic channel attention modeling.
* **Key Innovations:** Global Feature Squeezing, Adaptive Channel-wise Re-weighting.
* **Core Mechanics:**
  * Introduces an architectural sub-unit that performs **Global Average Pooling** to "Squeeze" spatial dimensions into a global channel descriptor.
  * Employs an "Excitation" step using a tiny, multi-layer perceptron (MLP) bottleneck to compute a set of per-channel weights.
  * Dynamically scales the feature maps channel-by-channel, forcing the network to selectively focus on relevant features while ignoring noise.

### 7. EfficientNet V1 / V2 (2019–2021)
* **Paradigm Shift:** Replacing ad-hoc structural scaling with structured, mathematically optimized model expansion.
* **Key Innovations:** Compound Scaling, Neural Architecture Search (NAS).
* **Core Mechanics:**
  * Proves that network scaling must balance depth, width, and input resolution concurrently using a fixed **compound coefficient**.
  * Scales all three structural dimensions uniformly, unlocking state-of-the-art accuracy while keeping the parameter budget and FLOP calculations highly optimized.

### 8. ConvNeXt V1 (2022)
* **Paradigm Shift:** Modernizing classical CNNs to compete directly with Vision Transformers (ViTs) without losing convolutional inductive biases.
* **Key Innovations:** Large Kernel Convolutions, Micro-architectural Modernization.
* **Core Mechanics:**
  * Swaps early small convolutions for **Depthwise Convolutions with large kernels (7 × 7)** to emulate the expansive receptive fields of ViT self-attention blocks.
  * Replaces excessive Batch Normalizations with a sparse distribution of **Layer Normalization (LN)**.
  * Swaps out standard dense ReLU layers for fewer, more expressive **GELU** activations, adopting an inverted bottleneck ratio of 4:1.

### 9. Modern State-of-the-Art: ConvNeXt V2 & Hybrid Networks (2023–2026)
* **Paradigm Shift:** Co-designing convolutional architectures alongside self-supervised pre-training frameworks.
* **Key Innovations:** Global Response Normalization (GRN), Masked Autoencoder (MAE) Integration, Sparse Convolutions.
* **Core Mechanics:**
  * Designed to excel when trained with self-supervised masked image modeling objectives (similar to modern vision frameworks like the **Swin Transformer**).
  * Integrates **Global Response Normalization (GRN)** layers directly into the channel-mixing blocks to combat feature collapse and reduce channel competition.
  * Optimizes internal layers for **Sparse Convolutions**, allowing the network to process heavily masked, downsampled inputs highly efficiently while maintaining structural accuracy across large-scale vision benchmarks.

---

## 📊 Summary Comparison Matrix

| Architecture | Paradigm Shift | Computational Complexity | Key Layer/Mechanism |
| :--- | :--- | :--- | :--- |
| **AlexNet** | Birth of Deep Learning | High (Dense FC parameters) | Large 11 × 11 Filters, Dropout |
| **ResNet** | Solved Degradation Problem | Medium-High | Identity Skip Connections, Bottleneck Blocks |
| **MobileNet** | Extreme Mobile Efficiency | Ultra-Low | Depthwise Separable Convolutions |
| **EfficientNet** | Optimized Compound Scaling | Low-Medium | Multi-Dimension NAS Scaling |
| **ConvNeXt V1/V2** | Modernized Convolutional Layout | Medium | 7 × 7 Depthwise, LayerNorm, GRN Layers |
