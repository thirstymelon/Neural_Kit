<div align="center">

<img src="icon/icon.png" alt="Neural_Kit Logo" width="250" height="250" />

# Neural_Kit

**A modern, deterministic, high-performance deep learning framework written from first principles in Ada 2022.**

[![Version](https://img.shields.io/badge/version-0.2.0-92a9e1.svg)](https://github.com/thirstymelon/Neural_Kit/releases)
[![Language](https://img.shields.io/badge/language-Ada%202022-blue.svg)](https://www.adaic.org/ada-resources/standards/ada22/)
[![Build](https://img.shields.io/badge/build-Alire%20%7C%20GNAT-brightgreen.svg)](https://alire.ada.dev/)
[![Tests](https://img.shields.io/badge/tests-113%20passed-success.svg)](tests/)
[![Memory](https://img.shields.io/badge/memory-Zero--GC%20Deterministic-informational.svg)](https://thirstymelon.github.io/Neural_Kit/developer-overview.html)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache%202.0-blue.svg)](LICENSE)

[Documentation Hub](https://thirstymelon.github.io/Neural_Kit/) • [Getting Started](https://thirstymelon.github.io/Neural_Kit/getting-started.html) • [Beginner's Course](https://thirstymelon.github.io/Neural_Kit/tutorials/beginner-guide.html) • [Developer Overview](https://thirstymelon.github.io/Neural_Kit/developer-overview.html) • [API Reference](https://thirstymelon.github.io/Neural_Kit/reference.html)

</div>

---

## Introduction

Neural_Kit v0.2.0 is a pure, strongly-typed machine learning framework written entirely in modern **Ada 2022**. It combines the expressive, modular API paradigms of modern machine learning libraries with Ada's formal contract verification, zero-GC memory model, and bitwise deterministic execution.

The framework provides the complete mathematical and architectural foundation for neural computing: unconstrained vector and matrix linear algebra, dense multi-neuron layers with analytical backpropagation, numerically hardened activation functions (Sigmoid, ReLU, Tanh, Softmax), robust loss functions, and an independent PCG32 pseudo-random number generator.

### Deterministic Zero-GC Execution

Unlike Python-based frameworks that rely on dynamic runtimes and unpredictable garbage-collection cycles, Neural_Kit allocates tensors using Ada unconstrained array objects on the stack and within predictable records. Intermediate computations are reclaimed immediately upon scope exit with zero garbage-collection pauses, delivering consistent throughput for CPU training and inference pipelines.

---

## Core Capabilities

- **Dense Multi-Neuron Layers (`Neural_Kit.Layer`)**: Fully connected layers parameterized by weight matrices (Rows = output neurons, Columns = input features) and bias vectors, supporting forward propagation and analytical backpropagation.
- **Pure Linear Algebra Engine (`Neural_Kit.Math`)**: Positional dot products, matrix-vector multiplication (`y = W · x + b`), matrix-matrix multiplication, element-wise Hadamard products, transposition, scaling, and outer products.
- **Contract-Verified Dimension Safety**: Ada 2022 preconditions (`Pre => ...`) enforce geometric compatibility across all tensor operations at compile time and runtime, eliminating silent broadcasting errors before weights can be corrupted.
- **Numerically Hardened Activations**: Sigmoid with exponential saturation clamping, ReLU, zero-centered Tanh, and Softmax with maximum-value subtraction to mathematically prevent floating-point overflow.
- **Robust Loss Functions**: Binary Cross-Entropy with ε-clamping (`ε = 10⁻⁷`) against log-zero evaluations, and Mean Squared Error for regression.
- **Encapsulated Reproducible PRNG (`Neural_Kit.Random`)**: Independent generator objects powered by PCG32 (Permuted Congruential Generator XSH-RR 64/32) and Box-Muller Gaussian transforms without hidden mutable global state.
- **Standard Weight Initializers**: Built-in Xavier (Glorot) Uniform and He (Kaiming) Normal initializers to prevent exploding or vanishing gradients during deep network initialization.
- **1-Based Indexing with Positional Slicing**: Natural mathematical 1-based indexing (`1 .. Length`) while seamlessly supporting arbitrary array slice bounds via positional offset normalization.

### Multi-Neuron Dense Layer Inference

Initialize layers with configurable activations and weight initialization strategies, then execute forward inference:

```ada
with Ada.Text_IO;       use Ada.Text_IO;
with Neural_Kit.Layer;  use Neural_Kit.Layer;
with Neural_Kit.Random; use Neural_Kit.Random;
with Neural_Kit.Types;  use Neural_Kit.Types;

procedure Main is
   Gen : Generator;

   --  Dense Layer: 2 inputs -> 3 hidden neurons with ReLU
   Hidden : Layer := New_Layer
     (Input_Dimension  => 2,
      Output_Dimension => 3,
      Gen              => Gen,
      Activation       => ReLU,
      Init_Strategy    => Xavier);

   --  Dense Layer: 3 hidden inputs -> 2 output classes with Softmax
   Classifier : Layer := New_Layer
     (Input_Dimension  => 3,
      Output_Dimension => 2,
      Gen              => Gen,
      Activation       => Softmax,
      Init_Strategy    => Xavier);

   Sample : constant Vector (1 .. 2) := (1.0, 0.5);
begin
   Set_Seed (Gen, 42);

   Hidden.Forward (Sample);
   Classifier.Forward (Hidden.Output);

   Put_Line ("Class 1 Prob: " & Scalar'Image (Classifier.Output (1)));
   Put_Line ("Class 2 Prob: " & Scalar'Image (Classifier.Output (2)));
end Main;
```

---

## Comprehensive Documentation

For in-depth explanations, setup guides, step-by-step interactive lessons, architectural deep dives, and complete API specifications, explore the official documentation website:

- [**Documentation Hub**](https://thirstymelon.github.io/Neural_Kit/): Central gateway, architecture overview, quick start, and Ctrl+K search palette.
- [**See What I Built with Neural_Kit**](https://thirstymelon.github.io/Neural_Kit/examples/index.html): Interactive application showcase demonstrating real-time client-side neural inference.
- [**Vision Canvas: Optical Digit Recognizer**](https://thirstymelon.github.io/Neural_Kit/examples/mnist.html): Draw digits on an interactive $420 \times 420$ canvas with 8-way shift-invariant pattern recognition.
- [**Neural Flora: Botanical Species Classifier**](https://thirstymelon.github.io/Neural_Kit/examples/iris.html): Real-time Fisher's Iris plant classification with dynamic measurement sliders and Softmax probability distributions.
- [**Getting Started Guide**](https://thirstymelon.github.io/Neural_Kit/getting-started.html): Installation, toolchain setup with Alire, building the static library, and configuring your first project.
- [**Beginner's Course (7 Lessons)**](https://thirstymelon.github.io/Neural_Kit/tutorials/beginner-guide.html): A complete hands-on course covering scalars, vectors, linear algebra, activations, loss functions, PRNG, single-neuron AND gate training, and dense multi-neuron layers with progress tracking.
- [**Developer & Architectural Overview**](https://thirstymelon.github.io/Neural_Kit/developer-overview.html): Deep dive into the unconstrained array memory model, Design by Contract semantics, matrix conventions, and numerical stability engineering.
- [**Technical API Reference**](https://thirstymelon.github.io/Neural_Kit/reference.html): Authoritative technical specifications for all types, subprograms, contracts, and error conditions across `Types`, `Math`, `Random`, `Neuron`, and `Layer`.

*The entire documentation suite is also available offline inside [`docs/`](docs/). Simply open `docs/index.html` in any web browser.*

---

## Building and Testing with Alire

Neural_Kit is built and managed with **Alire** (the Ada Literate REpository) and the **GNAT** compiler:

```bash
# Build the core library
alr build

# Run the Iris flower classification example
cd examples/iris_classification
alr run

# Run the MNIST handwritten digit recognition example
cd examples/mnist_classifier
alr run

# Run the automated test suite (113 assertions across 8 test suites)
gprbuild -P tests/tests.gpr
./tests/bin/test_runner
```

---

## Roadmap

- [x] **v0.1.0 — Single Neuron** (Binary classification, Sigmoid, BCE, Gradient Descent, AND Gate)
- [x] **v0.2.0 — Layers and Multi-Neuron Architecture** (Dynamic vector dimensions, multiple neurons, layer abstraction, weight matrices, bias vectors, forward propagation, analytical backpropagation)
- [ ] **v0.3.0 — Multi-Layer Networks** (Network abstraction, multiple hidden layers, backpropagation, XOR classification)
- [ ] **v0.4.0 — Optimization and Activation Functions** (SGD, Momentum, Adam, ReLU, Tanh, Softmax, learning-rate configuration)
- [ ] **v0.5.0 — Tensor Engine** (Multidimensional tensors, matrix operations, broadcasting, tensor initialization)
- [ ] **v0.6.0 — Training Infrastructure** (Batch training, mini-batch training, dataset abstraction, model evaluation)
- [ ] **v1.0.0 — Stable Framework** (Stable public API, model serialization, reproducible initialization, comprehensive tests, documentation)

---

## License

Neural_Kit is open-source software licensed under the **MIT License** or **Apache License 2.0 with LLVM exception**. Refer to the [LICENSE](LICENSE) file for complete terms.
