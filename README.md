# Neural_Kit

**Neural network library written entirely in Ada.**

Neural_Kit is an experimental, CPU-based neural network library designed to explore and implement neural networks from first principles using Ada.

The project focuses on building the fundamental components of neural networks incrementally, with a simple and strongly typed API.

## Current Status

**Version: 0.1.0**

The first release implements a trainable single neuron for binary classification.

The neuron uses:

- Weighted inputs and bias
- Sigmoid activation
- Binary cross-entropy loss
- Gradient descent
- Binary classification

The current implementation is CPU-only and does not depend on external machine-learning frameworks.

## Project Structure

The library is organized into small components that will grow as the project develops.

- `Neural_Kit.Types` — Common neural-network data types.
- `Neural_Kit.Neuron` — Single-neuron implementation.
- `example/` — Separate example project demonstrating how Neural_Kit can be used.

## Example

The included example trains a single neuron to learn the **AND logical operation**.

The training process demonstrates the complete basic machine-learning workflow:

1. Feed inputs into the neuron.
2. Calculate the prediction.
3. Calculate the loss.
4. Update the weights and bias.
5. Repeat the process over multiple epochs.
6. Test the trained neuron.

The example shows that a single neuron can learn a linearly separable binary classification problem such as AND.

## Roadmap

Neural_Kit will be developed progressively from individual neurons toward larger neural-network architectures.

- [x] Single neuron
- [x] Binary classification
- [x] Sigmoid activation
- [x] Binary cross-entropy
- [x] Gradient descent
- [ ] Multiple neurons
- [ ] Layers
- [ ] Neural networks
- [ ] Backpropagation
- [ ] Optimizers
- [ ] Tensors
- [ ] Advanced neural-network architectures

## Goals

The long-term goal of Neural_Kit is to provide a native Ada foundation for building and experimenting with neural networks, while keeping the implementation understandable, strongly typed, and suitable for high-performance CPU computation.

## License

MIT OR Apache-2.0 WITH LLVM-exception
