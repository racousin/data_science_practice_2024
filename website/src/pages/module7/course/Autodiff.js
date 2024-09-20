import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Autodiff = () => {
  return (
    <Container fluid>
      <h2 id="forward-reverse">Forward and Reverse Mode Autodiff</h2>
      <p>
        Automatic differentiation (autodiff) is a set of techniques to
        numerically evaluate the derivative of a function specified by a
        computer program. It is a key component in modern deep learning
        frameworks.
      </p>

      <h3>Forward Mode Autodiff</h3>
      <p>
        Forward mode computes the derivative alongside the evaluation of the
        function. It's efficient for functions with a small number of inputs
        relative to the number of outputs.
      </p>
      <BlockMath math="y = f(x) \quad \rightarrow \quad \dot{y} = f'(x) \cdot \dot{x}" />

      <h3>Reverse Mode Autodiff</h3>
      <p>
        Reverse mode (also known as backpropagation in neural networks) computes
        the gradient of a scalar-valued function with respect to its inputs.
        It's efficient for functions with a large number of inputs relative to
        the number of outputs.
      </p>
      <BlockMath math="y = f(x) \quad \rightarrow \quad \bar{x} = f'(x)^T \cdot \bar{y}" />

      <p>
        Reverse mode is typically more efficient for neural networks, as we
        usually have many parameters (inputs) and a scalar loss (output).
      </p>

      <h2 id="computational-graphs">Computational Graphs</h2>
      <p>
        Computational graphs are a way of representing a sequence of
        mathematical operations as a graph. They are fundamental to
        understanding how autodiff works in practice.
      </p>
      <p>In a computational graph:</p>
      <ul>
        <li>
          Nodes represent variables (inputs, outputs, intermediate values)
        </li>
        <li>Edges represent operations</li>
      </ul>
      <p>
        During the forward pass, we compute the values of all nodes. During the
        backward pass (in reverse mode), we compute the gradients with respect
        to each node.
      </p>

      <h2 id="pytorch-autograd">PyTorch's Autograd System</h2>
      <p>
        PyTorch's autograd package provides automatic differentiation for all
        operations on tensors. It uses a define-by-run framework, where
        backpropagation is defined by how your code is run, and every single
        iteration can be different.
      </p>

      <h3>Key Concepts in PyTorch Autograd</h3>
      <ul>
        <li>
          <code>torch.Tensor</code>: The main class for tensors in PyTorch. Set{" "}
          <code>requires_grad=True</code> to track operations for autodiff.
        </li>
        <li>
          <code>torch.autograd.Function</code>: Defines a formula for computing
          a function in the forward direction and its derivative in the backward
          direction.
        </li>
        <li>
          <code>backward()</code>: Computes the gradients of all leaves with{" "}
          <code>requires_grad=True</code>.
        </li>
      </ul>

      <h3>PyTorch Autograd Example</h3>
      <CodeBlock
        language="python"
        code={`
import torch

# Create tensors with requires_grad=True to track computations
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)

# Define a computation
z = x * y
out = z.mean()

# Compute gradients
out.backward()

# Print gradients
print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
        `}
      />

      <p>
        In this example, we create two tensors <code>x</code> and <code>y</code>{" "}
        with <code>requires_grad=True</code>. We then perform some computations
        and call <code>backward()</code> on the result. PyTorch automatically
        computes the gradients of <code>x</code> and <code>y</code>.
      </p>

      <h3>Custom Autograd Function</h3>
      <p>You can also define custom autograd functions in PyTorch:</p>
      <CodeBlock
        language="python"
        code={`
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2

# Usage
input = torch.tensor([1., 2., 3.], requires_grad=True)
output = CustomFunction.apply(input)
output.backward(torch.tensor([1., 1., 1.]))
print(f"Gradient of input: {input.grad}")
        `}
      />

      <p>
        This example demonstrates how to create a custom autograd function that
        doubles its input in the forward pass and its gradient in the backward
        pass.
      </p>
    </Container>
  );
};

export default Autodiff;
