import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const ActivationFunctions = () => {
  return (
    <Container fluid>
      <h2 id="common-functions">Sigmoid, tanh, ReLU, and their variants</h2>

      <h3>Sigmoid</h3>
      <p>
        The sigmoid function squashes its input into a range between 0 and 1.
      </p>
      <BlockMath math="\sigma(x) = \frac{1}{1 + e^{-x}}" />

      <h3>Hyperbolic Tangent (tanh)</h3>
      <p>
        The tanh function is similar to sigmoid but outputs values in the range
        -1 to 1.
      </p>
      <BlockMath math="\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}" />

      <h3>Rectified Linear Unit (ReLU)</h3>
      <p>
        ReLU is currently the most popular activation function due to its
        simplicity and effectiveness.
      </p>
      <BlockMath math="ReLU(x) = \max(0, x)" />

      <h3>Leaky ReLU</h3>
      <p>
        Leaky ReLU addresses the "dying ReLU" problem by allowing a small
        gradient when the unit is not active.
      </p>
      <BlockMath math="LeakyReLU(x) = \max(0.01x, x)" />

      <h3>Exponential Linear Unit (ELU)</h3>
      <p>
        ELU uses a logarithmic curve for negative values, which can help with
        the vanishing gradient problem.
      </p>
      <BlockMath math="ELU(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}" />

      <h2 id="properties">Properties and Use Cases</h2>

      <h3>Sigmoid</h3>
      <ul>
        <li>Used in the output layer for binary classification</li>
        <li>
          Suffers from vanishing gradient problem for very large or small inputs
        </li>
      </ul>

      <h3>Tanh</h3>
      <ul>
        <li>Often used in recurrent neural networks</li>
        <li>Zero-centered, which can help in convergence</li>
      </ul>

      <h3>ReLU</h3>
      <ul>
        <li>Computationally efficient</li>
        <li>Helps mitigate the vanishing gradient problem</li>
        <li>Can suffer from "dying ReLU" problem</li>
      </ul>

      <h3>Leaky ReLU</h3>
      <ul>
        <li>Addresses the "dying ReLU" problem</li>
        <li>
          Allows for negative values, which can be beneficial in some cases
        </li>
      </ul>

      <h3>ELU</h3>
      <ul>
        <li>
          Can produce negative outputs, which can help push mean unit
          activations closer to zero
        </li>
        <li>May converge faster than ReLU in some cases</li>
      </ul>

      <h2 id="custom-activations">
        Implementing Custom Activation Functions in PyTorch
      </h2>

      <p>
        PyTorch allows for easy implementation of custom activation functions.
        Here's an example of how to implement a custom activation function:
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class CustomReLU(nn.Module):
    def __init__(self, alpha=0.1):
        super(CustomReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)

# Usage
model = nn.Sequential(
    nn.Linear(10, 20),
    CustomReLU(alpha=0.05),
    nn.Linear(20, 1)
)

# Example input
x = torch.randn(1, 10)
output = model(x)
print(output)
        `}
      />

      <p>
        In this example, we've created a custom activation function similar to
        Leaky ReLU, but with a customizable slope for negative inputs.
      </p>

      <h3>Implementing Activation Functions with Autograd</h3>

      <p>
        For more control over the forward and backward passes, you can use{" "}
        <code>torch.autograd.Function</code>:
      </p>

      <CodeBlock
        language="python"
        code={`
import torch

class CustomELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        return torch.where(input > 0, input, alpha * (torch.exp(input) - 1))

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0] = grad_input[input <= 0] * (alpha * torch.exp(input[input <= 0]))
        return grad_input, None

# Usage
input = torch.randn(5, requires_grad=True)
output = CustomELU.apply(input, torch.tensor(1.0))
output.sum().backward()
print(f"Input gradient: {input.grad}")
        `}
      />

      <p>
        This example implements a custom ELU function with a manually defined
        backward pass for more efficient gradient computation.
      </p>
    </Container>
  );
};

export default ActivationFunctions;
