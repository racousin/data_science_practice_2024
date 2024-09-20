import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Backpropagation = () => {
  return (
    <Container fluid>
      <h2 id="chain-rule">Chain Rule and Gradient Descent</h2>
      <p>
        The chain rule is a fundamental concept in calculus that allows us to
        compute the derivative of composite functions. In the context of neural
        networks, it's crucial for calculating gradients during backpropagation.
      </p>
      <BlockMath math="\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)" />
      <p>
        Gradient descent is an optimization algorithm used to minimize the loss
        function by iteratively moving in the direction of steepest descent.
      </p>
      <BlockMath math="\theta = \theta - \alpha \nabla_\theta J(\theta)" />
      <p>
        Where <InlineMath math="\theta" /> represents the model parameters,{" "}
        <InlineMath math="\alpha" /> is the learning rate, and{" "}
        <InlineMath math="J(\theta)" /> is the loss function.
      </p>

      <h2 id="backpropagation-algorithm">Backpropagation Algorithm</h2>
      <p>
        Backpropagation is an efficient algorithm for computing gradients in
        neural networks. It consists of two main phases:
      </p>
      <ol>
        <li>Forward pass: Compute the output and loss</li>
        <li>Backward pass: Compute gradients and update weights</li>
      </ol>
      <p>
        The key idea is to use the chain rule to efficiently compute gradients
        for all parameters in the network.
      </p>

      <h2 id="sgd">Stochastic Gradient Descent (SGD) and its Variants</h2>
      <p>
        SGD is an iterative method for optimizing an objective function with
        suitable smoothness properties. Unlike standard gradient descent, SGD
        uses only a subset of the data (a mini-batch) to compute the gradient at
        each iteration.
      </p>
      <h3>Variants of SGD:</h3>
      <ul>
        <li>SGD with momentum</li>
        <li>Nesterov Accelerated Gradient</li>
        <li>AdaGrad</li>
        <li>RMSprop</li>
      </ul>

      <h2 id="optimizers">Learning Rate Schedules and Adaptive Optimizers</h2>
      <h3>Learning Rate Schedules</h3>
      <p>
        Learning rate schedules adjust the learning rate during training. Common
        approaches include:
      </p>
      <ul>
        <li>Step decay</li>
        <li>Exponential decay</li>
        <li>Cosine annealing</li>
      </ul>

      <h3>Adaptive Optimizers</h3>
      <p>
        Adaptive optimizers automatically adjust the learning rates of each
        parameter. Two popular adaptive optimizers are:
      </p>
      <h4>Adam (Adaptive Moment Estimation)</h4>
      <p>Adam combines ideas from RMSprop and momentum optimization.</p>
      <h4>RMSprop</h4>
      <p>
        RMSprop adapts the learning rates by dividing by an exponentially
        decaying average of squared gradients.
      </p>

      <h3>PyTorch Implementation Example</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Define loss function
criterion = nn.MSELoss()

# Define optimizer (Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# In the training loop:
scheduler.step()
        `}
      />
    </Container>
  );
};

export default Backpropagation;
