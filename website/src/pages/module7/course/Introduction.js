import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";

const Introduction = () => {
  return (
    <Container fluid>
      <h2 id="biological-inspiration">Biological Inspiration and History</h2>
      <p>
        Deep learning is a subset of machine learning inspired by the structure
        and function of the human brain. The concept of artificial neural
        networks, which form the basis of deep learning, dates back to the 1940s
        with the work of Warren McCulloch and Walter Pitts.
      </p>
      <p>Key milestones in the history of deep learning include:</p>
      <ul>
        <li>1958: The Perceptron, developed by Frank Rosenblatt</li>
        <li>
          1986: Backpropagation algorithm popularized by Rumelhart, Hinton, and
          Williams
        </li>
        <li>2006: Deep Belief Networks introduced by Geoffrey Hinton</li>
        <li>
          2012: AlexNet wins ImageNet competition, sparking the deep learning
          revolution
        </li>
      </ul>

      <h2 id="basic-components">
        Basic Components: Neurons, Layers, Activation Functions
      </h2>
      <h3>Neurons</h3>
      <p>
        Artificial neurons, inspired by biological neurons, are the basic units
        of computation in neural networks. Each neuron receives inputs, applies
        weights to these inputs, sums them up, and then passes the result
        through an activation function.
      </p>
      <h3>Layers</h3>
      <p>Neural networks are organized into layers:</p>
      <ul>
        <li>Input Layer: Receives the initial data</li>
        <li>Hidden Layers: Perform intermediate computations</li>
        <li>Output Layer: Produces the final result</li>
      </ul>
      <h3>Activation Functions</h3>
      <p>
        Activation functions introduce non-linearity into the network, allowing
        it to learn complex patterns. Common activation functions include:
      </p>
      <ul>
        <li>Sigmoid</li>
        <li>Hyperbolic Tangent (tanh)</li>
        <li>Rectified Linear Unit (ReLU)</li>
      </ul>

      <h2 id="feedforward">Feedforward Neural Networks</h2>
      <p>
        Feedforward neural networks are the simplest form of artificial neural
        networks. In these networks, information flows in one direction, from
        the input layer through the hidden layers to the output layer, without
        any cycles or loops.
      </p>
      <h3>Basic Structure of a Feedforward Network</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class FeedforwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedforwardNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# Example usage
model = FeedforwardNN(input_size=10, hidden_size=20, output_size=1)
input_data = torch.randn(1, 10)
output = model(input_data)
print(output)
        `}
      />
      <p>
        This example demonstrates a simple feedforward neural network with one
        hidden layer, using PyTorch. The network takes an input of size 10, has
        a hidden layer of size 20, and produces an output of size 1.
      </p>
    </Container>
  );
};

export default Introduction;
