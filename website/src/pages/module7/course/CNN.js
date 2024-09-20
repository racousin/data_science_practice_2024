import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const CNN = () => {
  return (
    <Container fluid>
      <h2 id="convolution">Convolution Operation and Intuition</h2>

      <p>
        Convolutional Neural Networks (CNNs) are a class of deep learning models
        particularly effective for processing grid-like data, such as images.
        The key operation in CNNs is the convolution.
      </p>

      <h3>Convolution Operation</h3>
      <p>
        In the context of CNNs, convolution is a mathematical operation that
        slides a filter (or kernel) over the input data, performing element-wise
        multiplication and summation to produce a feature map.
      </p>
      <BlockMath math="(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t-\tau)d\tau" />
      <p>In discrete terms, for a 2D image input I and a kernel K:</p>
      <BlockMath math="S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(m,n)K(i-m,j-n)" />

      <h3>Intuition</h3>
      <p>
        Convolution allows the network to learn spatial hierarchies of features.
        Lower layers might detect edges, while higher layers can recognize more
        complex patterns like textures or object parts.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 28 * 28, 10)  # Assuming 28x28 input image

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Create an instance of the model
model = SimpleCNN()

# Example usage
input_tensor = torch.randn(1, 1, 28, 28)  # (batch_size, channels, height, width)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([1, 10])
        `}
      />

      <h2 id="pooling">Pooling Layers</h2>

      <p>
        Pooling layers are used to reduce the spatial dimensions of the feature
        maps, helping to control overfitting and reduce computational
        complexity.
      </p>

      <h3>Types of Pooling</h3>
      <ul>
        <li>Max Pooling: Takes the maximum value in each pooling window</li>
        <li>Average Pooling: Takes the average value in each pooling window</li>
        <li>
          Global Pooling: Applies pooling operation across the entire feature
          map
        </li>
      </ul>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class CNNWithPooling(nn.Module):
    def __init__(self):
        super(CNNWithPooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 14 * 14, 10)  # 14x14 after pooling

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# Create an instance of the model
model = CNNWithPooling()

# Example usage
input_tensor = torch.randn(1, 1, 28, 28)
output = model(input_tensor)
print(output.shape)  # Should print torch.Size([1, 10])
        `}
      />

      <h3>Benefits of Pooling</h3>
      <ul>
        <li>Reduces the spatial dimensions of the feature maps</li>
        <li>Introduces a degree of translation invariance</li>
        <li>Helps control overfitting by reducing the number of parameters</li>
      </ul>
    </Container>
  );
};

export default CNN;
