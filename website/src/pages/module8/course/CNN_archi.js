import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const CNN = () => {
  return (
    <Container fluid>
      <h2 id="popular-models">Popular CNN Architectures</h2>

      <h3>AlexNet (2012)</h3>
      <p>
        Winner of the 2012 ImageNet competition, AlexNet marked the beginning of
        the deep learning era in computer vision.
      </p>
      <ul>
        <li>5 convolutional layers and 3 fully connected layers</li>
        <li>Introduced ReLU activation and dropout</li>
      </ul>

      <h3>VGG (2014)</h3>
      <p>VGG networks are known for their simplicity and depth.</p>
      <ul>
        <li>Uses 3x3 convolutions exclusively</li>
        <li>Popular variants: VGG16 and VGG19</li>
      </ul>

      <h3>ResNet (2015)</h3>
      <p>Introduced residual connections to train very deep networks.</p>
      <ul>
        <li>Solves vanishing gradient problem in deep networks</li>
        <li>Popular variants: ResNet50, ResNet101, ResNet152</li>
      </ul>

      <h3>Inception (2014)</h3>
      <p>
        Uses inception modules with parallel convolutions of different sizes.
      </p>
      <ul>
        <li>Efficient use of computational resources</li>
        <li>Popular variants: Inception v1 (GoogLeNet), Inception v3</li>
      </ul>

      <CodeBlock
        language="python"
        code={`
import torchvision.models as models

# Load pre-trained models
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
inception = models.inception_v3(pretrained=True)
        `}
      />

      <h2 id="transfer-learning">Transfer Learning and Fine-tuning</h2>
      <p>
        Transfer learning involves using a pre-trained model as a starting point
        for a new task. This is particularly useful when you have limited
        training data.
      </p>

      <h3>Steps for Transfer Learning:</h3>
      <ol>
        <li>Load a pre-trained model</li>
        <li>Freeze the weights of earlier layers</li>
        <li>Replace the final layer(s) with new ones for your task</li>
        <li>Train the new layers on your dataset</li>
      </ol>

      <h3>Fine-tuning:</h3>
      <p>
        After initial training, you can "fine-tune" by unfreezing some of the
        earlier layers and training the entire network with a low learning rate.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torchvision.models as models

def create_transfer_learning_model(num_classes):
    # Load pre-trained ResNet
    model = models.resnet50(pretrained=True)
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

# Create model for a new task with 10 classes
model = create_transfer_learning_model(num_classes=10)

# Train only the final layer
optimizer = torch.optim.Adam(model.fc.parameters())

# Later, for fine-tuning
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # Lower learning rate
        `}
      />
    </Container>
  );
};

export default CNN;
