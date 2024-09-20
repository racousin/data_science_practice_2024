import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Classification = () => {
  return (
    <Container fluid>
      <h2 id="multi-class">Multi-class and Multi-label Classification</h2>

      <h3>Multi-class Classification</h3>
      <p>
        In multi-class classification, each image belongs to exactly one of
        several mutually exclusive classes.
      </p>
      <ul>
        <li>
          Example: Classifying images of animals into cats, dogs, or birds
        </li>
        <li>Typically uses softmax activation in the final layer</li>
        <li>Loss function: Categorical Cross-Entropy</li>
      </ul>

      <h3>Multi-label Classification</h3>
      <p>
        In multi-label classification, each image can belong to multiple classes
        simultaneously.
      </p>
      <ul>
        <li>
          Example: Tagging an image with attributes like "sunny", "beach",
          "people"
        </li>
        <li>
          Typically uses sigmoid activation for each class in the final layer
        </li>
        <li>Loss function: Binary Cross-Entropy for each label</li>
      </ul>

      <h2 id="implementation">
        Implementing Image Classification with PyTorch
      </h2>

      <h3>Dataset and DataLoader</h3>
      <CodeBlock
        language="python"
        code={`
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder('path/to/train/data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Similarly for validation dataset
        `}
      />

      <h3>Model Definition</h3>
      <CodeBlock
        language="python"
        code={`
import torch.nn as nn
import torchvision.models as models

class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

model = ImageClassifier(num_classes=10)
        `}
      />

      <h3>Training Loop</h3>
      <CodeBlock
        language="python"
        code={`
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Add validation loop here
        `}
      />

      <h2 id="evaluation">Evaluation Metrics for Image Classification</h2>

      <h3>Accuracy</h3>
      <p>
        The proportion of correct predictions among the total number of cases
        examined.
      </p>
      <BlockMath math="Accuracy = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}" />

      <h3>Precision</h3>
      <p>
        The proportion of true positive predictions among all positive
        predictions.
      </p>
      <BlockMath math="Precision = \frac{\text{True Positives}}{\text{True Positives + False Positives}}" />

      <h3>Recall</h3>
      <p>
        The proportion of true positive predictions among all actual positive
        cases.
      </p>
      <BlockMath math="Recall = \frac{\text{True Positives}}{\text{True Positives + False Negatives}}" />

      <h3>F1 Score</h3>
      <p>The harmonic mean of precision and recall.</p>
      <BlockMath math="F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}" />

      <h3>Confusion Matrix</h3>
      <p>
        A table layout that allows visualization of the performance of an
        algorithm.
      </p>

      <h3>Implementation of Evaluation Metrics</h3>
      <CodeBlock
        language="python"
        code={`
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)
        `}
      />
    </Container>
  );
};

export default Classification;
