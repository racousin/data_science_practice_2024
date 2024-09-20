import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const Training = () => {
  return (
    <Container fluid>
      <h2 id="weight-initialization">Weight Initialization Strategies</h2>

      <p>
        Proper weight initialization is crucial for training deep neural
        networks efficiently. It helps in faster convergence and can prevent
        issues like vanishing or exploding gradients.
      </p>

      <h3>Common Initialization Techniques</h3>
      <ul>
        <li>Xavier/Glorot Initialization</li>
        <li>He Initialization</li>
        <li>Orthogonal Initialization</li>
      </ul>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

# Usage
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

model.apply(xavier_init)  # or model.apply(he_init)
        `}
      />

      <h2 id="gradient-clipping">Gradient Clipping</h2>

      <p>
        Gradient clipping is a technique used to prevent exploding gradients by
        limiting the maximum value of gradients during backpropagation.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

def train_step(inputs, targets):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    return loss.item()

# Usage in training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        loss = train_step(inputs, targets)
        print(f"Epoch {epoch}, Loss: {loss}")
        `}
      />

      <h2 id="curriculum-learning">Curriculum Learning</h2>

      <p>
        Curriculum learning involves training a model on easier examples before
        moving on to more difficult ones. This can lead to better performance
        and faster convergence.
      </p>

      <h3>Implementation Example</h3>
      <CodeBlock
        language="python"
        code={`
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class CurriculumDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_size, difficulty):
        self.data = torch.randn(num_samples, input_size)
        self.labels = torch.sum(self.data * difficulty, dim=1)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def train_with_curriculum(model, num_epochs, batch_size):
    difficulties = [0.1, 0.5, 1.0]  # Increasing difficulty
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    for difficulty in difficulties:
        dataset = CurriculumDataset(1000, 10, difficulty)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            print(f"Difficulty: {difficulty}, Epoch: {epoch}, Loss: {loss.item()}")

# Usage
model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
train_with_curriculum(model, num_epochs=5, batch_size=32)
        `}
      />

      <h2 id="mixed-precision">Mixed Precision Training</h2>

      <p>
        Mixed precision training involves using lower precision arithmetic
        (e.g., float16) to speed up training and reduce memory usage, while
        maintaining model accuracy.
      </p>

      <h3>Implementation in PyTorch</h3>
      <CodeBlock
        language="python"
        code={`
import torch
from torch.cuda.amp import autocast, GradScaler

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
).cuda()

optimizer = optim.Adam(model.parameters())
scaler = GradScaler()

def train_step(inputs, targets):
    with autocast():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    
    return loss.item()

# Usage in training loop
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        loss = train_step(inputs, targets)
        print(f"Epoch {epoch}, Loss: {loss}")
        `}
      />
    </Container>
  );
};

export default Training;
