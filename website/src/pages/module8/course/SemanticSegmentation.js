import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const SemanticSegmentation = () => {
  return (
    <Container fluid>
      <h2 id="fcn">Fully Convolutional Networks (FCN)</h2>

      <p>
        Fully Convolutional Networks (FCN) were a breakthrough in semantic
        segmentation, adapting classification networks to output spatial maps
        instead of classification scores.
      </p>

      <h3>Key Concepts of FCN:</h3>
      <ul>
        <li>Replace fully connected layers with convolutional layers</li>
        <li>Use transposed convolutions for upsampling</li>
        <li>
          Introduce skip connections to combine coarse, high layer information
          with fine, low layer information
        </li>
      </ul>

      <p>The FCN architecture can be represented as:</p>
      <BlockMath math="FCN = Encoder_{CNN} + Decoder_{TransposedConv}" />

      <h3>Types of FCN:</h3>
      <ul>
        <li>FCN-32s: Single stride-32 prediction</li>
        <li>FCN-16s: Fusing predictions from pool4 and pool5 layers</li>
        <li>FCN-8s: Further fusing predictions from pool3</li>
      </ul>

      <h2 id="unet">U-Net Architecture</h2>

      <p>
        U-Net is a popular architecture for semantic segmentation, especially in
        biomedical image segmentation.
      </p>

      <h3>Key Features of U-Net:</h3>
      <ul>
        <li>
          Symmetric architecture with a contracting path and an expansive path
        </li>
        <li>
          Skip connections between corresponding layers in the contracting and
          expansive paths
        </li>
        <li>Capable of learning from relatively small training sets</li>
      </ul>

      <p>The U-Net architecture can be conceptualized as:</p>
      <BlockMath math="U-Net = Encoder_{Contracting} + Decoder_{Expanding} + SkipConnections" />

      <h2 id="implementation">
        Implementing Semantic Segmentation with PyTorch
      </h2>

      <h3>Implementing a Simple U-Net Model:</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.encoder = nn.ModuleList([
            DoubleConv(3, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
        ])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(512, 1024)
        self.decoder = nn.ModuleList([
            DoubleConv(1024, 512),
            DoubleConv(512, 256),
            DoubleConv(256, 128),
            DoubleConv(128, 64)
        ])
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        ])
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for up, skip, dec in zip(self.upconv, skip_connections, self.decoder):
            x = up(x)
            x = torch.cat((skip, x), dim=1)
            x = dec(x)
        return self.final_conv(x)

# Initialize the model
model = UNet(num_classes=10)  # Assuming 10 classes for segmentation
        `}
      />

      <h3>Training Loop for Semantic Segmentation:</h3>
      <CodeBlock
        language="python"
        code={`
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Assume you have a custom dataset class 'SegmentationDataset'
train_dataset = SegmentationDataset('path/to/train/data', transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

model = UNet(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Remember to add validation and model saving
        `}
      />

      <h3>Evaluation Metrics for Semantic Segmentation:</h3>
      <ul>
        <li>Pixel Accuracy: Percentage of pixels correctly classified</li>
        <li>
          Mean Intersection over Union (mIoU): Average IoU across all classes
        </li>
        <li>
          Frequency Weighted Intersection over Union (FWIoU): Weighted version
          of mIoU based on pixel frequency of each class
        </li>
      </ul>

      <CodeBlock
        language="python"
        code={`
import numpy as np

def calculate_iou(pred_mask, gt_mask, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = pred_mask == cls
        gt_inds = gt_mask == cls
        intersection = np.logical_and(pred_inds, gt_inds).sum()
        union = np.logical_or(pred_inds, gt_inds).sum()
        iou = intersection / (union + 1e-10)
        ious.append(iou)
    return np.array(ious)

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    total_ious = []
    pixel_acc = 0
    total_pixels = 0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()
            
            total_ious.append(calculate_iou(preds, masks, num_classes))
            pixel_acc += (preds == masks).sum()
            total_pixels += preds.size
    
    mean_ious = np.mean(total_ious, axis=0)
    mean_iou = np.mean(mean_ious)
    pixel_acc = pixel_acc / total_pixels
    
    return mean_iou, pixel_acc, mean_ious

# Usage
mean_iou, pixel_acc, class_ious = evaluate(model, val_loader, device, num_classes=10)
print(f'Mean IoU: {mean_iou:.4f}, Pixel Accuracy: {pixel_acc:.4f}')
for cls, iou in enumerate(class_ious):
    print(f'Class {cls} IoU: {iou:.4f}')
        `}
      />

      <p>
        This implementation covers the basics of semantic segmentation with
        U-Net. In practice, you'd need to handle data loading, augmentation, and
        implement a more robust training and evaluation pipeline. Additionally,
        consider using more advanced architectures like DeepLab or PSPNet for
        state-of-the-art performance.
      </p>
    </Container>
  );
};

export default SemanticSegmentation;
