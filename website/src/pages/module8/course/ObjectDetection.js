import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const ObjectDetection = () => {
  return (
    <Container fluid>
      <h2 id="rcnn">Region-based CNNs (R-CNN, Fast R-CNN, Faster R-CNN)</h2>

      <h3>R-CNN (Regions with CNN features)</h3>
      <p>
        R-CNN was one of the first deep learning-based object detection models.
      </p>
      <ul>
        <li>Uses selective search to generate region proposals</li>
        <li>Applies CNN to each proposed region</li>
        <li>
          Uses SVM for classification and linear regression for bounding box
          refinement
        </li>
      </ul>

      <h3>Fast R-CNN</h3>
      <p>
        Fast R-CNN improved upon R-CNN by processing the entire image once
        through a CNN.
      </p>
      <ul>
        <li>
          Uses RoI (Region of Interest) pooling to extract features for each
          proposal
        </li>
        <li>Trains classification and bounding box regression jointly</li>
        <li>Much faster than R-CNN during both training and inference</li>
      </ul>

      <h3>Faster R-CNN</h3>
      <p>
        Faster R-CNN introduced the Region Proposal Network (RPN) for generating
        region proposals.
      </p>
      <ul>
        <li>
          RPN shares full-image convolutional features with the detection
          network
        </li>
        <li>Nearly cost-free region proposals</li>
        <li>End-to-end training for object detection</li>
      </ul>

      <h2 id="ssd-yolo">Single Shot Detectors (SSD, YOLO)</h2>

      <h3>Single Shot Detector (SSD)</h3>
      <p>
        SSD is a single-stage detector that performs object localization and
        classification in a single forward pass.
      </p>
      <ul>
        <li>
          Uses a set of default bounding boxes over different scales and aspect
          ratios
        </li>
        <li>
          Applies classification and bounding box regression on these default
          boxes
        </li>
        <li>Fast and accurate, especially for small objects</li>
      </ul>

      <h3>You Only Look Once (YOLO)</h3>
      <p>
        YOLO divides the image into a grid and predicts bounding boxes and class
        probabilities for each grid cell.
      </p>
      <ul>
        <li>Extremely fast, capable of real-time object detection</li>
        <li>
          Reasons globally about the image, leading to fewer background errors
        </li>
        <li>
          Has gone through several iterations (YOLOv2, YOLOv3, YOLOv4, YOLOv5)
        </li>
      </ul>

      <h2 id="implementation">Implementing Object Detection with PyTorch</h2>

      <h3>Using a pre-trained Faster R-CNN model</h3>
      <CodeBlock
        language="python"
        code={`
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess image
image = Image.open("path_to_your_image.jpg")
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
image_tensor = transform(image)

# Perform inference
with torch.no_grad():
    prediction = model([image_tensor])

# Extract bounding boxes, labels and scores
boxes = prediction[0]['boxes'].cpu().numpy()
labels = prediction[0]['labels'].cpu().numpy()
scores = prediction[0]['scores'].cpu().numpy()

# Visualize results
plt.imshow(image)
ax = plt.gca()

for box, label, score in zip(boxes, labels, scores):
    if score > 0.5:  # Confidence threshold
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             fill=False, color='red')
        ax.add_patch(rect)
        ax.text(box[0], box[1], f"{label}: {score:.2f}", bbox=dict(facecolor='white', alpha=0.5))

plt.axis('off')
plt.show()
        `}
      />

      <h3>Training a custom object detector</h3>
      <p>
        Training a custom object detector involves preparing a dataset with
        bounding box annotations, defining a model (e.g., Faster R-CNN), and
        training it on your data. Here's a simplified example:
      </p>
      <CodeBlock
        language="python"
        code={`
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Assume you have a custom dataset class 'ObjectDetectionDataset'
train_dataset = ObjectDetectionDataset('path/to/train/data')
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = get_model(num_classes=your_num_classes)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}')

# Remember to add validation and model saving
        `}
      />
    </Container>
  );
};

export default ObjectDetection;
