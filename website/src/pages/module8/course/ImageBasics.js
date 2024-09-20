import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import CodeBlock from "components/CodeBlock";
import { InlineMath, BlockMath } from "react-katex";

const ImageBasics = () => {
  return (
    <Container fluid>
      <h2 id="representation">Image Representation in Computers</h2>
      <p>
        Digital images are typically represented as 2D or 3D arrays of pixel
        values:
      </p>
      <ul>
        <li>Grayscale images: 2D array with values representing intensity</li>
        <li>
          Color images: 3D array with channels for red, green, and blue (RGB)
        </li>
      </ul>
      <p>
        The dimensions of an image array are usually height x width x channels.
        For example, a 224x224 RGB image would be represented as a 224x224x3
        array.
      </p>

      <h2 id="color-spaces">Color Spaces (RGB, HSV, etc.)</h2>
      <h3>RGB (Red, Green, Blue)</h3>
      <p>
        The most common color space, where each pixel is represented by the
        intensity of red, green, and blue components.
      </p>
      <h3>HSV (Hue, Saturation, Value)</h3>
      <p>Represents color using:</p>
      <ul>
        <li>Hue: Color type (0-360°)</li>
        <li>Saturation: Color intensity (0-100%)</li>
        <li>Value: Brightness (0-100%)</li>
      </ul>
      <h3>Other Color Spaces</h3>
      <p>
        CMYK (Cyan, Magenta, Yellow, Key/Black), YUV, LAB, etc., each with
        specific use cases.
      </p>

      <h2 id="normalization">Image Normalization and Standardization</h2>
      <p>Normalization and standardization are crucial preprocessing steps:</p>
      <h3>Min-Max Normalization</h3>
      <BlockMath math="x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}" />
      <h3>Standardization (Z-score normalization)</h3>
      <BlockMath math="x_{standardized} = \frac{x - \mu}{\sigma}" />
      <p>
        Where <InlineMath math="\mu" /> is the mean and{" "}
        <InlineMath math="\sigma" /> is the standard deviation.
      </p>

      <CodeBlock
        language="python"
        code={`
import torch
import torchvision.transforms as transforms

# Define normalization transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Apply to a tensor image
normalized_image = normalize(image_tensor)
        `}
      />

      <h2 id="augmentation">Data Augmentation Techniques</h2>
      <p>Data augmentation helps increase the diversity of training data:</p>
      <ul>
        <li>
          Geometric transformations: Rotation, scaling, flipping, cropping
        </li>
        <li>
          Color space transformations: Brightness, contrast, saturation
          adjustments
        </li>
        <li>Noise injection: Adding random noise to images</li>
        <li>Mixing images: Techniques like mixup or CutMix</li>
      </ul>

      <CodeBlock
        language="python"
        code={`
import torchvision.transforms as transforms

# Define augmentation pipeline
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
])

# Apply to an image
augmented_image = augmentation(original_image)
        `}
      />

      <h3>Advanced Augmentation Techniques</h3>
      <p>Some advanced techniques include:</p>
      <ul>
        <li>AutoAugment: Automatically searched augmentation policies</li>
        <li>RandAugment: Simplified version of AutoAugment</li>
        <li>
          AugMix: Consistently improved robustness and uncertainty measures
        </li>
      </ul>

      <CodeBlock
        language="python"
        code={`
from torchvision.transforms import autoaugment, transforms

policy = autoaugment.AutoAugmentPolicy.IMAGENET
auto_augment = transforms.AutoAugment(policy)

augmented_image = auto_augment(original_image)
        `}
      />
    </Container>
  );
};

export default ImageBasics;
